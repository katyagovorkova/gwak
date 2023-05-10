import os
import h5py
import argparse
import bilby
import numpy as np
import scipy.signal as sig
from scipy.stats import cosine as cosine_distribution
from gwpy.timeseries import TimeSeries
from typing import Callable
from libs.datagen.anomaly.datagen.waveforms import olib_time_domain_sine_gaussian


FS = 4096
SEGMENT_LENGTH = 5 #seconds
IFOS = ['H1', 'L1']

def find_h5(path):
    h5_file = None
    if not  os.path.exists(path): return None
    for file in os.listdir(path):
        if file[-3:] == ".h5":
            assert h5_file is None #make sure only 1 h5 file
            h5_file = path + "/" + file

    assert h5_file is not None #did not find h5 file
    return h5_file

def load_folder(path:str, IFOS:list[str]):
    '''
    load the glitch times and data associated with a "save" folder
    '''

    if path[-1] == "/": path = path[:-1] #hopefully there aren't two...
    folder_name = path.split("/")[-1]
    assert len(folder_name.split("_")) == 2
    start = int(folder_name.split("_")[0])
    end = int(folder_name.split("_")[1])
    #print("start", start)

    for ifo in IFOS:
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

    loaded_data = dict()
    for ifo in IFOS:
        # get the glitch times first
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

        with h5py.File(h5_file, "r") as f:
            print('loaded data from h5', h5_file)
            triggers = f['triggers'][:]

        with h5py.File(f"{path}/detec_data_{ifo}.h5", "r") as f:
            X = f['ts'][:]

        # some statistics on the data
        data_statistics=0
        if data_statistics:
            print(f"start: {start}, end: {end}")
            print(f"duration, seconds: {end-start}")
            print(f"data length: {len(X)}")
            print(f"with data sampled at {4*4096}, len(data)/duration= {len(X)/4/4096}")

        sample_rate = 4*4096
        resample_rate = 4096 # don't need so many samples

        data = TimeSeries(X, sample_rate = sample_rate, t0 = start)
        if data_statistics:
            print(f"after creating time series, len(data) = {len(data)}")
            before_resample = len(data)

        if sample_rate != resample_rate:
            data = data.resample(resample_rate)

        if data_statistics:
            print(f"after resampling, len(data) = {len(data)}")
            print(f"ratio before to after: {before_resample/len(data)}")

        fftlength = 1
        loaded_data[ifo] = {'triggers': triggers,
                            'data': data,
                            'asd': data.asd(
                                fftlength=fftlength,
                                overlap=0,
                                method='welch',
                                window='hanning')}

    return loaded_data

def get_bkg_segs(loaded_data, data, N, segment_length):
    # note - by here, N samples have NOT been drawn
    quiet_times_H1 = get_quiet_segments(loaded_data["H1"], N, segment_length)
    quiet_times_L1 = get_quiet_segments(loaded_data["L1"], N, segment_length)

    quiet_times = np.intersect1d(quiet_times_H1, quiet_times_L1)
    N = min(N, len(quiet_times))
    quiet_times = np.random.choice(quiet_times, N, replace=False)

    # passing loaded_data here for reference to values like t0 and fs
    bkg_segs, _ = slice_bkg_segments(loaded_data['H1'], data, quiet_times,
                                    segment_length)
    return bkg_segs

def inject_waveforms(waveform:Callable,
                    bkg_segs_2d:np.ndarray,
                    sample_rate:int,
                    t0:int,
                    segment_length:int,
                    prior_file,
                    waveform_arguments=None,
                    domain:str='time',
                    center_type:str=None):
    N_datapoints = segment_length * sample_rate

    if domain == 'time':
        waveform_generator = bilby.gw.WaveformGenerator(
                duration=segment_length,
                sampling_frequency=sample_rate,
                time_domain_source_model=waveform,
                waveform_arguments=waveform_arguments
            )
    else:
        assert domain in ['freq', 'frequency']
        waveform_generator = bilby.gw.WaveformGenerator(
                duration=segment_length,
                sampling_frequency=sample_rate,
                frequency_domain_source_model=waveform,
                waveform_arguments=waveform_arguments
            )

    # now sample params

    # support passing in either the name of the file or the actual prior
    if type(prior_file) == str:
        priors = bilby.gw.prior.PriorDict(prior_file)
    else:
        priors = prior_file
    injection_parameters = priors.sample(bkg_segs_2d.shape[1])

    # reshaping, code taken from https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/injection.py
    # lns 107-110
    injection_parameters = [
            dict(zip(injection_parameters, col))
            for col in zip(*injection_parameters.values())
        ]

    final_injects = []
    for i, ifo in enumerate(IFOS):
        bkg_segs = bkg_segs_2d[i]
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
                    ifo
                )
        # make full seg, basically a segment with all the bkg_segs appended together
        print('BKG SEG SHAPE', bkg_segs.shape, 'hstack', np.hstack(bkg_segs).shape)
        full_seg = TimeSeries(np.hstack(bkg_segs), sample_rate=sample_rate, t0 = 0)

        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

        for i, p in enumerate(injection_parameters):
            p['luminosity_distance']=np.random.uniform(50, 200)
            ra = p['ra']
            dec = p['dec']
            start_time = (i * segment_length)*sample_rate # start of the segment to be injected into
            geocent_time = t0 + start_time
            p['geocent_time'] = 0 * sample_rate * segment_length / 2 # center the injection in the middle of the signal
            psi = p['psi']

            # get hplus, hcross
            polarizations = waveform_generator.time_domain_strain(p)

            injection = np.zeros(len(full_seg))
            # loop over polarizaitions applying detector response
            for mode, polarization in polarizations.items():
                response = bilby_ifo.antenna_response( # this gives just a float, my interpretation is the
                    ra, dec, geocent_time, psi, mode)  # inclination angle of the detector to the source

                # approximate tricks for finding center and doing injection based on that
                if center_type == None: #just none for the sine gaussian models
                    centre = np.argmax(np.abs(polarization)) #this works for sine gaussian but can be problematic for other ones
                    # want center to be halfway through the segment, at least for now
                    a = start_time + N_datapoints//2 - centre
                    # truncate so it doesn't go over the edge, should be ok since all zeros
                    inj_slice = slice(a, a + centre*2)
                    injection[inj_slice] += response * polarization[:centre*2]
                elif center_type in ['BBH', 'bbh']:
                    # better idea, put the end of the BBH at the middle? that way clipping isn'ty an issue
                    injection[start_time:start_time+segment_length*sample_rate//2] += response*polarization[-(segment_length*sample_rate//2):]

            signal = TimeSeries(injection, times = full_seg.times, unit=full_seg.unit)
            # maybe need to do the shifting stuff here
            full_seg = full_seg.inject(signal)

        # now need to chop back up the full_seg
        injected_bkgs = np.array(full_seg).reshape(bkg_segs.shape)
        final_injects.append(injected_bkgs)


    final_bothdetector = np.stack(final_injects)
    print("FINAL INJECT SHAPE", final_bothdetector.shape)

    return final_bothdetector

def clipping(seg:TimeSeries, sample_rate:int, clip_edge:int=1):
    clip_edge_datapoints = int(sample_rate * clip_edge)
    return seg[clip_edge_datapoints:-clip_edge_datapoints]

def whiten_bkgs(bkg_segs_full:np.ndarray, sample_rate:int, H1_asd, L1_asd):
    clip_edge=1
    ASDs = {"H1":H1_asd,
                "L1":L1_asd}
    all_white_segs = []
    print("into whiten, full:", bkg_segs_full.shape)
    for i, ifo in enumerate(["H1", "L1"]):
        bkg_segs = bkg_segs_full[i]
        print("in whiten, ", bkg_segs.shape)
        final_shape = (bkg_segs.shape[0], bkg_segs.shape[1]-2*int(clip_edge*sample_rate))
        white_segs = np.zeros(final_shape)
        for i, bkg_seg in enumerate(bkg_segs):
            white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(asd=ASDs[ifo])
            white_segs[i] = clipping(white_seg, sample_rate, clip_edge=clip_edge)
        all_white_segs.append(white_segs)

    # have to do clipping because of edge effects when whitening? check this...yes! have to
    FINAL_WHITEN = np.stack(all_white_segs)
    print("FINAL WHITEN", FINAL_WHITEN.shape)
    return FINAL_WHITEN

def main(args):

    loaded_data = load_folder(args.folder_path, IFOS)
    if loaded_data == None:
        print("aborting due to missing data streams")
        return None

    # stack the loaded data into 2-d np array (2, N)
    data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    if signal_type=='bbh':
        # Injection
        assert loaded_data['H1']['data'].t0.value == loaded_data['L1']['data'].t0.value
        bkg_segs = get_bkg_segs(loaded_data, data, N, SEGMENT_LENGTH)

        BBH_waveform_args = dict(
            waveform_approximant='IMRPhenomPv2',
            reference_frequency=50.,
            minimum_frequency=20.)

        # BBH injection
        injected_segs = inject_waveforms(bilby.gw.source.lal_binary_black_hole,
            bkg_segs, FS, loaded_data['H1']['data'].t0.value ,SEGMENT_LENGTH,
            prior_file=bilby.gw.prior.BBHPriorDict(),
            waveform_arguments=BBH_waveform_args,
            domain='freq',
            center_type='BBH')

        # need both ASDs here
        whitened_segs = whiten_bkgs(injected_segs, FS, loaded_data['H1']['asd'], loaded_data['L1']['asd'])

        np.save(args.output, whitened_segs)

    elif signal_type=='sg':
        bkg_segs = get_bkg_segs(loaded_data, data, N, SEGMENT_LENGTH)
        injected_segs = inject_waveforms(olib_time_domain_sine_gaussian,
                bkg_segs, FS, loaded_data['H1']['data'].t0.value ,SEGMENT_LENGTH, prior_file)

        whitened_segs = whiten_bkgs(injected_segs, FS, loaded_data['H1']['asd'], loaded_data['L1']['asd'])

        np.save(args.output, whitened_segs)

    elif signal_type=='wnb':
        bkg_segs = get_bkg_segs(loaded_data, data, N, SEGMENT_LENGTH)
        injected_segs = inject_waveforms(olib_time_domain_sine_gaussian,
                bkg_segs, FS, loaded_data['H1']['data'].t0.value ,SEGMENT_LENGTH, prior_file)

        whitened_segs = whiten_bkgs(injected_segs, FS, loaded_data['H1']['asd'], loaded_data['L1']['asd'])

        np.save(args.output, whitened_segs)

    elif signal_type=='background':
        bkg_segs = get_bkg_segs(loaded_data, data, N, SEGMENT_LENGTH)

        whitened_segs = whiten_bkgs(bkg_segs, FS, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
        np.save(args.output, whitened_segs)

    elif signal_type=='glitch':
        loud_times_H1 = get_loud_segments(loaded_data['H1'], N, SEGMENT_LENGTH)
        loud_times_L1 = get_loud_segments(loaded_data['H1'], N, SEGMENT_LENGTH)

        loud_times = np.union1d(loud_times_H1, loud_times_L1)

        glitch_segs, _ = slice_bkg_segments(loaded_data['H1'], data, loud_times,
                                        SEGMENT_LENGTH)

        whitened_segs = whiten_bkgs(glitch_segs, FS, loaded_data['H1']['asd'], loaded_data['L1']['asd'])

        np.save(args.output, whitened_segs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('output', help='Where to save the injections',
        type=str)
    parser.add_argument('signal-type', help='Which injection to generate?',
        type=str,
        choices=['bbh','sg','background','glitch','wnb'])

    parser.add_argument('--N', help='Some parameter',
        type=int, default=20)
    parser.add_argument('--folder-path', help='Path to the raw detector data',
        type=str,
        default="Users/katya/Library/CloudStorage/GoogleDrive-likemetooo@gmail.com/.shortcut-targets-by-id/1meSQdJObNYt4y3CtpG1NzKbDVM-ESyTg/1240624412_1240654372/")
    parser.add_argument('--prior-file', help='Path to the prior file for SG injection',
        type=str,
        default="/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/prior.prior")

    args = parser.parse_args()
    main(args)