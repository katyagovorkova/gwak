import os
import h5py
import bilby
import numpy as np
from gwpy.timeseries import TimeSeries
from typing import Callable
from anomaly.datagen.waveforms import olib_time_domain_sine_gaussian

'''
some code taken from Ethan Marx
https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/injection.py
'''
def find_h5(path):
    h5_file = None
    if not  os.path.exists(path): return None
    for file in os.listdir(path):
        if file[-3:] == ".h5":
            assert h5_file is None #make sure only 1 h5 file 
            h5_file = path + "/" + file

    assert h5_file is not None #did not find h5 file
    return h5_file

def load_folder(path:str, ifos:list[str]):
    '''
    load the glitch times and data associated with a "save" folder
    '''
    #if len(os.listdir(path)) != 6: #check that all files are in place
    #    return None

    if path[-1] == "/": path = path[:-1] #hopefully there aren't two...
    folder_name = path.split("/")[-1]
    assert len(folder_name.split("_")) == 2
    start = int(folder_name.split("_")[0])
    end = int(folder_name.split("_")[1])
    #print("start", start)

    loaded_data = dict()
    for ifo in ifos:
        #get the glitch times first
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

        with h5py.File(h5_file, "r") as f:
            triggers = f['triggers'][:]

        with h5py.File(f"{path}/detec_data_{ifo}.h5", "r") as f:
            X = f['ts'][:]

        #some statistics on the data
        data_statistics=0
        if data_statistics:
            print(f"start: {start}, end: {end}")
            print(f"duration, seconds: {end-start}")
            print(f"data length: {len(X)}")
            print(f"with data sampled at {4*4096}, len(data)/duration= {len(X)/4/4096}")


        sample_rate = 4*4096
        resample_rate = 4096 #don't need so many samples

        data = TimeSeries(X, sample_rate = sample_rate, t0 = start)
        if data_statistics:
            print(f"after creating time series, len(data) = {len(data)}")
            before_resample = len(data)
        
         
        if sample_rate != resample_rate:
            data = data.resample(resample_rate)

        if data_statistics:
            print(f"after resampling, len(data) = {len(data)}")
            print(f"ratio before to after: {before_resample/len(data)}")

        fftlength=1
        loaded_data[ifo] = {"triggers": triggers, 
                            "data":data, 
                            "asd":data.asd(fftlength=fftlength,overlap=0, method='welch', window='hanning')}

    return loaded_data

def get_loud_segments(ifo_data:dict, N:int, segment_length:float):
    '''
    sort the glitches by SNR, and return the N loudest ones
    If there are < N glitches, return the number of glitches, i.e. will not crash
    '''
    glitch_snr_bar = 10

    glitch_times = ifo_data["triggers"]['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data'])/ifo_data['data'].sample_rate.value

    glitch_snrs = ifo_data["triggers"]['snr']
    sort_by_snr = glitch_snrs.argsort()[::-1] #create a sorting, in descending order
    #print(glitch_snrs[sort_by_snr])
    glitch_times_sorted = glitch_times[sort_by_snr]

    glitch_start_times = []
    for i in range(min(len(glitch_times_sorted), N)):
        #going to set a glitch SNR bar because the really quiet ones don't really do much
        if glitch_snrs[sort_by_snr][i] < glitch_snr_bar:
            continue
        #want times for the beginning of the segment, and lets have the glitch centered
        gt = glitch_times_sorted[i]
        glitch_start_times.append(gt-segment_length/2)

    return glitch_start_times

def get_quiet_segments(ifo_data:dict, N:int, segment_length:float):
    '''
    get N times that are away from the glitches
    '''
    edge = segment_length

    glitch_times = ifo_data["triggers"]['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data'])/ifo_data['data'].sample_rate.value
    
    valid_start_times = np.arange(t0, tend-segment_length, segment_length)
    open_times = np.ones(valid_start_times.shape)

    #print("t0", t0)
    #print("valid_start_times", valid_start_times)
    for gt in glitch_times:
        #print("searching", gt, "into", valid_start_times)
        idx = np.searchsorted(valid_start_times, gt)
        #print("got result", idx)
        #print([float(elem) for elem in valid_start_times[idx-1:idx+2]], gt)
        #get rid of 3 indicies
        for i in range(idx-2, idx+2):
           #print(idx)
            try:
                open_times[i] = 0
            except IndexError:
                #print("something happened out of bounds with", idx)
                None #this is dangerous, just using it for now to deal with glitches on the edge

    total_available = np.sum(open_times)
    #assert total_available >= N #if this fails, then need to revise choosing strategy
    if total_available < N: #manually setting the maximuim
        N = int(total_available)

    #convert to bool mask
    open_times = (open_times != 0)
    valid_start_times = valid_start_times[open_times]

    #now just take from valid start times without replacement

    quiet_times = np.random.choice(valid_start_times, N, replace=False)

    #one last check
    for elem in quiet_times:

        assert np.abs(glitch_times-elem).min() >= segment_length
        assert np.abs(glitch_times-(elem+segment_length)).min() >= segment_length
    return quiet_times
    
def slice_bkg_segments(ifo_data:dict, start_times:list[float], segment_length:float):
    #print("got for start times,", start_times)
    #turn the start times into background segments
    fs = ifo_data['data'].sample_rate.value
    t0 = ifo_data['data'].t0.value
    N_datapoints = int(segment_length * fs)
    bkg_segs = np.zeros(shape=(len(start_times), N_datapoints))
    bkg_timeseries = []

    for i in range(len(start_times)):
        slx = slice( int((start_times[i]-t0)*fs),  int((start_times[i]-t0)*fs)+N_datapoints)
        slice_segment = ifo_data['data'][slx]
        assert np.isclose(slice_segment.t0.value, start_times[i]) #check that timing lines up
        bkg_segs[i] = np.array(slice_segment)
        bkg_timeseries.append(slice_segment)

    return bkg_segs, bkg_timeseries


def inject_waveforms(waveform:Callable,
                    bkg_segs:np.ndarray,
                    sample_rate:int,
                    t0:int,
                    segment_length:int,
                    prior_file,
                    ifo:str,
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
        assert domain in ["freq", "frequency"]
        waveform_generator = bilby.gw.WaveformGenerator(
                duration=segment_length,
                sampling_frequency=sample_rate,
                frequency_domain_source_model=waveform,
                waveform_arguments=waveform_arguments
            )

    #now sample params

    #support passing in either the name of the file or the actual prior
    if type(prior_file) == str:
        priors = bilby.gw.prior.PriorDict(prior_file)
    else:
        priors = prior_file
    injection_parameters = priors.sample(len(bkg_segs))
    #print("injection parameters,", injection_parameters)

    #reshaping, code taken from https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/injection.py
    #lns 107-110
    injection_parameters = [
            dict(zip(injection_parameters, col))
            for col in zip(*injection_parameters.values())
        ]

    bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
                ifo
            )
    #make full seg, basically a segment with all the bkg_segs appended together
    print("BKG SEG SHAPE", bkg_segs.shape, "hstack", np.hstack(bkg_segs).shape)
    full_seg = TimeSeries(np.hstack(bkg_segs), sample_rate=sample_rate, t0 = 0)

    bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

    for i, p in enumerate(injection_parameters):
        #print("param set", p)
        p['luminosity_distance']=np.random.uniform(50, 200)
        ra = p['ra']
        dec = p['dec']
        start_time = (i * segment_length)*sample_rate #start of the segment to be injected into
        geocent_time = t0 + start_time
        p['geocent_time'] = 0 * sample_rate * segment_length / 2 #center the injection in the middle of the signal
        psi = p['psi']
        #print("start time", start_time)
        #print('geocent_time', p['geocent_time'])

        #get hplus, hcross
        polarizations = waveform_generator.time_domain_strain(p)

        injection = np.zeros(len(full_seg))
        #print("full seg length", len(full_seg))
        # loop over polarizaitions applying detector response
        for mode, polarization in polarizations.items():

            response = bilby_ifo.antenna_response( #this gives just a float, my interpretation is the
                ra, dec, geocent_time, psi, mode #inclination angle of the detector to the source
            )

            #print(injection.shape, response.shape, polarization.shape)
            #approximate tricks for finding center and doing injection based on that
            if center_type == None: #just none for the sine gaussian models
                centre = np.argmax(np.abs(polarization)) #this works for sine gaussian but can be problematic for other ones
                #want center to be halfway through the segment, at least for now
                a = start_time + N_datapoints//2 - centre
                #truncate so it doesn't go over the edge, should be ok since all zeros
                inj_slice = slice(a, a + centre*2)
                #print(start_time, centre)
                #all good here for BBH polarization
                #print("centre", centre, "polar shape", polarization.shape)
                #print("polarization shape", polarization.shape)
                #np.save("./inj_polarization.npy", polarization)
                injection[inj_slice] += response * polarization[:centre*2]
            elif center_type in ["BBH", "bbh"]:
                #print("full seg", len(full_seg), "polar", len(polarization))
                #print("A", segment_length*sample_rate )
                #print("indexing injection in ", start_time,start_time+segment_length*sample_rate)
                #print("response", response)
                #injection[start_time:start_time+(segment_length-2)*sample_rate] += response * polarization[-((segment_length-2)*sample_rate):]

                #better idea, put the end of the BBH at the middle? that way clipping isn'ty an issue
                injection[start_time:start_time+segment_length*sample_rate//2] += response*polarization[-(segment_length*sample_rate//2):]

        signal = TimeSeries(injection, times = full_seg.times, unit=full_seg.unit)
        #maybe need to do the shifting stuff here
        full_seg = full_seg.inject(signal)

    #now need to chop back up the full_seg
    injected_bkgs = np.array(full_seg).reshape(bkg_segs.shape)
    
    return injected_bkgs

def clipping(seg:TimeSeries, sample_rate:int, clip_edge:int=1):
    clip_edge_datapoints = int(sample_rate * clip_edge)
    return seg[clip_edge_datapoints:-clip_edge_datapoints]

def whiten_bkgs(bkg_segs:np.ndarray, sample_rate:int, full_asd):
    clip_edge=1
    final_shape = (bkg_segs.shape[0], bkg_segs.shape[1]-2*int(clip_edge*sample_rate))
    white_segs = np.zeros(final_shape)
    for i, bkg_seg in enumerate(bkg_segs):
        white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(asd=full_asd)
        #white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten()
        white_segs[i] = clipping(white_seg, sample_rate, clip_edge=clip_edge)

    #have to do clipping because of edge effects when whitening? check this...yes! have to
    return white_segs

#not used for now
def main_injection(savedir, N = 20, folder_path = None, prior_file = None):
    #just going to work with H1 for now
    #ifos = ["H1"]
    ifos = ["H1", "L1"]

    if folder_path == None: #long default argument
        folder_path = "/home/ryan.raikman/s22/anomaly/data2/glitches/1252150173_1252152348/"
    
    loaded_data = load_folder(folder_path, ifos)

    if prior_file == None:
        prior_file = "/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/prior.prior"

    fs = 4096
    segment_length = 5 #seconds
    for ifo in ifos:
        quiet_times = get_quiet_segments(loaded_data[ifo], N, segment_length) 
        bkg_segs, _ = slice_bkg_segments(loaded_data[ifo], quiet_times, 
                                        segment_length)

        #SG injection for now
        injected_segs = inject_waveforms(olib_time_domain_sine_gaussian,
             bkg_segs, fs, loaded_data[ifo]['data'].t0.value ,segment_length, prior_file, ifo)

        whitened_segs = whiten_bkgs(injected_segs, fs, loaded_data[ifo]["asd"])

        np.save(savedir + "./injected_segs.npy", whitened_segs)

#also not used for now
def main_bkg(savedir, N=20, folder_path = None, prior_file = None):
    #just going to work with H1 for now
    ifos = ["H1"]
    if folder_path == None: #long default argument
        folder_path = "/home/ryan.raikman/s22/anomaly/data2/glitches/1252150173_1252152348/"
    
    loaded_data = load_folder(folder_path, ifos)

    if prior_file == None:
        prior_file = "/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/prior.prior"

    fs = 4096
    segment_length = 5 #seconds
    for ifo in ifos:
        quiet_times = get_quiet_segments(loaded_data[ifo], N, segment_length) 

        bkg_segs, _ = slice_bkg_segments(loaded_data[ifo], quiet_times, 
                                        segment_length)

        whitened_segs = whiten_bkgs(bkg_segs, fs, loaded_data[ifo]["asd"])
        np.save(savedir + "./bkg_segs.npy", whitened_segs)

#also not used
def main_glitches(savedir, N=20, folder_path = None, prior_file = None):
    #just going to work with H1 for now
    ifos = ["H1"]
    if folder_path == None: #long default argument
        folder_path = "/home/ryan.raikman/s22/anomaly/data2/glitches/1252150173_1252152348/"
    
    loaded_data = load_folder(folder_path, ifos)

    if prior_file == None:
        prior_file = "/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/prior.prior"

    fs = 4096
    segment_length = 5 #seconds
    for ifo in ifos:
        loud_times = get_loud_segments(loaded_data[ifo], N, segment_length) 

        glitch_segs, _ = slice_bkg_segments(loaded_data[ifo], loud_times, 
                                        segment_length)

        whitened_segs = whiten_bkgs(glitch_segs, fs, loaded_data[ifo]["asd"])

        np.save(savedir + "./glitch_segs.npy", whitened_segs)



def main_all3(savedir, N=20, folder_path = None, prior_file = None):
    try: 
        os.makedirs(savedir)
    except FileExistsError:
        None
    #just going to work with H1 for now
    ifos = ["H1", "L1"]
    if folder_path == None: #long default argument
        folder_path = "/home/ryan.raikman/s22/anomaly/data2/glitches/1252150173_1252152348/"
    
    loaded_data = load_folder(folder_path, ifos)
    if loaded_data == None:
        print("aborting due to missing data streams")
        return None

    if prior_file == None:
        prior_file = "/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/prior.prior"

    fs = 4096
    segment_length = 5 #seconds
    for ifo in ifos:
        try: 
            os.makedirs(savedir + f"/{ifo}/")
        except FileExistsError:
            None

        #INJECTION
        quiet_times = get_quiet_segments(loaded_data[ifo], N, segment_length) 
        bkg_segs, _ = slice_bkg_segments(loaded_data[ifo], quiet_times, 
                                        segment_length)

        BBH_waveform_args = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)
        if 1:
            #BBH injection
            injected_segs = inject_waveforms(bilby.gw.source.lal_binary_black_hole,
                bkg_segs, fs, loaded_data[ifo]['data'].t0.value ,segment_length, 
                prior_file=bilby.gw.prior.BBHPriorDict(), ifo=ifo,
                waveform_arguments = BBH_waveform_args, domain="freq",
                center_type="BBH")

            whitened_segs = whiten_bkgs(injected_segs, fs, loaded_data[ifo]["asd"])

            np.save(savedir + f"/{ifo}/" + "/bbh_segs.npy", whitened_segs)

        #SG injection for now
        injected_segs = inject_waveforms(olib_time_domain_sine_gaussian,
             bkg_segs, fs, loaded_data[ifo]['data'].t0.value ,segment_length, prior_file, ifo)

        whitened_segs = whiten_bkgs(injected_segs, fs, loaded_data[ifo]["asd"])

        np.save(savedir + f"/{ifo}/" + "/injected_segs.npy", whitened_segs)

        if 0:
            #SG injection for now
            injected_segs = inject_waveforms(olib_time_domain_sine_gaussian,
                bkg_segs, fs, loaded_data[ifo]['data'].t0.value ,segment_length, prior_file, ifo)

            whitened_segs = whiten_bkgs(injected_segs, fs, loaded_data[ifo]["asd"])

            np.save(savedir + f"/{ifo}/" + "/injected_segs.npy", whitened_segs)

        #BACKGROUND
        quiet_times = get_quiet_segments(loaded_data[ifo], N, segment_length) 

        bkg_segs, _ = slice_bkg_segments(loaded_data[ifo], quiet_times, 
                                        segment_length)

        whitened_segs = whiten_bkgs(bkg_segs, fs, loaded_data[ifo]["asd"])
        np.save(savedir + f"/{ifo}/" + "/bkg_segs.npy", whitened_segs)

        #GLITCHES
        loud_times = get_loud_segments(loaded_data[ifo], N, segment_length) 

        glitch_segs, _ = slice_bkg_segments(loaded_data[ifo], loud_times, 
                                        segment_length)

        whitened_segs = whiten_bkgs(glitch_segs, fs, loaded_data[ifo]["asd"])

        np.save(savedir + f"/{ifo}/" + "/glitch_segs.npy", whitened_segs)

#main_bkg("/home/ryan.raikman/s22/anomaly/injection_tests/")
#main_all3(".")