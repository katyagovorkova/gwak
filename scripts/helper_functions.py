import os
import h5py
import time
import bilby
import numpy as np
<<<<<<< HEAD

from scipy.stats import cosine as cosine_distribution
from gwpy.timeseries import TimeSeries
from lalinference import BurstSineGaussian, BurstSineGaussianF

from constants import (
    IFOS,
    SAMPLE_RATE,
    GLITCH_SNR_BAR
    )


def find_h5(path: str):
    h5_file = None
    if not os.path.exists(path):
        return None
    for file in os.listdir(path):
        if file[-3:] == '.h5':
            assert h5_file is None  # make sure only 1 h5 file
            h5_file = path + '/' + file

    assert h5_file is not None  # did not find h5 file
    return h5_file


def load_folder(path: str):
    '''
    load the glitch times and data associated with a "save" folder
    '''

    if path[-1] == '/':
        path = path[:-1]  # hopefully there aren't two...
    folder_name = path.split('/')[-1]
    assert len(folder_name.split('_')) == 2
    start = int(folder_name.split('_')[0])
    end = int(folder_name.split('_')[1])

    loaded_data = dict()
    for ifo in IFOS:
        # get the glitch times first
        h5_file = find_h5(f'{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/')
        if h5_file == None:
            return None

        with h5py.File(h5_file, 'r') as f:
            print('loaded data from h5', h5_file)
            triggers = f['triggers'][:]

        with h5py.File(f'{path}/detec_data_{ifo}.h5', 'r') as f:
            X = f['ts'][:]

        # some statistics on the data
        data_statistics = 0
        if data_statistics:
            print(f'start: {start}, end: {end}')
            print(f'duration, seconds: {end-start}')
            print(f'data length: {len(X)}')
            print(f'with data sampled at {4*SAMPLE_RATE}, len(data)/duration= {len(X)/4/SAMPLE_RATE}')

        sample_rate = 4 * SAMPLE_RATE
        resample_rate = SAMPLE_RATE  # don't need so many samples

        data = TimeSeries(X, sample_rate=sample_rate, t0=start)
        if data_statistics:
            print(f'after creating time series, len(data) = {len(data)}')
            before_resample = len(data)

=======
import scipy.signal as sig
from scipy.stats import cosine as cosine_distribution
from gwpy.timeseries import TimeSeries
from typing import Callable
from anomaly.datagen.waveforms import olib_time_domain_sine_gaussian
from lalinference import BurstSineGaussian, BurstSineGaussianF
'''
some code taken from Ethan Marx
https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/injection.py
'''

IFOS = ["H1", "L1"]


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

    for ifo in IFOS:
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

    loaded_data = dict()
    for ifo in IFOS:
        #get the glitch times first
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

        with h5py.File(h5_file, "r") as f:
            print('loaded data from h5', h5_file)
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


>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
        if sample_rate != resample_rate:
            data = data.resample(resample_rate)

        if data_statistics:
<<<<<<< HEAD
            print(f'after resampling, len(data) = {len(data)}')
            print(f'ratio before to after: {before_resample/len(data)}')

        fftlength = 1
        loaded_data[ifo] = {'triggers': triggers,
                            'data': data,
                            'asd': data.asd(fftlength=fftlength, overlap=0, method='welch', window='hanning')}

    return loaded_data


def get_loud_segments(
        ifo_data: dict,
        N: int,
        segment_length: float):
=======
            print(f"after resampling, len(data) = {len(data)}")
            print(f"ratio before to after: {before_resample/len(data)}")

        fftlength=1
        loaded_data[ifo] = {"triggers": triggers,
                            "data":data,
                            "asd":data.asd(fftlength=fftlength,overlap=0, method='welch', window='hanning')}

    return loaded_data

def get_loud_segments(ifo_data:dict, N:int, segment_length:float):
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
    '''
    sort the glitches by SNR, and return the N loudest ones
    If there are < N glitches, return the number of glitches, i.e. will not crash
    '''
<<<<<<< HEAD
    glitch_times = ifo_data['triggers']['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data']) / ifo_data['data'].sample_rate.value

    glitch_snrs = ifo_data['triggers']['snr']
    # create a sorting, in descending order
    sort_by_snr = glitch_snrs.argsort()[::-1]
=======
    glitch_snr_bar = 10

    glitch_times = ifo_data["triggers"]['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data'])/ifo_data['data'].sample_rate.value

    glitch_snrs = ifo_data["triggers"]['snr']
    sort_by_snr = glitch_snrs.argsort()[::-1] #create a sorting, in descending order
    #print(glitch_snrs[sort_by_snr])
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
    glitch_times_sorted = glitch_times[sort_by_snr]

    glitch_start_times = []
    for i in range(min(len(glitch_times_sorted), N)):
<<<<<<< HEAD
        # going to set a glitch SNR bar because the really quiet ones don't
        # really do much
        if glitch_snrs[sort_by_snr][i] < GLITCH_SNR_BAR:
            continue
        # want times for the beginning of the segment, and lets have the glitch
        # centered
        gt = glitch_times_sorted[i]
        glitch_start_times.append(gt - segment_length / 2)

    return glitch_start_times


def get_quiet_segments(
        ifo_data: dict,
        N: int,
        segment_length: float):
    '''
    get N times that are away from the glitches
    '''
    glitch_times = ifo_data['triggers']['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data']) / ifo_data['data'].sample_rate.value

    # cut off the edge effects already
    valid_start_times = np.arange(
        t0, tend - segment_length, segment_length / 100)
    open_times = np.ones(valid_start_times.shape)
    for gt in glitch_times:
        idx = np.searchsorted(valid_start_times, gt)
        bottom_cut = max(t0, gt - 2)
        top_cut = min(tend, gt + 2)

        # convert to indicies
        bottom_idx = np.searchsorted(valid_start_times, bottom_cut)
        top_idx = np.searchsorted(valid_start_times, top_cut)
        open_times[bottom_idx:top_idx] = np.zeros((top_idx - bottom_idx))
        if 0:
            for i in range(idx - 2, idx + 2):
                try:
                    open_times[i] = 0
                except IndexError:
                    None  # this is dangerous, just using it for now to deal with glitches on the edge

    total_available = np.sum(open_times)
    if total_available < N:  # manually setting the maximuim
        N = int(total_available)

    # convert to bool mask
    open_times = (open_times != 0)
    valid_start_times = valid_start_times[open_times]

    # now just take from valid start times without replacement
    print('valid start times', valid_start_times)
    return valid_start_times
    quiet_times = np.random.choice(valid_start_times, N, replace=False)

    # one last check
    for elem in quiet_times:

        assert np.abs(glitch_times - elem).min() >= segment_length
        assert np.abs(glitch_times - (elem + segment_length)
                      ).min() >= segment_length
    return quiet_times


def slice_bkg_segments(
        ifo_data: dict,
        data,
        start_times: list[float],
        segment_length: float):
    # turn the start times into background segments
=======
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

    valid_start_times = np.arange(t0, tend-segment_length, segment_length/100) #cut off the edge effects already
    #print("VST len", valid_start_times.shape)
    #print("valid start times", valid_start_times )
    open_times = np.ones(valid_start_times.shape)
    #print("open times shape", open_times.shape)
    #print("t0", t0)
    #print("valid_start_times", valid_start_times)
    for gt in glitch_times:
        #print("searching", gt, "into", valid_start_times)
        idx = np.searchsorted(valid_start_times, gt)
        #print("got result", idx)
        #print([float(elem) for elem in valid_start_times[idx-1:idx+2]], gt)
        #get rid of 3 indicies
        bottom_cut = max(t0, gt-2)
        top_cut = min(tend, gt+2)

        #convert to indicies
        bottom_idx = np.searchsorted(valid_start_times, bottom_cut)
        top_idx = np.searchsorted(valid_start_times, top_cut)
        open_times[bottom_idx:top_idx] = np.zeros((top_idx-bottom_idx))
        if 0:
            for i in range(idx-2, idx+2):
            #print(idx)
                try:
                    open_times[i] = 0
                except IndexError:
                    #print("something happened out of bounds with", idx)
                    None #this is dangerous, just using it for now to deal with glitches on the edge

    total_available = np.sum(open_times)
    #print("total available", total_available)
    #assert total_available >= N #if this fails, then need to revise choosing strategy
    if total_available < N: #manually setting the maximuim
        N = int(total_available)

    #convert to bool mask
    open_times = (open_times != 0)
    valid_start_times = valid_start_times[open_times]

    #now just take from valid start times without replacement
    print("valid start times", valid_start_times)
    return valid_start_times
    quiet_times = np.random.choice(valid_start_times, N, replace=False)

    #one last check
    for elem in quiet_times:

        assert np.abs(glitch_times-elem).min() >= segment_length
        assert np.abs(glitch_times-(elem+segment_length)).min() >= segment_length
    return quiet_times

def slice_bkg_segments(ifo_data:dict, data, start_times:list[float], segment_length:float):
    #print("got for start times,", start_times)
    #turn the start times into background segments
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
    fs = ifo_data['data'].sample_rate.value
    t0 = ifo_data['data'].t0.value
    N_datapoints = int(segment_length * fs)
    bkg_segs = np.zeros(shape=(2, len(start_times), N_datapoints))
    bkg_timeseries = []

    for i in range(len(start_times)):
<<<<<<< HEAD
        slx = slice(int((start_times[i] - t0) * fs),
                    int((start_times[i] - t0) * fs) + N_datapoints)
        slice_segment = data[:, slx]
        toggle_noise = 1
=======
        slx = slice( int((start_times[i]-t0)*fs),  int((start_times[i]-t0)*fs)+N_datapoints)
        slice_segment = data[:, slx]
        #print("WHERE THE ASSERTION WAS FAILING", slice_segment.t0.value, start_times[i])
        #assert np.isclose(slice_segment.t0.value, start_times[i]) #check that timing lines up
        toggle_noise=1
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
        bkg_segs[:, i] = np.array(slice_segment) * toggle_noise
        bkg_timeseries.append(slice_segment)

    return bkg_segs, bkg_timeseries

<<<<<<< HEAD

def clipping(
        seg: TimeSeries,
        sample_rate: int,
        clip_edge: int=1):
    clip_edge_datapoints = int(sample_rate * clip_edge)
    return seg[clip_edge_datapoints:-clip_edge_datapoints]


def whiten_bandpass_bkgs(
        bkg_segs_full: np.ndarray,
        sample_rate: int,
        H1_asd,
        L1_asd):
    clip_edge = 1
    ASDs = {'H1': H1_asd,
            'L1': L1_asd}
    all_white_segs = []
    for i, ifo in enumerate(IFOS):
        bkg_segs = bkg_segs_full[i]
        final_shape = (bkg_segs.shape[0], bkg_segs.shape[
                       1] - 2 * int(clip_edge * sample_rate))
        white_segs = np.zeros(final_shape)
        for i, bkg_seg in enumerate(bkg_segs):
            white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(
                asd=ASDs[ifo]).bandpass(30, 1500)
            white_segs[i] = clipping(
                white_seg, sample_rate, clip_edge=clip_edge)
        all_white_segs.append(white_segs)

    # have to do clipping because of edge effects when whitening? check
    # this...yes! have to
    final_whiten = np.stack(all_white_segs)

    return final_whiten


def get_background_segs(
        loaded_data,
        data,
        n_backgrounds,
        segment_length):
    # note - by here, N samples have NOT been drawn
    quiet_times_H1 = get_quiet_segments(loaded_data['H1'], n_backgrounds, segment_length)
    quiet_times_L1 = get_quiet_segments(loaded_data['L1'], n_backgrounds, segment_length)

    quiet_times = np.intersect1d(quiet_times_H1, quiet_times_L1)
    n_backgrounds = min(n_backgrounds, len(quiet_times))
    quiet_times = np.random.choice(quiet_times, n_backgrounds, replace=False)

    # passing loaded_data here for reference to values like t0 and fs
    bkg_segs, _ = slice_bkg_segments(loaded_data['H1'], data, quiet_times,
                                     segment_length)
    return bkg_segs


def calc_psd(
        data,
        df,
        fftlength=2):
    # heavily inspired by
    # https://github.com/ML4GW/ml4gw/blob/main/ml4gw/spectral.py
    default_psd_kwargs = dict(method='median', window='hann')
    x = data.psd(fftlength, **default_psd_kwargs)
    if x.df.value != df:
        x = x.interpolate(df)

    return x.value


def calc_SNR_new(
        datae,
        detec_data,
        fs,
        detector_psds=None,
        return_psds=False):
    # heavily inspired by:
    # https://github.com/ML4GW/ml4gw/blob/main/ml4gw/gw.py
    snr_tot = np.zeros(datae.shape[1])
    if return_psds:
        save_psds = dict()
    for ifo_num, ifo in enumerate(IFOS):
        data = datae[ifo_num]
        df = fs / data.shape[-1]

=======
def clipping(seg:TimeSeries, sample_rate:int, clip_edge:int=1):
    clip_edge_datapoints = int(sample_rate * clip_edge)
    return seg[clip_edge_datapoints:-clip_edge_datapoints]

def whiten_bandpass_bkgs(bkg_segs_full:np.ndarray, sample_rate:int, H1_asd, L1_asd):
    clip_edge=1
    ASDs = {"H1":H1_asd,
                "L1":L1_asd}
    all_white_segs = []
    #print("into whiten, full:", bkg_segs_full.shape)
    for i, ifo in enumerate(["H1", "L1"]):
        bkg_segs = bkg_segs_full[i]
        #print("in whiten, ", bkg_segs.shape)
        final_shape = (bkg_segs.shape[0], bkg_segs.shape[1]-2*int(clip_edge*sample_rate))
        white_segs = np.zeros(final_shape)
        for i, bkg_seg in enumerate(bkg_segs):
            white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(asd=ASDs[ifo]).bandpass(30, 1500)
            #white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten()
            white_segs[i] = clipping(white_seg, sample_rate, clip_edge=clip_edge)
        all_white_segs.append(white_segs)

    #have to do clipping because of edge effects when whitening? check this...yes! have to
    FINAL_WHITEN = np.stack(all_white_segs)
    #print("FINAL WHITEN", FINAL_WHITEN.shape)
    return FINAL_WHITEN

def get_bkg_segs(loaded_data, data, N, segment_length):
    #note - by here, N samples have NOT been drawn
    quiet_times_H1 = get_quiet_segments(loaded_data["H1"], N, segment_length)
    quiet_times_L1 = get_quiet_segments(loaded_data["L1"], N, segment_length)

    quiet_times = np.intersect1d(quiet_times_H1, quiet_times_L1)
    #print("after intersection", quiet_times)
    N = min(N, len(quiet_times))
    quiet_times = np.random.choice(quiet_times, N, replace=False)

   # print("quiet times, ", quiet_times)

    #assert False
    #passing loaded_data here for reference to values like t0 and fs
    bkg_segs, _ = slice_bkg_segments(loaded_data["H1"], data, quiet_times,
                                    segment_length)
    return bkg_segs

def calc_psd(data, df, fftlength=2):
    #heavily inspired by https://github.com/ML4GW/ml4gw/blob/main/ml4gw/spectral.py
    default_psd_kwargs = dict(method='median', window="hann")
    x = data.psd(fftlength, **default_psd_kwargs)
    if x.df.value != df:
        x = x.interpolate(df)
    #x = x.crop(0, target_
    return x.value

def calc_SNR_new(datae, detec_data, fs, highpass=None, detector_psds=None, return_psds=False):
    #heavily inspired by:
    #https://github.com/ML4GW/ml4gw/blob/main/ml4gw/gw.py
    snr_tot = np.zeros(datae.shape[1])
    if return_psds:
        save_psds = dict()
    for ifo_num, ifo in enumerate(["H1", "L1"]):
        data = datae[ifo_num]
        df = fs / data.shape[-1]
        #print("df, ", df)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
        single_detec_data = detec_data[ifo]['data']

        fft_of_template = np.fft.rfft(data)
        fft_of_template = np.abs(fft_of_template) / fs
<<<<<<< HEAD
        # calculate detector psd

        if detector_psds is None:
            # this is the desired df value
            detec_psd = calc_psd(single_detec_data, df)
=======
        #calculate detector psd
        #print(single_detec_data)
        #print(np.array(single_detec_data))
        #ts = time.time()
        if detector_psds is None:
            detec_psd = calc_psd(single_detec_data, df)#this is the desired df value
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
        else:
            detec_psd = detector_psds[ifo]
        if return_psds:
            save_psds[ifo] = detec_psd
<<<<<<< HEAD

=======
        #print("calculating detector psd", time.time()-ts)

        #print("fft data shape, asds[ifo] shape", fft_data.shape, asds[ifo].shape)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
        fft_data = fft_of_template
        integrand = fft_data / detec_psd ** 0.5
        integrand = integrand ** 2

<<<<<<< HEAD
        integrated = integrand.sum(axis=-1) * df
        integrated = 4 * integrated
        snr_tot = snr_tot + integrated  # adding SNR^2

=======

        if highpass is not None:
            freqs = np.fft.rfftfreq(responses.shape[-1], 1/fs)
            mask = freqs >= highpass
            integrand *= mask

        integrated = integrand.sum(axis=-1)*df
        integrated = 4 * integrated
        #np.sqrt(integrated)
        snr_tot = snr_tot+ integrated #adding SNR^2

    #print("SNR tot before compile", snr_tot)
    #print(snr_tot**0.5)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
    if return_psds:
        return snr_tot**0.5, save_psds
    return snr_tot**0.5

<<<<<<< HEAD

def inject_hplus_hcross(
        bkg_seg: np.ndarray,
        pols: dict,
        sample_rate: int,
        segment_length: int,
        background=None,
        SNR=None,
        get_psds=False,
        detector_psds=None):

    final_injects = []
    final_injects_nonoise = []
    # sample ra, dec, geocent_time, psi for the signal
    ra = np.random.uniform(0, 2 * np.pi)
    psi = np.random.uniform(0, 2 * np.pi)
    dec = cosine_distribution.rvs(size=1)[0]
    geocent_time = segment_length / 2  # put it in the middle

    for i, ifo in enumerate(IFOS):
        bkgX = bkg_seg[i]  # get rid of noise? to see if signal is showing up
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
            ifo
        )

        full_seg = TimeSeries(bkgX, sample_rate=sample_rate, t0=0)
        nonoise_seg = TimeSeries(0 * bkgX, sample_rate=sample_rate, t0=0)
        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

        injection = np.zeros(len(bkgX))
        for mode, polarization in pols.items():  # make sure that this is formatted correctly
            ts = time.time()
            response = bilby_ifo.antenna_response(  # this gives just a float, my interpretation is the
                ra, dec, geocent_time, psi, mode)  # inclination angle of the detector to the source

            midp = len(injection) // 2
            if len(polarization) % 2 == 1:
                polarization = polarization[:-1]
            half_polar = len(polarization) // 2
            # off by one datapoint if this fails, don't think it matters
            assert len(polarization) % 2 == 0
            slx = slice(midp - half_polar, midp + half_polar)
            injection[slx] += response * polarization

        signal = TimeSeries(
            injection, times=full_seg.times, unit=full_seg.unit)
=======
def inject_hplus_hcross(bkg_seg:np.ndarray,
                        pols:dict,
                        sample_rate:int,
                        segment_length:int,
                        cheat:bool=True,
                        inject_on_edge:bool=False,
                        background=None,
                        SNR=None,
                        return_loc=False,
                        sky_location=None,
                        get_psds = False,
                        detector_psds=None):

    final_injects = []
    final_injects_nonoise = []
    if sky_location is None:
    #sample ra, dec, geocent_time, psi for the signal
        ra = np.random.uniform(0, 2*np.pi)
        psi = np.random.uniform(0, 2*np.pi)
        dec = cosine_distribution.rvs(size=1)[0]
    else:
        ra, psi, dec = sky_location
    geocent_time = segment_length/2#put it in the middle

    for i, ifo in enumerate(["H1", "L1"]):
        #ts = time.time()
        bkgX = bkg_seg[i]# get rid of noise? to see if signal is showing up
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
                    ifo
                )

        full_seg = TimeSeries(bkgX, sample_rate=sample_rate, t0 = 0)
        nonoise_seg = TimeSeries(0*bkgX, sample_rate=sample_rate, t0 = 0)
        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)
        #print(f"initializing detectors {time.time()-ts :.2f}")

        #ts_ = time.time()
        injection = np.zeros(len(bkgX))
        for mode, polarization in pols.items(): #make sure that this is formatted correctly
            ts = time.time()
            response = bilby_ifo.antenna_response( #this gives just a float, my interpretation is the
                ra, dec, geocent_time, psi, mode) #inclination angle of the detector to the source
            #print(f"response calculation : {time.time()-ts :.2f}")

           # response *= 1e23
            if not inject_on_edge:
                midp = len(injection)//2
                if len(polarization) % 2 == 1:
                    polarization=polarization[:-1]
                half_polar = len(polarization)//2
                assert len(polarization) % 2 == 0  #off by one datapoint if this fails, don't think it matters
                slx = slice(midp-half_polar, midp+half_polar)
            else:
                #this is particularly for BNS, so the polarization files are same
                #length as the background segments
                slx = slice(0, len(polarization)) #...should work
            #print("injection shape, polarizaiton shape, response,", injection.shape, polarization.shape, response)
            injection[slx] += response * polarization

        signal = TimeSeries(injection, times = full_seg.times, unit=full_seg.unit)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

        full_seg = full_seg.inject(signal)
        nonoise_seg = nonoise_seg.inject(signal)
        final_injects.append(full_seg)
        final_injects_nonoise.append(nonoise_seg)
<<<<<<< HEAD

    if SNR is None:
        # no amplitude modifications
        return np.stack(final_injects)[:, np.newaxis, :], None
    final_bothdetector = np.stack(final_injects)[:, np.newaxis, :]
    final_bothdetector_nonoise = np.stack(
        final_injects_nonoise)[:, np.newaxis, :]

    if get_psds:
        _, psds = calc_SNR_new(final_bothdetector_nonoise,
                               background, sample_rate, return_psds=True)
        return psds
    computed_SNR = calc_SNR_new(final_bothdetector_nonoise,
        background, sample_rate, detector_psds=detector_psds)
    response_scale = np.array(SNR / computed_SNR)[0]

    # now do the injection again
    final_injects = []
    final_injects_nonoise = []
    for i, ifo in enumerate(IFOS):
        bkgX = bkg_seg[i]  # get rid of noise? to see if signal is showing up
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
            ifo
        )
        full_seg = TimeSeries(bkgX, sample_rate=sample_rate, t0=0)
        nonoise_seg = TimeSeries(0 * bkgX, sample_rate=sample_rate, t0=0)
        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

        injection = np.zeros(len(bkgX))
        for mode, polarization in pols.items():  # make sure that this is formatted correctly
            response = bilby_ifo.antenna_response(  # this gives just a float, my interpretation is the
                # inclination angle of the detector to the source
                ra, dec, geocent_time, psi, mode

            )
            response *= response_scale
            midp = len(injection) // 2
            if len(polarization) % 2 == 1:
                polarization = polarization[:-1]
            # betting on the fact that the length of polarization is even
            half_polar = len(polarization) // 2
            # off by one datapoint if this fails, don't think it matters
            assert len(polarization) % 2 == 0
            slx = slice(midp - half_polar, midp + half_polar)
            injection[slx] += response * polarization

        signal = TimeSeries(
            injection, times=full_seg.times, unit=full_seg.unit)
=======
        #print(f"injection time: {time.time()-ts_ :.2f}")


    if SNR is None:
        #no amplitude modifications
        #print("360", np.stack(final_injects).shape)
        return np.stack(final_injects)[:, np.newaxis,  :], None
    final_bothdetector = np.stack(final_injects)[:, np.newaxis, :]
    #ts = time.time()
    final_bothdetector_nonoise = np.stack(final_injects_nonoise)[:, np.newaxis, :]

    if get_psds:
        _, psds = calc_SNR_new(final_bothdetector_nonoise, background, sample_rate, return_psds=True)
        return psds
    computed_SNR = calc_SNR_new(final_bothdetector_nonoise, background, sample_rate, detector_psds=detector_psds)
    #print(f"SNR calculation: {time.time()-ts :.2f}")
    #print("computed SNR", computed_SNR)
    response_scale = np.array(SNR/computed_SNR)[0]
    #print("response scale", response_scale)
    #now do the injection again
    final_injects = []
    final_injects_nonoise = []
    for i, ifo in enumerate(["H1", "L1"]):
        bkgX = bkg_seg[i]# get rid of noise? to see if signal is showing up
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
                    ifo
                )
        full_seg = TimeSeries(bkgX, sample_rate=sample_rate, t0 = 0)
        nonoise_seg = TimeSeries(0*bkgX, sample_rate=sample_rate, t0 = 0)
        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

        injection = np.zeros(len(bkgX))
        for mode, polarization in pols.items(): #make sure that this is formatted correctly
            response = bilby_ifo.antenna_response( #this gives just a float, my interpretation is the
                ra, dec, geocent_time, psi, mode #inclination angle of the detector to the source

            )
            response *= response_scale
            if not inject_on_edge:
                midp = len(injection)//2
                if len(polarization) % 2 == 1:
                    polarization=polarization[:-1]
                #betting on the fact that the length of polarization is even
                half_polar = len(polarization)//2
                #print(len(polarization))
                assert len(polarization) % 2 == 0  #off by one datapoint if this fails, don't think it matters
                slx = slice(midp-half_polar, midp+half_polar)
            else:
                #this is particularly for BNS, so the polarization files are same
                #length as the background segments
                slx = slice(0, len(polarization)) #...should work
            #print("injection shape, polarizaiton shape, response,", injection.shape, polarization.shape, response)
            injection[slx] += response * polarization

        signal = TimeSeries(injection, times = full_seg.times, unit=full_seg.unit)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

        full_seg = full_seg.inject(signal)
        nonoise_seg = nonoise_seg.inject(signal)
        final_injects.append(full_seg)
        final_injects_nonoise.append(nonoise_seg)

<<<<<<< HEAD
    final_bothdetector = np.stack(final_injects)  # [:, np.newaxis, :]
    final_bothdetector_nonoise = np.stack(
        final_injects_nonoise)  # [:, np.newaxis, :]

    return final_bothdetector, final_bothdetector_nonoise


def olib_time_domain_sine_gaussian(
        time_array,  # times at which to evaluate the model
        hrss,  # hrss of source
        q,  # quality factor of the burst, determines the width in frequency
        frequency,  # the central frequency of the burst
        phase,  # a reference phase parameter
        eccentricity,  # signal eccentricity, determines the relative fraction of each polarisation mode
        geocent_time,
        **kwargs):  # A 'hack' to make this compatible with bilby_pipe.
        # Currently bilby_pipe passes a lot of
        # bbh specific parameters to the waveform model.
        # This should allow those to be handled gracefully.

    """
    Collection of waveforms used for injections, or PE recovery via bilby

    COPIED EXACTLY from ETHAN MARX OLIB BY RYAN RAIKMAN
    https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/waveforms.py

    A wrapper to the oLIB source model in the time domain,
    convenient for use with bilby

=======
    final_bothdetector = np.stack(final_injects)#[:, np.newaxis, :]
    final_bothdetector_nonoise = np.stack(final_injects_nonoise)#[:, np.newaxis, :]

    return final_bothdetector, final_bothdetector_nonoise

    if return_loc:
        return final_bothdetector, final_bothdetector_nonoise, [ra, psi, dec]
    else:
        return final_bothdetector, final_bothdetector_nonoise



"""
Collection of waveforms used for injections, or PE recovery via bilby

COPIED EXACTLY from ETHAN MARX OLIB BY RYAN RAIKMAN
https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/waveforms.py
"""


def olib_time_domain_sine_gaussian(
    time_array,
    hrss,
    q,
    frequency,
    phase,
    eccentricity,
    geocent_time,
    **kwargs,
):
    """
    A wrapper to the oLIB source model in the time domain,
    convenient for use with bilby

    Args:
        time_array:
            times at which to evaluate the model
        hrss:
            hrss of source
        q:
            quality factor of the burst, determines the width in frequency
        frequency:
            the central frequency of the burst
        phase:
            a reference phase parameter
        eccentricity:
            signal eccentricity, determines the relative fraction of each
            polarisation mode
        **kwargs:
            A 'hack' to make this compatible with bilby_pipe.
            Currently bilby_pipe passes a lot of
            bbh specific parameters to the waveform model.
            This should allow those to be handled gracefully.
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
    Returns:
        dict containing the two polarization modes
    """

    # get dt from time array
    dt = time_array[1] - time_array[0]

    # cast arguments as proper types for lalinference
    hrss = float(hrss)
    eccentricity = float(hrss)
    phase = float(phase)
    frequency = float(frequency)
    q = float(q)

    # produces wf's with t0 at 0
    hplus, hcross = BurstSineGaussian(
        q, frequency, hrss, eccentricity, phase, dt
    )

    plus = np.zeros(len(time_array))
    cross = np.zeros(len(time_array))

<<<<<<< HEAD
    plus[:len(hplus.data.data)] = hplus.data.data
    cross[:len(hcross.data.data)] = hcross.data.data
=======
    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

    return dict(plus=plus, cross=cross)


def olib_freq_domain_sine_gaussian(
<<<<<<< HEAD
        freq_array,
        hrss,
        q,
        frequency,
        phase,
        eccentricity,
        geocent_time,
        **kwargs):
=======
    freq_array, hrss, q, frequency, phase, eccentricity, geocent_time, **kwargs
):
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

    deltaF = freq_array[1] - freq_array[0]
    deltaT = 0.5 / (freq_array[-1])
    hplus, hcross = BurstSineGaussianF(
        q, frequency, hrss, eccentricity, phase, deltaF, deltaT
    )

    plus = np.zeros(len(freq_array), dtype=complex)
    cross = np.zeros(len(freq_array), dtype=complex)

    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data

<<<<<<< HEAD
    return dict(plus=plus, cross=cross)

def WNB(duration, fs, fmin, fmax, enveloped=True, sidePad=None):
    """Generate a random signal of given duration with constant power
    in a given frequency range band (and zero power out of the this range).

        Parameters
        ----------

        duration: int/float
            The desirable duration of the signal. Duration must be bigger than 1/fs
        fs: int
            The sample frequncy of the signal
        fmin: int/float
            The minimum frequency
        fmax: int/float
            The maximum frequency
        enveloped: bool (optional)
            If set to True it returns the signal within a sigmoid envelope on the edges.
            If not specified it is set to False.
        sidePad: int/bool(optional)
            An option to pad with sidePad number of zeros each side of the injection. It is suggested
            to have zeropaded injections for the timeshifts to represent 32 ms, to make it easier
            for the projectwave function. If not specified or
            False it is set to 0. If set True it is set to ceil(fs/32). WARNING: Using sidePad will
            make the injection length bigger than the duration


        Returns
        -------

        numpy.ndarray
            The WNB waveform
    """
    if not (isinstance(fmin, (int, float)) and fmin >= 1):
        raise ValueError('fmin must be greater than 1')
    if not (isinstance(fmax, (int, float)) and fmax > fmin):
        raise ValueError('fmax must be greater than fmin')
    if not (isinstance(fs, int) and fs >= 2 * fmax):
        raise ValueError('fs must be greater than 2*fax')
    if not (isinstance(duration, (int, float)) and duration > 1 / fs):
        raise ValueError('duration must be bigger than 1/fs')
    if sidePad is None:
        sidePad = 0
    if isinstance(sidePad, bool):
        if sidePad:
            sidePad = ceil(fs / 32)
        else:
            sidePad = 0
    elif isinstance(sidePad, (int, float)) and sidePad >= 0:
        sidePad = int(sidePad)

    else:
        raise TypeError('sidePad can be boolean or int value.'
                        + ' If set True it is set to ceil(fs/32).')

    df = fmax - fmin
    T = ceil(duration)

    # Generate white noise with duration dt at sample rate df. This will be
    # white over the band [-df/2,df/2].
    nSamp = ceil(T * df)
    h = []
    for _h in range(2):

        x_ = TimeSeries(np.random.randn(nSamp),sample_rate=1/T)

        # Resample to desired sample rate fs.
        x = x_.resample(fs / df)

        # Heterodyne up by f+df/2 (moves zero frequency to center of desired
        # band).
        fshift = fmin + df / 2.
        x = x * np.exp(-2 * np.pi * 1j * fshift /
                       fs * np.arange(1, len(x) + 1))

        # Take real part and adjust length to duration and normalise to 1.
        x = np.array(np.real(x))[:int(fs * duration)] / np.sqrt(2)
        x = x / np.abs(np.max(x))
        h.append(x)

    hp = h[0]
    hc = h[1]

    if enveloped:
        hp = envelope(hp, option='sigmoid')
        hc = envelope(hc, option='sigmoid')

    if sidePad != 0:
        hp = np.hstack((np.zeros(sidePad), hp, np.zeros(sidePad)))
        hc = np.hstack((np.zeros(sidePad), hc, np.zeros(sidePad)))

    return (hp, hc)

def sigmoid(timeRange, t0=None, stepTime=None, ascending=True):
    """Sigmoid is a functIon of a simple sigmoid.

        Parameters
        ----------

        timeRange: list/numpy.ndarray
            The time range that the sigmoid will be applied.

        t0: int/float (optional)
            The time in the center of the sigmoid. If not specified,
            the default value is the center of the timeRange.

        stepTime: int/float (optional)
            The time interval where the function will go from 0.01
            to 0.99 (or the oposite). If not specified, the default value
            is the duration of the timeRange.

        ascending: bool (optional)
            If True the sigmoid will go from 0 to 1, if False from 1 to 0.
            If not specified the default value is True.

        Returns
        -------

        numpy.ndarray
            A sigmoid function
    """
    if t0 is None:
        t0 = (timeRange[-1] - timeRange[0]) / 2
    if stepTime is None:
        stepTime = (timeRange[-1] - timeRange[0])
    if ascending is None:
        ascending

    a = ((-1)**(int(not ascending))) * (np.log(0.01 / 0.99) / (stepTime - t0))

    y = 1 / (1 + np.exp(a * (np.array(timeRange) - t0)))
    return y

def envelope(strain, option=None, **kwargs):
    """Envelope is a wrapper function that covers all types of envelopes available here.

        Arguments
        ---------

        strain: list/array/gwpy.TimeSeries
            The timeseries to be enveloped

        option: {'sigmoid'} (optional)
            The type of envelope you want to apply. If not specified it defaults to 'sigmoid'

        **kwargs: Any keyword arguments acompany the option of envelope.

        Returns
        -------
        numpy.ndarray
            The same timeseries that had in the input, but enveloped.


    """

    if option is None:
        option = 'sigmoid'

    if 'fs' in kwargs:
        if not (isinstance(kwargs['fs'], int) and kwargs['fs'] >= 1):
            raise ValueError('fs must be greater than 2*fax')
        else:
            fs = kwargs['fs']
            duration = len(strain) / fs
    else:
        fs = len(strain)
        duration = 1

    if option == 'sigmoid':

        envStart = sigmoid(np.arange(int(len(strain) / 10)))
        envEnd = sigmoid(np.arange(int(len(strain) / 10)), ascending=False)
        env = np.hstack(
            (envStart,
             np.ones(
                 len(strain) -
                 len(envStart) -
                 len(envEnd)) -
                0.01,
                envEnd))

    if option == 'wnb':

        envStart = sigmoid(np.arange(int(len(strain) / 10)))
        envEnd = sigmoid(np.arange(int(len(strain) / 10)), ascending=False)

        fmin = 1
        fmax = fmin + 1 + np.random.rand() * 10
        wnb = 1.2 + np.random.rand() * 1.8 + WNB(duration=duration,
                                                 fs=fs, fmin=fmin, fmax=fmax)
        env = wnb * np.hstack((envStart, np.ones(len(strain) -
                              len(envStart) - len(envEnd)) - 0.01, envEnd))
        env = env / np.abs(np.max(env))

    return env * np.array(strain)
=======
    return dict(plus=plus, cross=cross)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
