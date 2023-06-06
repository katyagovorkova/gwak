import os
import h5py
import time
# import bilby
import numpy as np
import torch

from scipy.stats import cosine as cosine_distribution
from gwpy.timeseries import TimeSeries
from lalinference import BurstSineGaussian, BurstSineGaussianF

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    IFOS,
    SAMPLE_RATE,
    INJECT_AT_END,
    EDGE_INJECT_SPACING,
    GLITCH_SNR_BAR,
    STRAIN_START,
    STRAIN_STOP,
    LOADED_DATA_SAMPLE_RATE,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    GW_EVENT_CLEANING_WINDOW,
    SEG_NUM_TIMESTEPS,
    SEGMENT_OVERLAP,
    CLASS_ORDER,
    SIGNIFICANCE_NORMALIZATION_DURATION,
    DEVICE)


def mae(a, b):
    '''
    compute MAE across a, b
    using first dimension as representing each sample
    '''
    norm_factor = a[0].size
    assert a.shape == b.shape
    diff = np.abs(a - b)
    N = len(diff)

    # sum across all axes except the first one
    return np.sum(diff.reshape(N, -1), axis=1) / norm_factor

def mae_torch(a, b):
    # compute the MAE, do .mean twice:
    # once for the time axis, second time for detector axis
    return torch.abs(a-b).mean(axis=-1).mean(axis=-1)

def std_normalizer(data):
    feature = 1
    if data.shape[1] == 2:
        feature_axis = 2
        
    std_vals = np.std(data, dim=feature_axis)[:, :, np.newaxis]
    return data / std_vals

def std_normalizer_torch(data):
    feature_axis = 1
    if data.shape[1] == 2:
        feature_axis = 2
        
    std_vals = torch.std(data, dim=feature_axis)[:, :, None]
    print("data, std_vals", data.shape, std_vals.shape)
    return data / std_vals

def stack_dict_into_tensor(data_dict):
    '''
    Input is a dictionary of keys, stack it into *torch* tensor
    '''
    device = DEVICE
    fill_len = len(data_dict['bbh'])
    stacked_tensor = torch.empty((fill_len, 4), device=device)
    for class_name in data_dict.keys():
        stack_index = CLASS_ORDER.index(class_name)
        stacked_tensor[:, stack_index] = data_dict[class_name]
    
    return stacked_tensor


def reduce_to_significance(data):
    '''
    Data is a torch tensor of shape (N_samples, N_features),
    normalize / calculate significance for each of the axis 
    by std/mean of SIGNIFICANCE_NORMALIZATION_DURATION
    '''
    N_samples, N_features = data.shape
    norm_datapoints = int(SIGNIFICANCE_NORMALIZATION_DURATION * SAMPLE_RATE)
    assert  N_samples > norm_datapoints

    device = DEVICE

    #take the mean, std by sliding window of size norm_datapoints
    stacked = torch.reshape(
        data[: (N_samples // norm_datapoints) * norm_datapoints ],
        (norm_datapoints, -1, N_features)
    )
    means = stacked.mean(dim=0)
    stds = stacked.std(dim=0)

    computed_significance = torch.empty((N_samples - norm_datapoints, N_features), device=device)

    N_splits = means.shape[0]
    for i in range(N_splits):
        start, end = i*norm_datapoints, (i+1)*norm_datapoints
        computed_significance[start:end] = (data[start+norm_datapoints:end+norm_datapoints] - means[i])/stds[i]
        
    return computed_significance


def find_h5(path: str):
    h5_file = None
    print("looking for", path)
    assert os.path.exists(path)
    if not os.path.exists(path):
        return None
    for file in os.listdir(path):
        if file[-3:] == '.h5':
            assert h5_file is None  # make sure only 1 h5 file
            h5_file = path + '/' + file

    assert h5_file is not None  # did not find h5 file
    return h5_file


def load_folder(
        path: str,
        load_start: int = None,
        load_stop: int = None):
    '''
    load the glitch times and data associated with a "save" folder
    '''
    start = STRAIN_START
    end = STRAIN_STOP

    path = path+f'{start}_{end}/'

    loaded_data = dict()
    for ifo in IFOS:
        # get the glitch times first
        h5_file = find_h5(f'{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/')
        if h5_file == None:
            return None

        with h5py.File(h5_file, 'r') as f:
            print('loading data from h5', h5_file, "...")
            triggers = f['triggers'][:]
            

        with h5py.File(f'{path}/detec_data_{ifo}.h5', 'r') as f:
            if load_start == None or load_stop == None:
                X = f['ts'][:]
            else:
                datapoints_start = int(load_start * LOADED_DATA_SAMPLE_RATE)
                datapoints_stop = int(load_stop * LOADED_DATA_SAMPLE_RATE)
                X = f['ts'][datapoints_start:datapoints_stop]

        # some statistics on the data
        data_statistics = 0
        if data_statistics:
            print(f'start: {start}, end: {end}')
            print(f'duration, seconds: {end-start}')
            print(f'data length: {len(X)}')
            print(f'with data sampled at {4*SAMPLE_RATE}, len(data)/duration= {len(X)/4/SAMPLE_RATE}')

        sample_rate = LOADED_DATA_SAMPLE_RATE
        resample_rate = SAMPLE_RATE  # don't need so many samples

        data = TimeSeries(X, sample_rate=sample_rate, t0=start)
        if data_statistics:
            print(f'after creating time series, len(data) = {len(data)}')
            before_resample = len(data)

        if sample_rate != resample_rate:
            data = data.resample(resample_rate)

        if data_statistics:
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
    '''
    sort the glitches by SNR, and return the N loudest ones
    If there are < N glitches, return the number of glitches, i.e. will not crash
    '''
    glitch_times = ifo_data['triggers']['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data']) / ifo_data['data'].sample_rate.value

    glitch_snrs = ifo_data['triggers']['snr']
    # create a sorting, in descending order
    sort_by_snr = glitch_snrs.argsort()[::-1]
    glitch_times_sorted = glitch_times[sort_by_snr]

    glitch_start_times = []
    for i in range(min(len(glitch_times_sorted), N)):
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

    return valid_start_times


def slice_bkg_segments(
        ifo_data: dict,
        data,
        start_times: list[float],
        segment_length: float):
    # turn the start times into background segments
    fs = ifo_data['data'].sample_rate.value
    t0 = ifo_data['data'].t0.value
    N_datapoints = int(segment_length * fs)
    bkg_segs = np.zeros(shape=(2, len(start_times), N_datapoints))
    bkg_timeseries = []

    for i in range(len(start_times)):
        slx = slice(int((start_times[i] - t0) * fs),
                    int((start_times[i] - t0) * fs) + N_datapoints)
        slice_segment = data[:, slx]
        toggle_noise = 1
        bkg_segs[:, i] = np.array(slice_segment) * toggle_noise
        bkg_timeseries.append(slice_segment)

    return bkg_segs, bkg_timeseries


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
        L1_asd,
        clip_edge: int=1):

    ASDs = {'H1': H1_asd,
            'L1': L1_asd}
    all_white_segs = []
    for i, ifo in enumerate(IFOS):
        bkg_segs = bkg_segs_full[i]
        final_shape = (bkg_segs.shape[0], bkg_segs.shape[
                       1] - 2 * int(clip_edge * sample_rate))
        white_segs = np.zeros(final_shape)
        for j, bkg_seg in enumerate(bkg_segs):
            if BANDPASS_HIGH == SAMPLE_RATE//2:
                # On attempting to bandpass with Nyquist frequency, an
                # error is thrown. This was only the BANDPASS_LOW is used
                # with a highpass filter
                white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(
                    asd=ASDs[ifo]).highpass(BANDPASS_LOW)
            else:
                white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(
                    asd=ASDs[ifo]).bandpass(BANDPASS_LOW, BANDPASS_HIGH)
            white_segs[j] = clipping(
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
        single_detec_data = detec_data[ifo]['data']

        fft_of_template = np.fft.rfft(data)
        fft_of_template = np.abs(fft_of_template) / fs
        # calculate detector psd

        if detector_psds is None:
            # this is the desired df value
            detec_psd = calc_psd(single_detec_data, df)
        else:
            detec_psd = detector_psds[ifo]
        if return_psds:
            save_psds[ifo] = detec_psd
        fft_data = fft_of_template
        integrand = fft_data / detec_psd ** 0.5
        integrand = integrand ** 2

        integrated = integrand.sum(axis=-1) * df
        integrated = 4 * integrated
        snr_tot = snr_tot + integrated  # adding SNR^2

    if return_psds:
        return snr_tot**0.5, save_psds
    return snr_tot**0.5


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
            response = bilby_ifo.antenna_response(  # this gives just a float, my interpretation is the
                ra, dec, geocent_time, psi, mode)  # inclination angle of the detector to the source

            midp = len(injection) // 2
            if len(polarization) % 2 == 1:
                polarization = polarization[:-1]
            half_polar = len(polarization) // 2
            # off by one datapoint if this fails, don't think it matters
            assert len(polarization) % 2 == 0
            slx = slice(midp - half_polar, midp + half_polar)
            if INJECT_AT_END:
                center = int(len(injection) - SAMPLE_RATE*EDGE_INJECT_SPACING - half_polar)
                slx = slice(center-half_polar, center+half_polar)

            injection[slx] += response * polarization



        signal = TimeSeries(
            injection, times=full_seg.times, unit=full_seg.unit)

        full_seg = full_seg.inject(signal)
        nonoise_seg = nonoise_seg.inject(signal)
        final_injects.append(full_seg)
        final_injects_nonoise.append(nonoise_seg)

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
            if INJECT_AT_END:
                center = int(len(injection) - SAMPLE_RATE*EDGE_INJECT_SPACING - half_polar)
                slx = slice(center-half_polar, center+half_polar)
            injection[slx] += response * polarization

        signal = TimeSeries(
            injection, times=full_seg.times, unit=full_seg.unit)

        full_seg = full_seg.inject(signal)
        nonoise_seg = nonoise_seg.inject(signal)
        final_injects.append(full_seg)
        final_injects_nonoise.append(nonoise_seg)

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

    plus[:len(hplus.data.data)] = hplus.data.data
    cross[:len(hcross.data.data)] = hcross.data.data

    return dict(plus=plus, cross=cross)


def olib_freq_domain_sine_gaussian(
        freq_array,
        hrss,
        q,
        frequency,
        phase,
        eccentricity,
        geocent_time,
        **kwargs):

    deltaF = freq_array[1] - freq_array[0]
    deltaT = 0.5 / (freq_array[-1])
    hplus, hcross = BurstSineGaussianF(
        q, frequency, hrss, eccentricity, phase, deltaF, deltaT
    )

    plus = np.zeros(len(freq_array), dtype=complex)
    cross = np.zeros(len(freq_array), dtype=complex)

    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data

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

def clean_gw_events(
    event_times,
    data,
    ts,
    tend):

    
    convert_index = lambda t: int(SAMPLE_RATE * (t-ts))
    bad_times = []
    for et in event_times:
        # only take the GW events past 5 second edge, otherwise there will 
        # be problems with removing it from the segment
        if et > ts+GW_EVENT_CLEANING_WINDOW and et < tend-GW_EVENT_CLEANING_WINDOW:
            # convert the GPS time of GW events into indicies
            bad_times.append( convert_index(et))

    clean_window = int(GW_EVENT_CLEANING_WINDOW*SAMPLE_RATE) #seconds

    # enforce even clean window so that it is symmetric
    clean_window = (clean_window // 2) *2
    if len(bad_times) == 0:
        # no GW events to excise
        return data[:, clean_window:-clean_window]

    # just cut off the edge instead of dealing with BBH's there
    cleaned_data = np.zeros(shape=(2, int(data.shape[1]-clean_window*len(bad_times) )) )

    # start, stop correspond to the indicies for data (original)
    start = 0
    stop = 0
    # marker corresponds to indicies for the new cleaned_seg
    marker = 0
    for event_time in bad_times:
        stop = int(event_time - clean_window//2)
        seg_len = stop - start
        # fill in the data up to the start time of BBH event
        cleaned_data[:, marker:marker+seg_len] = data[:, start:stop]
        marker += seg_len
        start = int(event_time + clean_window//2)

    # return, while chopping off the first and last 5 seconds
    # since those weren't treated for potential GW events
    return cleaned_data[:, clean_window:-clean_window]

def split_into_segments(data, 
                        overlap=SEGMENT_OVERLAP, 
                        seg_len=SEG_NUM_TIMESTEPS):
    '''
    Function to slice up data into overlapping segments
    seg_len: length of resulting segments
    overlap: overlap of the windows in units of indicies

    assuming that data is of shape (N_samples, axis_to_slice_on, features)
    '''
    print("data segment input shape", data.shape)
    N_slices = (data.shape[1]-seg_len)//overlap
    print("N slices,", N_slices)
    print("going to make it to, ", N_slices*overlap+seg_len)
    data = data[:, :N_slices*overlap+seg_len]

    result = np.empty((data.shape[0], N_slices, seg_len, data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(N_slices):
            start = j*overlap
            end = j*overlap + seg_len
            #print("SHAPES 21", result[i, j, :, :].shape,data[i, start:end, :].shape)
            result[i, j, :, :] = data[i, start:end, :]

    return result

def split_into_segments_torch(data, 
                        overlap=SEGMENT_OVERLAP, 
                        seg_len=SEG_NUM_TIMESTEPS):
    '''
    Function to slice up data into overlapping segments
    seg_len: length of resulting segments
    overlap: overlap of the windows in units of indicies

    assuming that data is of shape (N_samples, features, axis_to_slice_on)
    '''
    N_slices = (data.shape[2]-seg_len)//overlap
    data = data[:, :, :N_slices*overlap+seg_len]

    device = DEVICE

    # resulting shape: (batches, N_slices, 2, 100)
    result = torch.empty((data.shape[0], N_slices, data.shape[1], seg_len), 
                        device=device)

    for i in range(data.shape[0]):
        for j in range(N_slices):
            start = j*overlap
            end = j*overlap + seg_len
            result[i, j, :, :] = data[i, :, start:end]

    return result
