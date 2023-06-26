import os
import numpy as np
import bilby
import argparse

from typing import Callable
from helper_functions import (
    load_folder,
    whiten_bandpass_bkgs,
    olib_time_domain_sine_gaussian,
    inject_hplus_hcross,
    get_background_segs,
    WNB,
    clean_gw_events,
    get_loud_segments,
    slice_bkg_segments
    )

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    IFOS,
    SEG_NUM_TIMESTEPS,
    SAMPLE_RATE,
    STRAIN_START,
    N_TRAIN_INJECTIONS,
    N_TEST_INJECTIONS,
    N_FM_INJECTIONS,
    DATA_SEGMENT_LOAD_START,
    DATA_SEGMENT_LOAD_STOP,
    TRAIN_INJECTION_SEGMENT_LENGTH,
    FM_INJECTION_SEGMENT_LENGTH,
    BBH_WINDOW_LEFT,
    BBH_WINDOW_RIGHT,
    BBH_AMPLITUDE_BAR,
    BBH_N_SAMPLES,
    SG_WINDOW_LEFT,
    SG_WINDOW_RIGHT,
    SG_AMPLITUDE_BAR,
    SG_N_SAMPLES,
    GLITCH_WINDOW_LEFT,
    GLITCH_WINDOW_RIGHT,
    GLITCH_N_SAMPLES,
    GLITCH_AMPLITUDE_BAR,
    BKG_N_SAMPLES,
    FM_INJECTION_SNR,
    N_VARYING_SNR_INJECTIONS,
    VARYING_SNR_DISTRIBUTION,
    VARYING_SNR_LOW,
    VARYING_SNR_HIGH,
    VARYING_SNR_SEGMENT_INJECTION_LENGTH,
    CURRICULUM_SNRS
    )


def generate_timeslides(
    folder_path:str,
    event_times_path:str):
    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    event_times = np.load(event_times_path)

    data = data[:, np.newaxis, :]
    whitened = whiten_bandpass_bkgs(data, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
    whitened = np.swapaxes(whitened, 0, 1)[0] # batch dimension removed

    data_cleaned = clean_gw_events(event_times,
                                  whitened,
                                  STRAIN_START+DATA_SEGMENT_LOAD_START,
                                  STRAIN_START+DATA_SEGMENT_LOAD_STOP)
    return data_cleaned

def bbh_polarization_generator(
    n_injections,
    segment_length=2):

    bbh_waveform_args = dict(waveform_approximant='IMRPhenomPv2',
                             reference_frequency=50., minimum_frequency=20.)
    waveform = bilby.gw.source.lal_binary_black_hole
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=segment_length,
        sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=waveform,
        waveform_arguments=bbh_waveform_args
    )
    priors = bilby.gw.prior.BBHPriorDict()
    injection_parameters = priors.sample(n_injections)
    injection_parameters = [
        dict(zip(injection_parameters, col))
        for col in zip(*injection_parameters.values())
    ]
    for i, p in enumerate(injection_parameters):
        dist = np.random.uniform(50, 200)
        p['luminosity_distance'] = dist

    crosses = []
    plusses = []
    for i, p in enumerate(injection_parameters):
        p['geocent_time'] = 0
        pols = waveform_generator.time_domain_strain(p)
        cross = pols['cross']
        plus = pols['plus']

        crosses.append(cross)
        plusses.append(plus)

    crosses = np.vstack(crosses)
    plusses = np.vstack(plusses)

    half = crosses.shape[1] // 2

    crosses = np.hstack([crosses[:, half:], crosses[:, :half]])
    plusses = np.hstack([plusses[:, half:], plusses[:, :half]])

    return [crosses, plusses]

def sg_polarization_generator(
    n_injections,
    segment_length=2):

    waveform = olib_time_domain_sine_gaussian
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=segment_length,
        sampling_frequency=SAMPLE_RATE,
        time_domain_source_model=waveform,
        waveform_arguments=None
    )
    prior_file = 'data/SG.prior'
    priors = bilby.gw.prior.PriorDict(prior_file)
    injection_parameters = priors.sample(n_injections)
    injection_parameters = [
        dict(zip(injection_parameters, col))
        for col in zip(*injection_parameters.values())
    ]
    crosses = []
    plusses = []
    for i, p in enumerate(injection_parameters):
        p['geocent_time'] = 0
        pols = waveform_generator.time_domain_strain(p)

        peak = np.argmax((pols['cross']))

        half = len(pols['cross']) // 2
        cross = pols['cross']
        plus = pols['plus']
        cross = np.concatenate(
            [cross[(half + peak):], cross[:(half + peak)]])
        plus = np.concatenate([plus[(half + peak):], plus[:(half + peak)]])
        crosses.append(cross)
        plusses.append(plus)

    crosses = np.vstack(crosses)
    plusses = np.vstack(plusses)

    return [crosses, plusses]

def wnb_polarization_generator(
    n_injections,
    fmin=400,
    fmax=1000,
    duration=1): # in seconds

    # merge with background
    wnb_hplus = np.zeros((n_injections, SAMPLE_RATE))
    wnb_hcross = np.zeros((n_injections, SAMPLE_RATE))
    for i in range(n_injections):
        wnb = WNB(duration, SAMPLE_RATE, fmin, fmax,
            enveloped=True, sidePad=(SAMPLE_RATE-SAMPLE_RATE*duration)//2)
        if wnb[0].shape[0]==4095:
            gen_wnb_hplus = np.append(wnb[0], np.array([0]))
            gen_wnb_hcross = np.append(wnb[1], np.array([0]))
        elif wnb[0].shape[0]==4094:
            gen_wnb_hplus = np.append(wnb[0], np.array([0,0]))
            gen_wnb_hcross = np.append(wnb[1], np.array([0,0]))
        else:
            gen_wnb_hplus = wnb[0]
            gen_wnb_hcross = wnb[1]

        wnb_hplus[i,:] = gen_wnb_hplus
        wnb_hcross[i,:] = gen_wnb_hcross

    return wnb_hcross, wnb_hplus

def inject_signal(
        folder_path: str,  # source of detector data, includes detector data and the omicron glitches/corresponding SNRs
        # source of the polarization files to be injected into the data
        data=None,
        SNR=None,
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH, # length of background segment to fetch for each injection
        inject_at_end=False,
        return_injection_snr=False):

    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    polarizations = []
    crosses, plusses = data
    for i in range(len(plusses)):
        polarizations.append({'plus': plusses[i], 'cross': crosses[i]})
    print(f'requested shape {len(polarizations) * 1}')
    bkg_segs = get_background_segs(loaded_data, detector_data, len(
        polarizations) * 1, segment_length)
    detector_psds = inject_hplus_hcross(bkg_segs[:, 0, :],
                                        polarizations[0],
                                        SAMPLE_RATE,
                                        segment_length,
                                        SNR=1,
                                        background=loaded_data,
                                        get_psds=True,
                                        inject_at_end=inject_at_end)
    print(f'background segments shape {bkg_segs.shape}')
    final_data = []
    sampled_SNR = []
    for i, pols in enumerate(polarizations):
        # didn't generate enough bkg samples, this is generally fine unless
        # small overall samples
        if i >= bkg_segs.shape[1]:
            break
        sample_snr = None
        if SNR is not None:
            sample_snr = SNR()
            sampled_SNR.append(sample_snr)
        for j in range(1):
            injected_waveform, _ = inject_hplus_hcross(bkg_segs[:, i, :],
                                                       pols,
                                                       SAMPLE_RATE,
                                                       segment_length,
                                                       SNR=sample_snr,
                                                       background=loaded_data,
                                                       detector_psds=detector_psds,
                                                       inject_at_end=inject_at_end)
            bandpass_segs = whiten_bandpass_bkgs(injected_waveform, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
            final_data.append(bandpass_segs)

    if return_injection_snr:
        return np.hstack(final_data), np.array(sampled_SNR)
    return np.hstack(final_data)

def inject_signal_curriculum(
        folder_path: str,  # source of detector data, includes detector data and the omicron glitches/corresponding SNRs
        # source of the polarization files to be injected into the data
        data=None,
        SNR=None,
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH, # length of background segment to fetch for each injection
        inject_at_end=False,
        return_injection_snr=False):
    
    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    polarizations = []
    crosses, plusses = data
    for i in range(len(plusses)):
        polarizations.append({'plus': plusses[i], 'cross': crosses[i]})
    #print(f'requested shape {len(polarizations) * 1}')
    bkg_segs = get_background_segs(loaded_data, detector_data, len(
        polarizations) * 1, segment_length)
    detector_psds = inject_hplus_hcross(bkg_segs[:, 0, :],
                                        polarizations[0],
                                        SAMPLE_RATE,
                                        segment_length,
                                        SNR=1,
                                        background=loaded_data,
                                        get_psds=True,
                                        inject_at_end=inject_at_end)
    #print(f'background segments shape {bkg_segs.shape}')
    final_data = []
    sampled_SNR = []
    for i, pols in enumerate(polarizations):
        # didn't generate enough bkg samples, this is generally fine unless
        # small overall samples
        if i >= bkg_segs.shape[1]:
            break
        injected_waveforms, clean_waveforms = inject_hplus_hcross(bkg_segs[:, i, :],
                                                    pols,
                                                    SAMPLE_RATE,
                                                    segment_length,
                                                    SNR=np.array(CURRICULUM_SNRS),
                                                    background=loaded_data,
                                                    detector_psds=detector_psds,
                                                    inject_at_end=inject_at_end)
        bandpass_segs = whiten_bandpass_bkgs(injected_waveform, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
        final_data.append(bandpass_segs)

    if return_injection_snr:
        return np.hstack(final_data), np.array(sampled_SNR)
    return np.hstack(final_data)

def generate_backgrounds(
        folder_path: str,
        n_backgrounds: int,
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH):

    loaded_data = load_folder(folder_path,
                              DATA_SEGMENT_LOAD_START,
                              DATA_SEGMENT_LOAD_STOP)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    bkg_segs = get_background_segs(
        loaded_data, detector_data, n_backgrounds, segment_length)
    whitened_segs = whiten_bandpass_bkgs(bkg_segs, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
    return whitened_segs

def generate_glitches(
        folder_path: str,
        n_glitches: int,
        segment_length=TRAIN_INJECTION_SEGMENT_LENGTH,
        load_start=DATA_SEGMENT_LOAD_START,
        load_stop=DATA_SEGMENT_LOAD_STOP):

    loaded_data = load_folder(folder_path,
                              load_start,
                              load_stop)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    N = n_glitches
    loud_times_H1 = get_loud_segments(loaded_data["H1"], N, segment_length)
    loud_times_L1 = get_loud_segments(loaded_data["L1"], N, segment_length)
    loud_times = np.union1d(loud_times_H1, loud_times_L1)
    glitch_segs, _ = slice_bkg_segments(loaded_data["H1"], detector_data, loud_times,
                                    segment_length)
    whitened_segs = whiten_bandpass_bkgs(glitch_segs, SAMPLE_RATE, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"])
    return whitened_segs

def sampler(
        data,
        n_samples: int,
        bound_low: int,
        bound_high: int,
        amplitude_bar: int,
        sample_len=SEG_NUM_TIMESTEPS):

    fill = np.empty((len(data) * n_samples, 2, sample_len))
    print("SAMPLER fill shape", fill.shape)
    print("len(data), n_samples, sample_len", len(data), n_samples, sample_len)
    midp = data.shape[-1] // 2
    filled_count = 0
    for n in range(len(data)):
        for j in range(n_samples):
            max_amp = -1
            attempts = 0
            while max_amp < amplitude_bar and attempts < n_samples * 2:
                attempts += 1
                start_index = int(np.random.uniform(
                    bound_low, bound_high - sample_len))
                start_index += midp
                segment = data[n, :, start_index:start_index + sample_len]
                max_amp = np.amax(np.abs(segment))
            if max_amp >= amplitude_bar:
                # at this point, it's passed the amplitude test
                fill[filled_count, :, :] = segment
                filled_count += 1

    return fill[:filled_count]


def sample_injections_main(
        source: str,  # directory containing the injection files
        # list of classes on which you would like to perform preparation for
        # training
        target_class: str,
        sample_len: int=SEG_NUM_TIMESTEPS,
        data=None):

    sampler_args = {
        'bbh': [BBH_N_SAMPLES, int(BBH_WINDOW_LEFT*SAMPLE_RATE),
                int(BBH_WINDOW_RIGHT*SAMPLE_RATE), BBH_AMPLITUDE_BAR],
        'sg': [SG_N_SAMPLES, int(SG_WINDOW_LEFT*SAMPLE_RATE),
                int(SG_WINDOW_RIGHT*SAMPLE_RATE), SG_AMPLITUDE_BAR],
        'background': [BKG_N_SAMPLES, None, None, 0],
        'glitch': [GLITCH_N_SAMPLES, int(GLITCH_WINDOW_LEFT*SAMPLE_RATE),
                   int(GLITCH_WINDOW_RIGHT * SAMPLE_RATE), GLITCH_AMPLITUDE_BAR]
        }

    data = data.swapaxes(0, 1)
    N_samples, bound_low, bound_high, amplitude_bar = sampler_args[
        target_class]
    
    if bound_low is None:
        assert bound_high is None
        midp = data.shape[-1] // 2
        bound_low = -midp
        bound_high = midp - sample_len

    training_data = sampler(
        data, N_samples, bound_low, bound_high, amplitude_bar)

    print("out from sample injections", training_data.shape)
    return training_data

def make_snr_sampler(distribution, low, hi):
    if distribution == "uniform":
        def sampler():
            return np.random.uniform(low, hi)
    else:
        print("Invalid or unimplemented distribution choice", distribution)
        assert False

    return sampler
def main(args):
    sampled_snr = None

    if args.stype == 'bbh':
        # 1: generate the polarization files for the signal classes of interest
        BBH_cross, BBH_plus = bbh_polarization_generator(N_TRAIN_INJECTIONS)

        # 2: create the injections with those signal classes
        BBH_injections = inject_signal(folder_path=args.folder_path,
                                      data=[BBH_cross, BBH_plus],
                                      )
        # 3: Turn the injections into segments, ready for training
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               data=BBH_injections)

    elif args.stype == 'sg':
        # 1: generate the polarization files for the signal classes of interest
        SG_cross, SG_plus = sg_polarization_generator(N_TRAIN_INJECTIONS)

        # 2: create the injections with those signal classes
        SG_injections = inject_signal(folder_path=args.folder_path,
                                     data=[SG_cross, SG_plus])
        # 3: Turn the injections into segments, ready for training
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               data=SG_injections)

    elif args.stype == 'wnb':
        # 1: generate the polarization files for the signal classes of interest
        WNB_cross, WNB_plus = wnb_polarization_generator(N_TEST_INJECTIONS)

        # 2: create the injections with those signal classes
        training_data = inject_signal(folder_path=args.folder_path,
                                     data=[WNB_cross, WNB_plus])

    elif args.stype == 'background':

        # 2.5: generate/fetch the background classes
        backgrounds = generate_backgrounds(folder_path=args.folder_path,
                                       n_backgrounds=N_TRAIN_INJECTIONS)
        # 3: Turn the injections into segments, ready for training
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               data=backgrounds)

    elif args.stype == 'glitch':
        if args.start is not None:
            assert args.stop is not None
            glitches = generate_glitches(folder_path=args.folder_path,
                                    n_glitches=N_TRAIN_INJECTIONS,
                                    load_start=int(args.start), load_stop=int(args.stop))
            args.save_file = f"{args.save_file[:-4]}_{args.start}_{args.stop}{args.save_file[-4:]}"
        else:
            glitches = generate_glitches(folder_path=args.folder_path,
                                        n_glitches=N_TRAIN_INJECTIONS)
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               data=glitches)

        print("FINAL SHAPE FROM GLITCHES", training_data.shape)

    elif args.stype == 'timeslides':
        event_times_path = '/home/ryan.raikman/s22/LIGO_EVENT_TIMES.npy'
        training_data = generate_timeslides(args.folder_path, event_times_path=event_times_path)

    elif args.stype == "bbh_fm_optimization":
        # 1: generate the polarization files for the signal classes of interest
        BBH_cross, BBH_plus = bbh_polarization_generator(N_FM_INJECTIONS)

        sampler = make_snr_sampler("uniform", FM_INJECTION_SNR, FM_INJECTION_SNR)
        # 2: create the injections with those signal classes
        BBH_injections = inject_signal(folder_path=args.folder_path,
                                      data=[BBH_cross, BBH_plus],
                                      segment_length=FM_INJECTION_SEGMENT_LENGTH,
                                      inject_at_end=True,
                                      SNR=sampler)
        training_data = BBH_injections.swapaxes(0, 1)

    elif args.stype == "sg_fm_optimization":
        # 1: generate the polarization files for the signal classes of interest
        SG_cross, SG_plus = sg_polarization_generator(N_FM_INJECTIONS)

        sampler = make_snr_sampler("uniform", FM_INJECTION_SNR, FM_INJECTION_SNR)
        # 2: create the injections with those signal classes
        SG_injections = inject_signal(folder_path=args.folder_path,
                                      data=[SG_cross, SG_plus],
                                      segment_length=FM_INJECTION_SEGMENT_LENGTH,
                                      inject_at_end=True,
                                      SNR=sampler)
        training_data = SG_injections.swapaxes(0, 1)

    elif args.stype == "bbh_varying_snr":
        # 1: generate the polarization files for the signal classes of interest
        BBH_cross, BBH_plus = bbh_polarization_generator(N_VARYING_SNR_INJECTIONS)

        sampler = make_snr_sampler(VARYING_SNR_DISTRIBUTION, VARYING_SNR_LOW, VARYING_SNR_HIGH)
        # 2: create the injections with those signal classes
        BBH_injections, sampled_snr = inject_signal(folder_path=args.folder_path,
                                      data=[BBH_cross, BBH_plus],
                                      segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                      inject_at_end=True,
                                      SNR=sampler,
                                      return_injection_snr = True)
        training_data = BBH_injections.swapaxes(0, 1)

    elif args.stype == "sg_varying_snr":
        # 1: generate the polarization files for the signal classes of interest
        SG_cross, SG_plus = sg_polarization_generator(N_VARYING_SNR_INJECTIONS)

        sampler = make_snr_sampler(VARYING_SNR_DISTRIBUTION, VARYING_SNR_LOW, VARYING_SNR_HIGH)
        # 2: create the injections with those signal classes
        SG_injections, sampled_snr = inject_signal(folder_path=args.folder_path,
                                      data=[SG_cross, SG_plus],
                                      segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                      inject_at_end=True,
                                      SNR=sampler,
                                      return_injection_snr = True)
        training_data = SG_injections.swapaxes(0, 1)

    elif args.stype == 'wnb_varying_snr':
        # 1: generate the polarization files for the signal classes of interest
        WNB_cross, WNB_plus = wnb_polarization_generator(N_TEST_INJECTIONS)

        sampler = make_snr_sampler(VARYING_SNR_DISTRIBUTION, VARYING_SNR_LOW, VARYING_SNR_HIGH)
        # 2: create the injections with those signal classes
        training_data, sampled_snr = inject_signal(folder_path=args.folder_path,
                                     data=[WNB_cross, WNB_plus],
                                     segment_length=VARYING_SNR_SEGMENT_INJECTION_LENGTH,
                                     inject_at_end = True,
                                     SNR=sampler,
                                     return_injection_snr=True)

    
    np.save(args.save_file, training_data)

    if sampled_snr is not None:
        snr_save_path = f"{args.save_file[:-4]}_SNR{args.save_file[-4:]}" # drop it in between name and .npy
        np.save(snr_save_path, sampled_snr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('folder_path', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('save_file', help='Where to save the file with injections',
                        type=str)

    parser.add_argument('--stype', help='Which type of the injection to generate',
                        type=str, choices=['bbh', 'sg', 'background',
                                           'glitch', 'wnb', 'ccsn', 'timeslides',
                                           'bbh_fm_optimization', 'sg_fm_optimization',
                                           'bbh_varying_snr', 'sg_varying_snr'])

    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--stop', type=str, default=None)
    args = parser.parse_args()
    main(args)
