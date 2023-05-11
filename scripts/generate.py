import os
import numpy as np
import bilby
import argparse

from typing import Callable
from helper_functions import (
<<<<<<< HEAD
    load_folder,
    whiten_bandpass_bkgs,
    olib_time_domain_sine_gaussian,
    inject_hplus_hcross,
    get_background_segs,
    WNB,
    )

from constants import (
    IFOS,
    SAMPLE_RATE,
    N_INJECTIONS
    )
=======
    olib_time_domain_sine_gaussian,
    inject_hplus_hcross,
)

SAMPLE_RATE = 4096
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

def bbh_polarization_generator(
    n_injections,
    segment_length=2):

<<<<<<< HEAD
=======
def bbh_polarization_generator(
    n_injections,
    segment_length=2):

>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
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
<<<<<<< HEAD
    prior_file = "SG.prior"
=======
    prior_file = "/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/SG_prior2.prior"
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
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
<<<<<<< HEAD

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
        segment_length=4):  # length of background segment to fetch for each injection

    loaded_data = load_folder(folder_path)
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
                                        get_psds=True)
    print(f'background segments shape {bkg_segs.shape}')
    final_data = []
    for i, pols in enumerate(polarizations):
        # didn't generate enough bkg samples, this is generally fine unless
        # small overall samples
        if i >= bkg_segs.shape[1]:
            break
        for j in range(1):
            injected_waveform, _ = inject_hplus_hcross(bkg_segs[:, i, :],
                                                       pols,
                                                       SAMPLE_RATE,
                                                       segment_length,
                                                       SNR=None,
                                                       background=loaded_data,
                                                       detector_psds=detector_psds)

            bandpass_segs = whiten_bandpass_bkgs(injected_waveform, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
            final_data.append(bandpass_segs)

    return np.hstack(final_data)


def generate_backgrounds(
        n_backgrounds: int,
        folder_path: str,
        segment_length=4):

    loaded_data = load_folder(folder_path)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    bkg_segs = get_background_segs(
        loaded_data, detector_data, n_backgrounds, segment_length)
    whitened_segs = whiten_bandpass_bkgs(bkg_segs, SAMPLE_RATE, loaded_data['H1']['asd'], loaded_data['L1']['asd'])
    return whitened_segs
=======


def inject_signal(
        folder_path: str,  # source of detector data, includes detector data and the omicron glitches/corresponding SNRs
        # source of the polarization files to be injected into the data
        data=None,
        segment_length=4):  # length of background segment to fetch for each injection

    loaded_data = load_folder(folder_path, IFOS)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])

    polarizations = []
    crosses, plusses = data
    for i in range(len(plusses)):
        polarizations.append({'plus': plusses[i], 'cross': crosses[i]})
    print('requested shape', len(polarizations) * 1)
    bkg_segs = get_bkg_segs(loaded_data, detector_data, len(
        polarizations) * 1, segment_length)
    detector_psds = inject_hplus_hcross(bkg_segs[:, 0, :],
                                        polarizations[0],
                                        SAMPLE_RATE,
                                        segment_length, return_loc=False, SNR=1, background=loaded_data, get_psds=True)
    print('bkg segs shape', bkg_segs.shape)
    final_data = []
    for i, pols in enumerate(polarizations):
        # didn't generate enough bkg samples, this is generally fine unless
        # small overall samples
        if i >= bkg_segs.shape[1]:
            break
        for j in range(1):
            injected_waveform, _ = inject_hplus_hcross(bkg_segs[:, i, :],
                                                       pols,
                                                       SAMPLE_RATE,
                                                       segment_length, return_loc=False, SNR=None, background=loaded_data, detector_psds=detector_psds)

            bandpass_segs = whiten_bandpass_bkgs(injected_waveform, SAMPLE_RATE, loaded_data[
                                                 'H1']['asd'], loaded_data['L1']['asd'])
            final_data.append(bandpass_segs)

    return np.hstack(final_data)


def main_backgrounds(
        n_backgrounds: int,
        folder_path: str,
        run_background: bool=True,
        run_glitch: bool=False):

    loaded_data = load_folder(folder_path, IFOS)
    detector_data = np.vstack([loaded_data['H1']['data'], loaded_data['L1']['data']])
    returns = []
    segment_length = 4
    if run_background:
        bkg_segs = get_bkg_segs(
            loaded_data, detector_data, n_backgrounds, segment_length)
        whitened_segs = whiten_bandpass_bkgs(bkg_segs, SAMPLE_RATE, loaded_data['H1'][
                                             'asd'], loaded_data['L1']['asd'])

        if save_dir is None:
            returns.append(whitened_segs)
        else:
            np.save(f'{save_dir}/bkg_segs.npy', whitened_segs)

    if run_glitch:
        loud_times_H1 = get_loud_segments(loaded_data['H1'], N, segment_length)
        loud_times_L1 = get_loud_segments(loaded_data['L1'], N, segment_length)
        loud_times = np.union1d(loud_times_H1, loud_times_L1)

        glitch_segs, _ = slice_bkg_segments(loaded_data['H1'], detector_data, loud_times,
                                            segment_length)
        whitened_segs = whiten_bandpass_bkgs(glitch_segs, SAMPLE_RATE, loaded_data['H1'][
                                             'asd'], loaded_data['L1']['asd'])
        returns.append(whitened_segs)

    return returns
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc


def sampler(
        data,
        n_samples: int,
        bound_low: int,
        bound_high: int,
        amplitude_bar: int,
        sample_len=100):
    fill = np.empty((len(data) * n_samples, 2, sample_len))
    midp = data.shape[-1] // 2
    filled_count = 0
    for n in range(len(data)):
        datum = data[n, :, :]
        for j in range(n_samples):
            max_amp = -1
            attempts = 0
            while max_amp < amplitude_bar and attempts < n_samples * 2:
                attempts += 1
                # print("iteration")
                start_index = int(np.random.uniform(
                    bound_low, bound_high - sample_len))
                start_index += midp
                #print("start index", start_index)
                segment = data[n, :, start_index:start_index + sample_len]
                max_amp = np.amax(np.abs(segment))
            if max_amp >= amplitude_bar:
                # at this point, it's passed the amplitude test
                fill[n * N_samples + j, :, :] = segment
                filled_count += 1

    return fill[:filled_count]


def sample_injections_main(
        source: str,  # directory containing the injection files
<<<<<<< HEAD
        # list of classes on which you would like to perform preparation for
        # training
        target_class: str,
        sample_len: int=100,
        data=None):

    sampler_args = {
        'BBH': [5, -150, 30, 5],
        'SG': [5, -200, 200, 5],
        'background': [5, None, None, 0],
        }
=======
        save_dir: str,  # directory to which save the training files
        # list of classes on which you would like to perform preparation for
        # training
        target_class: str,
        data=None):

    sample_len = 100
    sampler_args = {
        'BBH': [5, -150, 30, 5],
        'SG': [5, -200, 200, 5],
        'BKG': [5, None, None, 0],
        'GLITCH': [20, -100, 100, 5]
    }
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

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
<<<<<<< HEAD

    return training_data
=======
    np.save(f'{save_dir}/{target_class}.npy', training_data)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc


def main(args):

<<<<<<< HEAD
    if args.stype == 'bbh':
        # 1: generate the polarization files for the signal classes of interest
        BBH_cross, BBH_plus = bbh_polarization_generator(N_INJECTIONS)
=======
    if args.stype == 'BBH':
        # 1: generate the polarization files for the signal classes of interest
        BBH_cross, BBH_plus = bbh_polarization_generator(args.n_injections)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

        # 2: create the injections with those signal classes
        BBH_injections = inject_signal(folder_path=args.folder_path,
                                      data=[BBH_cross, BBH_plus])
        # 3: Turn the injections into segments, ready for training
<<<<<<< HEAD
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               data=BBH_injections)

    elif args.stype == 'sg':
        # 1: generate the polarization files for the signal classes of interest
        SG_cross, SG_plus = sg_polarization_generator(N_INJECTIONS)
=======
        sample_injections_main(source=None, save_dir=args.save_dir,
                               target_classes='BBH',
                               data=BBH_injections)

    elif args.stype == 'SG':
        # 1: generate the polarization files for the signal classes of interest
        SG_cross, SG_plus = sg_polarization_generator(args.n_injections)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc

        # 2: create the injections with those signal classes
        SG_injections = inject_signal(folder_path=args.folder_path,
                                     data=[SG_cross, SG_plus])
        # 3: Turn the injections into segments, ready for training
<<<<<<< HEAD
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               data=SG_injections)

    elif args.stype == 'wnb':
        # 1: generate the polarization files for the signal classes of interest
        WNB_cross, WNB_plus = wnb_polarization_generator(N_INJECTIONS)

        # 2: create the injections with those signal classes
        training_data = inject_signal(folder_path=args.folder_path,
                                     data=[WNB_cross, WNB_plus])

    elif args.stype == 'background':

        # 2.5: generate/fetch the background classes
        backgrounds = generate_backgrounds(folder_path=args.folder_path,
                                       n_backgrounds=N_INJECTIONS,
                                       run_background=True)
        # 3: Turn the injections into segments, ready for training
        training_data = sample_injections_main(source=None,
                               target_class=args.stype,
                               direct_data=backgrounds)

    elif args.stype == 'glitch':
        # 3.5: additionally, save the previously generated glitches to that same destination
        training_data = np.load(
            '/home/ryan.raikman/s22/training_files/3_26_train/GLITCH.npy')

    np.save(args.save_file, training_data)
=======
        sample_injections_main(source=None, save_dir=args.save_dir,
                               target_classes='SG',
                               data=SG_injections)

    elif args.stype == 'background':

        # 2.5: generate/fetch the background classes
        backgrounds = main_backgrounds(save_dir=None, folder_path=args.folder_path,
                                       n_backgrounds=args.n_injections,
                                       run_background=True, run_glitch=False)[0]
        # 3: Turn the injections into segments, ready for training
        sample_injections_main(source=None, save_dir=args.save_dir,
                               target_classes='BKG',
                               direct_data=backgrounds)

    elif args.stype == 'glitch':
        # 3.5: additionally, save the previously generated glitches to that same
        # destination
        GLITCHES = np.load(
            '/home/ryan.raikman/s22/training_files/3_26_train/GLITCH.npy')
        np.save(f'{save_dir}/GLITCHES.npy', GLITCHES)
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
<<<<<<< HEAD
    parser.add_argument('folder_path', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('save_file', help='Where to save the file with injections',
                        type=str)

    parser.add_argument('--stype', help='Which type of the injection to generate',
                        type=str, choices=['bbh', 'sg', 'background', 'glitch', 'wnb', 'ccsn'])
    args = parser.parse_args()
=======
    parser.add_argument('folder_path', help='-',
                        type=str)
    parser.add_argument('save_dir', help='Where to save injections',
                        type=str)
    parser.add_argument('stype', help='Which type of the injection to generate',
                        type=str, choices=['bbh', 'sg', 'background', 'glitch', 'wnb', 'ccsn'])

    # Additional arguments
    parser.add_argument('--n-injections', help='How many injections to generate',
                        type=int, default=500)
    # parser.add_argument('--data-path', help='Where is the data to do train/test split on',
    #     type=str)
    # args = parser.parse_args()
>>>>>>> de51ac6ce5815e66547b5478cf90dae4b89f19cc
    main(args)
