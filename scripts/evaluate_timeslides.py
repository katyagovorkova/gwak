import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from final_metric_optimization import LinearModel
from evaluate_data import full_evaluation
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    FM_TIMESLIDE_TOTAL_DURATION,
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    GPU_NAME,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    RETURN_INDIV_LOSSES
)
DEVICE = torch.device(GPU_NAME)


def engineered_features(data):

    newdata = np.zeros(data.shape)

    for i in range(4):
        a, b = data[:, :, 2 * i], data[:, :, 2 * i + 1]
        newdata[:, :, 2 * i] = (a + b) / 2
        newdata[:, :, 2 * i + 1] = abs(a - b)  # / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    return newdata


def engineered_features_torch(data):

    newdata = torch.zeros(data.shape).to(DEVICE)

    for i in range(4):
        a, b = data[:, :, 2 * i], data[:, :, 2 * i + 1]
        newdata[:, :, 2 * i] = (a + b) / 2
        newdata[:, :, 2 * i + 1] = abs(a - b)  # / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    return newdata


def main(args):
    if args.metric_coefs_path is not None:
        # initialize histogram
        n_bins = 2 * int(HISTOGRAM_BIN_MIN / HISTOGRAM_BIN_DIVISION)
        hist = np.zeros(n_bins)
        np.save(args.save_path, hist)

    data = np.load(args.data_path)['data']
    data = torch.from_numpy(data).to(DEVICE)

    reduction = 20  # for things to fit into memory nicely

    timeslide_total_duration = TIMESLIDE_TOTAL_DURATION
    if args.fm_shortened_timeslides:
        timeslide_total_duration = FM_TIMESLIDE_TOTAL_DURATION

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(timeslide_total_duration // sample_length) * reduction
    print('Number of timeslides:', n_timeslides)
    for timeslide_num in range(1, n_timeslides + 1):
        print(f'starting timeslide: {timeslide_num}/{n_timeslides}')

        indicies_to_slide = np.random.uniform(
            SAMPLE_RATE, data.shape[1] - SAMPLE_RATE)
        indicies_to_slide = int(indicies_to_slide)
        timeslide = torch.empty(data.shape, device=DEVICE)

        # hanford unchanged
        timeslide[0, :] = data[0, :]

        # livingston slid
        timeslide[1, :indicies_to_slide] = data[1, -indicies_to_slide:]
        timeslide[1, indicies_to_slide:] = data[1, :-indicies_to_slide]

        # make a random cut with the reduced shape
        reduced_len = int(data.shape[1] / reduction)
        start_point = int(np.random.uniform(
            0, data.shape[1] - SAMPLE_RATE - reduced_len))
        timeslide = timeslide[:, start_point:start_point + reduced_len]

        timeslide = timeslide[:, :(timeslide.shape[1] // 1000) * 1000]
        final_values = full_evaluation(
            timeslide[None, :, :], args.model_folder_path)

        if args.metric_coefs_path is not None:

            # compute the dot product and save that instead
            metric_vals = np.load(args.metric_coefs_path)
            norm_factors = np.load(args.norm_factor_path)
            metric_vals = torch.from_numpy(metric_vals).float().to(DEVICE)
            norm_factors = torch.from_numpy(norm_factors).float().to(DEVICE)

            # flatten batch dimension
            final_values = torch.reshape(final_values, (final_values.shape[
                                         0] * final_values.shape[1], final_values.shape[2]))
            means, stds = norm_factors[0], norm_factors[1]
            final_values = (final_values - means) / stds

            if RETURN_INDIV_LOSSES:
                model = LinearModel(21).to(DEVICE)
                model.load_state_dict(torch.load(
                    args.fm_model_path, map_location=GPU_NAME))
                final_values = model(final_values).detach()
            else:

                final_values = torch.matmul(final_values, metric_vals)

            update = torch.histc(final_values, bins=n_bins,
                                 min=-HISTOGRAM_BIN_MIN, max=HISTOGRAM_BIN_MIN)
            past_hist = np.load(args.save_path)
            new_hist = past_hist + update.cpu().numpy()
            np.save(args.save_path, new_hist)

        else:
            print('saving, individually')
            means, stds = torch.mean(
                final_values, axis=-2), torch.std(final_values, axis=-2)
            means, stds = means.detach().cpu().numpy(), stds.detach().cpu().numpy()
            np.save(f'{args.save_path}/normalization_params_{timeslide_num}.npy', np.stack([means, stds], axis=0))
            final_values = final_values.detach().cpu().numpy()

            # save as a numpy file, with the index of timeslide_num
            np.save(f'{args.save_path}/timeslide_evals_{timeslide_num}.npy', final_values)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str,
                        help='Directory containing the timeslides')

    parser.add_argument('save_path', type=str,
                        help='Folder to which save the timeslides')

    parser.add_argument('model_folder_path', nargs='+', type=str,
                        help='Path to the folder containing the models')

    # Additional arguments
    parser.add_argument('--fm-model-path', type=str,
                        help='Final metric model')

    parser.add_argument('--metric-coefs-path', type=str, default=None,
                        help='Pass in path to metric coefficients to compute dot product'
                        )
    parser.add_argument('--norm-factor-path', type=str, default=None,
                        help='Pass in path to significance normalization factors')

    parser.add_argument('--fm-shortened-timeslides', type=str, default='False',
                        help='Generate reduced timeslide samples to train final metric')

    args = parser.parse_args()
    args.fm_shortened_timeslides = args.fm_shortened_timeslides == 'True'
    main(args)
