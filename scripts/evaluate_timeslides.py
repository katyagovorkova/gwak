import os
import argparse
import numpy as np
import time

import torch

from evaluate_data import full_evaluation
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    TIMESLIDE_STEP,
    FM_TIMESLIDE_TOTAL_DURATION,
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    GPU_NAME,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN
    )
DEVICE = torch.device(GPU_NAME)

def main(args):
    if args.metric_coefs_path is not None:
        # initialize histogram
        n_bins = 2*int(HISTOGRAM_BIN_MIN/HISTOGRAM_BIN_DIVISION)
        hist = np.zeros(n_bins)
        np.save(args.save_path, hist)

    data = np.load(args.data_path)
    data = torch.from_numpy(data).to(DEVICE)

    timeslide_total_duration = TIMESLIDE_TOTAL_DURATION
    if args.fm_shortened_timeslides:
        timeslide_total_duration = FM_TIMESLIDE_TOTAL_DURATION

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(timeslide_total_duration // sample_length)
    print("Number of timeslides:", n_timeslides)
    for timeslide_num in range(1, n_timeslides+1):
        print(f"starting timeslide: {timeslide_num}/{n_timeslides}")
        ts = time.time()
        indicies_to_slide = int(timeslide_num*TIMESLIDE_STEP*SAMPLE_RATE)
        timeslide = torch.empty(data.shape, device = DEVICE)

        # hanford unchanged
        timeslide[0, :] = data[0, :]
        # livingston slid
        timeslide[1, :indicies_to_slide] = data[1, -indicies_to_slide:]
        timeslide[1, indicies_to_slide:] = data[1, :-indicies_to_slide] 
        timeslide = timeslide[:, :(timeslide.shape[1]//1000)*1000]

        final_values = full_evaluation(timeslide[None, :, :], args.model_folder_path)

        if args.metric_coefs_path is not None:
            # compute the dot product and save that instead
            metric_vals = np.load(args.metric_coefs_path)
            metric_vals = torch.from_numpy(metric_vals).float().to(DEVICE)
            # flatten batch dimension
            final_values = torch.reshape(final_values, (final_values.shape[0]*final_values.shape[1], final_values.shape[2]))
            final_values = torch.matmul(final_values, metric_vals)

            update = torch.histc(final_values, bins=n_bins, 
                                 min=-HISTOGRAM_BIN_MIN, max=HISTOGRAM_BIN_MIN)
            past_hist = np.load(args.save_path)
            print("total filled:", np.sum(past_hist))
            new_hist = past_hist + update.cpu().numpy()
            np.save(args.save_path, new_hist)

        else:
            final_values = final_values.detach().cpu().numpy()
            # save as a numpy file, with the index of timeslide_num
            np.save(f"{args.save_path}/timeslide_evals_{timeslide_num}.npy", final_values)
        
        print(f"Iteration, {timeslide_num}, done in {time.time() - ts :.3f} s")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', help='Directory containing the timeslides',
        type=str)
    
    parser.add_argument('save_path', help = "Folder to which save the timeslides",
        type=str)
        
    parser.add_argument('model_folder_path', help = "Path to the folder containing the models",
        nargs="+", type=str)

    # Additional arguments
    parser.add_argument('--metric-coefs-path', help="Pass in path to metric coefficients to compute dot product",
                        type = str, default=None)
    
    parser.add_argument('--fm-shortened-timeslides', help="Generate reduced timeslide samples to train final metric",
                        type = str, default="False")
    
    args = parser.parse_args()
    args.fm_shortened_timeslides = args.fm_shortened_timeslides == "True"
    main(args)

