import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from quak_predict import quak_eval
from pearson import pearson_computation

from helper_functions import (
    mae, 
    std_normalizer_torch, 
    split_into_segments_torch,
    stack_dict_into_tensor,
    reduce_to_significance)

from config import (
    TIMESLIDE_STEP,
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    DEVICE)


def main(args):
    data = np.load(args.data_path)
    device = DEVICE
    data = torch.from_numpy(data).to(device)

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(TIMESLIDE_TOTAL_DURATION // sample_length)
    print("Number of timeslides:", n_timeslides)
    for timeslide_num in range(1, n_timeslides+1):
        indicies_to_slide = int(timeslide_num*TIMESLIDE_STEP*SAMPLE_RATE)
        timeslide = torch.empty(data.shape, device = device)

        # hanford unchanged
        timeslide[0, :] = data[0, :]
        # livingston slid
        timeslide[1, :indicies_to_slide] = data[1, -indicies_to_slide:]
        timeslide[1, indicies_to_slide:] = data[1, :-indicies_to_slide]

        # do the evaluation
        segments = split_into_segments_torch(timeslide[None, :, :])[0]
        segments_normalized = std_normalizer_torch(segments)

        quak_predictions_dict = quak_eval(segments_normalized, 
                                          [f"{args.model_folder_path}/{elem}" for elem in  os.listdir(args.model_folder_path)])
        quak_predictions = stack_dict_into_tensor(quak_predictions_dict)
        pearson_values, (edge_start, edge_end) = pearson_computation(timeslide[None, :, :])
        print("pearson values", pearson_values)
        print("shapes 49", quak_predictions.shape, pearson_values.shape)
        quak_predictions = quak_predictions[edge_start:edge_end]

        final_values = torch.cat([quak_predictions, pearson_values], dim=-1)

        final_values = reduce_to_significance(final_values)

        if args.metric_coefs_path is not None:
            # compute the dot product and save that instead
            metric_vals = np.load(args.metric_coefs_path)
            metric_vals = torch.from_numpy(metric_vals).float().to(device)
            final_values = torch.matmul(final_values, metric_vals)

        final_values = final_values.detach().cpu().numpy()
        assert 0
        # save as a numpy file, with the index of timeslide_num
        np.save(f"{args.save_path}/timeslide_evals_{timeslide_num}.npy")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', help='Directory containing the timeslides',
        type=str)
    
    parser.add_argument('save_path', help = "Folder to which save the timeslides",
        type=str)
        
    parser.add_argument('model_folder_path', help = "Path to the folder containing the models",
        type=str)

    # Additional arguments
    parser.add_argument('--metric_coefs_path', help="Pass in path to metric coefficients to compute dot product",
                        type = str, default=None)

    args = parser.parse_args()
    main(args)

