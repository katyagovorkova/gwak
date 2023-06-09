import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from quak_predict import quak_eval
from pearson import pearson_computation
import time

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
    GPU_NAME,
    CLASS_ORDER)
DEVICE = torch.device(GPU_NAME)

def full_evaluation(data, model_folder_path):
    '''
    Passed in data is of shape (N_samples, 2, time_axis)
    '''
    print("Warning: Implement smoothing!")
    device = DEVICE
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).to(device)

    assert data.shape[1] == 2

    clipped_time_axis = (data.shape[2] // 5) * 5
    data = data[:, :, :clipped_time_axis]

    segments = split_into_segments_torch(data)
    segments_normalized = std_normalizer_torch(segments)

    # segments_normalized at this point is (N_batches, N_samples, 2, 100) and
    # must be reshaped into (N_batches * N_samples, 2, 100) to work with quak_predictions
    N_batches, N_samples = segments_normalized.shape[0], segments_normalized.shape[1]
    segments_normalized = torch.reshape(segments_normalized, (N_batches * N_samples, 2, 100))
    quak_predictions_dict = quak_eval(segments_normalized, 
                                        [f"{model_folder_path}/{elem}" for elem in  os.listdir(model_folder_path)])
    quak_predictions = stack_dict_into_tensor(quak_predictions_dict)
    quak_predictions = torch.reshape(quak_predictions, (N_batches, N_samples, len(CLASS_ORDER)))
    pearson_values, (edge_start, edge_end) = pearson_computation(data)
    pearson_values = pearson_values[:, :, None]
    quak_predictions = quak_predictions[:, edge_start:edge_end, :]

    final_values = torch.cat([quak_predictions, pearson_values], dim=-1)
    final_values = reduce_to_significance(final_values)

    return final_values

def main(args):
    data = np.load(args.data_path)
    model_folder_path = args.model_folder_path

    result = full_evaluation(data, model_folder_path).cpu().numpy()

    np.save(args.save_path, result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', help='Directory containing the injections to evaluate',
        type=str)
    
    parser.add_argument('save_path', help = "Folder to which save the evaluated injections",
        type=str)
        
    parser.add_argument('model_folder_path', help = "Path to the folder containing the models",
        type=str)

    args = parser.parse_args()
    main(args)