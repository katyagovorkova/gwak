import os
import argparse
import numpy as np
import time
import torch
from torchaudio.functional import convolve

from quak_predict import quak_eval
from helper_functions import (
    std_normalizer_torch, 
    split_into_segments_torch,
    stack_dict_into_tensor,
    reduce_to_significance,
    pearson_computation
    )
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SEGMENT_OVERLAP,
    GPU_NAME,
    CLASS_ORDER,
    N_SMOOTHING_KERNEL,
    DATA_EVAL_MAX_BATCH,
    DO_SMOOTHING,
    SEG_NUM_TIMESTEPS
    )
DEVICE = torch.device(GPU_NAME)


def full_evaluation(data, model_folder_path, return_midpoints=False):
    '''
    Passed in data is of shape (N_samples, 2, time_axis)
    '''
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).to(DEVICE)

    assert data.shape[1] == 2


    clipped_time_axis = (data.shape[2] // SEGMENT_OVERLAP) * SEGMENT_OVERLAP
    data = data[:, :, :clipped_time_axis]

    segments = split_into_segments_torch(data)
    slice_midpoints = np.arange(SEG_NUM_TIMESTEPS//2, segments.shape[1] * (SEGMENT_OVERLAP) + SEG_NUM_TIMESTEPS//2, SEGMENT_OVERLAP)

    segments_normalized = std_normalizer_torch(segments)

    # segments_normalized at this point is (N_batches, N_samples, 2, 100) and
    # must be reshaped into (N_batches * N_samples, 2, 100) to work with quak_predictions
    N_batches, N_samples = segments_normalized.shape[0], segments_normalized.shape[1]
    segments_normalized = torch.reshape(segments_normalized, (N_batches * N_samples, 2, SEG_NUM_TIMESTEPS))
    quak_predictions_dict = quak_eval(segments_normalized, model_folder_path)
    quak_predictions = stack_dict_into_tensor(quak_predictions_dict)
    quak_predictions = torch.reshape(quak_predictions, (N_batches, N_samples, len(CLASS_ORDER)))

    pearson_values, (edge_start, edge_end) = pearson_computation(data)
    
    pearson_values = pearson_values[:, :, None]
    quak_predictions = quak_predictions[:, edge_start:edge_end, :]
    slice_midpoints = slice_midpoints[edge_start:edge_end]
    #print(quak_predictions.shape, pearson_values.shape)
    final_values = torch.cat([quak_predictions, pearson_values], dim=-1)

    if DO_SMOOTHING:
        # do it before significance?
        kernel = torch.ones((N_batches, final_values.shape[-1], N_SMOOTHING_KERNEL)).float().to(DEVICE)/N_SMOOTHING_KERNEL

    if return_midpoints:
        return final_values, slice_midpoints
    return final_values


def main(args):
    data = np.load(args.data_path)
    print("loaded data shape:", data.shape)
    if data.shape[0] == 2:
        data = data.swapaxes(0, 1)
    n_batches_total = data.shape[0]

    _, timeaxis_size, feature_size = full_evaluation(data[:2], args.model_paths).cpu().numpy().shape
    result = np.zeros((n_batches_total, timeaxis_size, feature_size))
    n_splits = n_batches_total//DATA_EVAL_MAX_BATCH
    if n_splits * DATA_EVAL_MAX_BATCH != n_batches_total:
        n_splits += 1
    for i in range(n_splits):
        result[DATA_EVAL_MAX_BATCH*i:DATA_EVAL_MAX_BATCH*(i+1)] = full_evaluation(data[DATA_EVAL_MAX_BATCH*i:DATA_EVAL_MAX_BATCH*(i+1)], args.model_paths).cpu().numpy()

    np.save(args.save_path, result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', help='Directory containing the injections to evaluate',
        type=str)
    
    parser.add_argument('save_path', help = "Folder to which save the evaluated injections",
        type=str)
        
    parser.add_argument('model_paths', help = "List of models",
        nargs='+', type=str)

    args = parser.parse_args()
    main(args)