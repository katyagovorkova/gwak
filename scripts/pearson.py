import argparse
import numpy as np
import scipy
import torch

from config import (
    MAX_SHIFT,
    SEG_NUM_TIMESTEPS,
    SEG_STEP,
    SHIFT_STEP,
    SAMPLE_RATE,
    GPU_NAME
)
DEVICE = torch.device(GPU_NAME)

def pearson_computation(data,
                        max_shift=MAX_SHIFT,
                        seg_len=SEG_NUM_TIMESTEPS, 
                        seg_step=SEG_STEP,
                        shift_step=SHIFT_STEP):
    device = DEVICE
    max_shift = int(max_shift*SAMPLE_RATE)
    offset_families = np.arange(max_shift, max_shift+seg_len, seg_step)
    
    feature_length_full = data.shape[-1]
    feature_length = (data.shape[-1]//100)*100
    n_manual = (feature_length_full - feature_length) // seg_step
    n_batches = data.shape[0]
    data[:, 1, :] = -1 * data[:, 1, :] #inverting livingston
    family_count = len(offset_families)
    final_length = 0
    for family_index in range(family_count):
        end = feature_length-seg_len+offset_families[family_index]
        if end > feature_length - max_shift:
            # correction: reduce by 1
            final_length -= 1
        final_length += (feature_length - seg_len)//seg_len
    family_fill_max = final_length
    final_length += n_manual
    all_corrs = torch.zeros((n_batches, final_length), device=DEVICE)
    for family_index in range(family_count):
        end = feature_length-seg_len+offset_families[family_index]
        if end > feature_length - max_shift:
            end -= seg_len
        hanford = data[:, 0, offset_families[family_index]:end].reshape(n_batches, -1, 100)
        hanford = hanford - hanford.mean(dim=2)[:, :, None]
        best_corrs = -1 * torch.ones((hanford.shape[0], hanford.shape[1]), device=device)
        for shift_amount in np.arange(-max_shift, max_shift+shift_step, shift_step):

            livingston = data[:, 1, offset_families[family_index]+shift_amount:end+shift_amount].reshape(n_batches, -1, 100)
            livingston = livingston - livingston.mean(dim=2)[:, :, None]

            corrs = torch.sum(hanford*livingston, axis=2) / torch.sqrt( torch.sum(hanford*hanford, axis=2) * torch.sum(livingston*livingston, axis=2) )  
            best_corrs = torch.maximum(best_corrs, corrs)
 
        all_corrs[:, family_index:family_fill_max:family_count] = best_corrs
    
    # fill in pieces left over at end
    for k, center in enumerate(np.arange(feature_length - max_shift - seg_len//2+seg_step, \
                                feature_length_full - max_shift - seg_len//2 + seg_step, \
                                seg_step)):
        hanford = data[:, 0, center - seg_len//2:center + seg_len//2]
        hanford = hanford - hanford.mean(dim=1)[:, None]
        best_corr = -1 * torch.ones((n_batches), device=device)
        for shift_amount in np.arange(-max_shift, max_shift+shift_step, shift_step):
            livingston = data[:, 1, center - seg_len//2+shift_amount:center + seg_len//2 + shift_amount]
            livingston = livingston - livingston.mean(dim=1)[:, None]
            corr = torch.sum(hanford*livingston, axis=1) / torch.sqrt( torch.sum(hanford*hanford, axis=1) * torch.sum(livingston*livingston, axis=1) )
            best_corr = torch.maximum(best_corr, corr)
        all_corrs[:, -(k+1)] = best_corr

    edge_start, edge_end = max_shift // seg_step, -(max_shift // seg_step) + 1
    return all_corrs, (edge_start, edge_end)

def main_gpu(args):
    '''
    INPUTS: 

        data: np.ndarray of shape (N_samples, feature_length, 2)
                N_samples: number of strain segments over which to compute iterated pearson correlation
                feature_length: time axis
                2: corresponds to the number of detectors
        max_shift: int
                    maximum time-like shift of the data corresponding to travel time between
                    Hanford and Livingston detectors
        seg_len: int
                    Segment length over which to compute the pearson correlation
        seg_step: int
                    Stepping size used to compute centers at which the iterated
                    pearson correlation will be computed
        shift_step: int
                    Step size used for iterate over (-maxshift, maxshift)

    OUTPUTS:

        correlations: np.ndarray of shape (N_samples, num_centeres)
                array of the iterated correlations corresponding to seg_len and stepsize

        edge_cut : slice object
                slice object to remove samples on the edge that couldn't have pearson corr computed
    '''
    data = np.load(args.data)
    max_shift, seg_len, seg_step, shift_step = args.max_shift, args.seg_len, args.shift_step
    correlations, (edge_start, edge_end) = pearson_computation(data, max_shift, seg_len, seg_step, shift_step)

    # Need to do something with the edge cut! - have another savedir?
    np.save(args.save_file, correlations.cpu().numpy())

    print("WARNING: Implement saving the edges for slicing!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str,
        help='''np.ndarray of shape (N_samples, feature_length, 2)
                N_samples: number of strain segments over which to compute iterated pearson correlation
                feature_length: time axis
                2: corresponds to the number of detectors''')
    parser.add_argument('save_file', type=str,
        help='Where to save the computed correlations')
    parser.add_argument('--max-shift', type=int, default=int(MAX_SHIFT*SAMPLE_RATE)//5,
        help='Maximum time-like shift of the data corresponding to travel time between Hanford and Livingston detectors')
    parser.add_argument('--seg-len', type=int, default=100,
        help='Segment length over which to compute the pearson correlation')
    parser.add_argument('--seg-step', type=int, default=5,
        help='Stepping size used to compute centers at which the iterated pearson correlation will be computed')
    parser.add_argument('--shift-step', type=int, default=2,
        help='Step size used for iterate over (-args.max_shift, args.max_shift)')
    args = parser.parse_args()
    main_gpu(args)
