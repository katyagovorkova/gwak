import argparse
import numpy as np
import scipy
import torch

from config import (
    MAX_SHIFT,
    SEG_NUM_TIMESTEPS,
    SEG_STEP,
    SHIFT_STEP
)

def main_cpu(args):

    data = np.load(args.data_path)

    args.shift_step = 1
    args.max_shift = int(10e-3*4096)//5 # 10 ms at 4096 Hz
    best_pearsons = np.zeros((len(data), 2*args.max_shift//args.shift_step))
    for shift in np.arange(0, args.max_shift, args.shift_step):
        data_H = data[:,0, shift:]
        data_L = data[:,1, :100-shift]
        for i in range(len(data)):
            best_pearsons[i, shift//args.shift_step] = (scipy.stats.pearsonr(data_H[i], -data_L[i])[0])

        # augment the other way
        data_H = data[:,0, :100-shift]
        data_L = data[:,1, shift: ]
        for i in range(len(data)):
            best_pearsons[i, shift//args.shift_step+args.max_shift//args.shift_step] = (scipy.stats.pearsonr(data_H[i], -data_L[i])[0])

    np.save(args.save_file, np.amax(abs(best_pearsons), axis=1))

def pearson_computation(data, 
                        max_shift=MAX_SHIFT, 
                        seg_len=SEG_NUM_TIMESTEPS, 
                        seg_step=SEG_STEP,
                        shift_step=SHIFT_STEP):
    N_samples = data.shape[0]
    feature_length = data.shape[1]

    half_seg_len = seg_len//2
    centres = np.arange(half_seg_len, feature_length-half_seg_len-seg_step, seg_step)
    edge_start, edge_end = max_shift//seg_step, -max_shift//seg_step
    edge_cut = slice(edge_start, edge_end)
    centres = centres[edge_cut]

    device = torch.device("cuda:0")
    data = torch.from_numpy(data).to(device)
    data[:, :, 1] = -1 * data[:, :, 1] #inverting livingston

    N_centres = len(centres)
    correlations = -1 * torch.ones(N_samples, N_centres, device=device) #initializing with minumim possible value
    
    for shift_amount in np.arange(-max_shift, max_shift, shift_step):
        hanford = torch.empty(size=(N_samples, N_centres, 100), device=device)
        livingston = torch.empty(size=(N_samples, N_centres, 100), device=device)    
        for i, center in enumerate(centres):
            
            left, right = center - half_seg_len, center + half_seg_len
            hanford[:, i, :] = data[:, left:right, 0]
            livingston[:, i, :] = data[:, left+shift_amount:right+shift_amount, 1] 
        

        #reshape into N_samples, 100, 2
        hanford = torch.transpose(torch.reshape(hanford, (N_samples*N_centres, 100)), 0, 1)
        livingston = torch.transpose(torch.reshape(livingston, (N_samples*N_centres, 100)), 0, 1)
        
        hanford = hanford - torch.mean(hanford, axis=0)
        livingston = livingston - torch.mean(livingston, axis=0)
        comp_corrs = torch.sum(hanford*livingston, axis=0) / torch.sqrt( torch.sum(hanford*hanford, axis=0) * torch.sum(livingston * livingston, axis=0) )
        comp_corrs = torch.reshape(comp_corrs, (N_samples, N_centres))
        correlations = torch.maximum(correlations, comp_corrs)

    return correlations, (edge_start, edge_end)

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
    parser.add_argument('--max-shift', type=int, default=int(10e-3*2048)//5,
        help='Maximum time-like shift of the data corresponding to travel time between Hanford and Livingston detectors')
    parser.add_argument('--seg-len', type=int, default=100,
        help='Segment length over which to compute the pearson correlation')
    parser.add_argument('--seg-step', type=int, default=5,
        help='Stepping size used to compute centers at which the iterated pearson correlation will be computed')
    parser.add_argument('--shift-step', type=int, default=2,
        help='Step size used for iterate over (-args.max_shift, args.max_shift)')
    args = parser.parse_args()
    main_gpu(args)