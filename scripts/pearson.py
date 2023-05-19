import argparse
import numpy as np
import scipy


def main(args):

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
    parser.add_argument('--max-shift', type=int, default=int(10e-3*4096)//5,
        help='Maximum time-like shift of the data corresponding to travel time between Hanford and Livingston detectors')
    parser.add_argument('--seg-len', type=int, default=100,
        help='Segment length over which to compute the pearson correlation')
    parser.add_argument('--seg-step', type=int, default=5,
        help='Stepping size used to compute centers at which the iterated pearson correlation will be computed')
    parser.add_argument('--shift-step', type=int, default=2,
        help='Step size used for iterate over (-args.max_shift, args.max_shift)')
    args = parser.parse_args()
    main(args)