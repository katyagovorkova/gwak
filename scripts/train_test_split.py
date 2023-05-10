import os
import argparse
import numpy as np

def main(args):

    for file in sorted(os.listdir(args.data_path)):

        # split into training and testing data, process
        data = np.load(f'{args.data_path}/{file}')
        print('right after loading', data.shape)
        p = np.random.permutation(len(data))
        data = data[p]

        print('before pre-processing', data.shape)
        split_index = int(args.test_split * len(data))

        np.save(f'{args.train_dir}/{file}', data[:split_index])
        np.save(f'{args.test_dir}/{file}', data[split_index:])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('train_dir', help='Required output directory for train dataset',
        type=str)
    parser.add_argument('test_dir', help='Required output directory for test dataset',
        type=str)

    # Additional arguments
    parser.add_argument('--test-split', help='Part of the dataset that is going to be used for training',
        type=float, default=0.9)
    parser.add_argument('--data-path', help='Where is the data to do train/test split on',
        type=str)
    args = parser.parse_args()
    main(args)