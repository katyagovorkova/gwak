import os
import numpy as np
import argparse

from constants import TEST_SPLIT


def process(data):
    print('Data shape', data.shape)
    # individually normalize each segment
    # find the bigger axis to average over
    time_axis = 2
    feature_axis = 1
    if data.shape[1] > data.shape[2]:
        time_axis = 1
        feature_axis = 2
    print('time axis, feature axis:', time_axis, feature_axis)

    std_vals = np.std(data, axis=time_axis)
    print('std vals shape', std_vals.shape)
    if time_axis == 2:
        data /= std_vals[:,:, np.newaxis]
    else:
        assert time_axis == 1
        data /= std_vals[:,np.newaxis, :]

    print('now data shape:', data.shape)

    return data # unless using 2 detector streams!


def main(args):

    data = np.load(args.input_file)
    p = np.random.permutation(len(data))
    data = data[p]

    print('before pre-processing', data.shape)
    split_index = int(TEST_SPLIT * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]

    train_pdata = process(train_data)
    test_pdata = process(test_data)

    np.save(args.train_file, train_pdata)
    np.save(args.test_file, test_pdata)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('input_file', help='Required input data file location',
        type=str)
    parser.add_argument('train_file', help='Required output file for training dataset',
        type=str)
    parser.add_argument('test_file', help='Required output file for testing dataset',
        type=str)
    args = parser.parse_args()
    main(args)