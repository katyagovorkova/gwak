import os
import numpy as np
import argparse

from keras.models import load_model


def mae(a, b):
    '''
    compute MAE across a, b
    using first dimension as representing each sample
    '''
    norm_factor = a[0].size
    assert a.shape == b.shape
    diff = np.abs(a - b)
    N = len(diff)

    # sum across all axes except the first one
    return np.sum(diff.reshape(N, -1), axis=1) / norm_factor


def main(args):

    data = np.load(args.test_data)

    model = load_model(args.model_path)
    preds = model.predict(data)

    loss = mae(data, preds)
    np.save(args.save_file, loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('test_data', help='Required path to the test data file',
                        type=str)
    parser.add_argument('model_path', help='Required path to trained model',
                        type=str)
    parser.add_argument('save_file', help='Required path to save the file to',
                        type=str)
    args = parser.parse_args()
    main(args)
