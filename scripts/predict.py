import os
import numpy as np
import argparse

from keras.models import load_model

from helper_functions import mae


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
