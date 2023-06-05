import os
import numpy as np
import argparse

import torch
from helper_functions import mae
from models import LSTM_AE
from config import (NUM_IFOS,
                    SEG_NUM_TIMESTEPS,
                    BOTTLENECK,
                    FACTOR)

def main(args):

    # load the data
    data = np.load(args.test_data)
    device = torch.device("cuda:0")
    data = torch.from_numpy(data).to(device)

    model = LSTM_AE(num_ifos=NUM_IFOS,
                    num_timesteps=SEG_NUM_TIMESTEPS,
                    BOTTLENECK=BOTTLENECK,
                    FACTOR=FACTOR).to(device)

    # check if the evaluation has to be done for one model or for several
    if len(args.model_path) > 1:

        loss_fn = torch.nn.L1Loss()
        losses = dict()

        for dpath in args.test_data:
            model.load_state_dict(torch.load(args.model_path))
            losses[os.path.basename(dpath).strip('.h5')] = \
                (loss_fn(data, model(data), reduction=None)).detach().cpu().numpy()

        np.savez(args.save_file, **losses)

    else:

        model = load_model(args.model_path)
        preds = model.predict(data)

        loss = mae(data, preds)
        np.save(args.save_file, loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('test_data', help='Required path to the test data file')
    parser.add_argument('model_path', help='Required path to trained model',
                        type=str, nargs='+')
    parser.add_argument('save_file', help='Required path to save the file to',
                        type=str)
    args = parser.parse_args()
    main(args)
