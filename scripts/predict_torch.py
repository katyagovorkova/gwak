import os
import numpy as np
import argparse

import torch

from helper_functions import mae
from autoencoder_model import LSTM_AE
from config import (NUM_IFOS, 
                    SEG_NUM_TIMESTEPS,
                    BOTTLENECK,
                    FACTOR)

def main(args):

    data = np.load(args.test_data)
    device = torch.device("cuda:0")
    data = torch.from_numpy(data).to(device)

    model = LSTM_AE(num_ifos=NUM_IFOS, 
                    num_timesteps=SEG_NUM_TIMESTEPS,
                    BOTTLENECK=BOTTLENECK,
                    FACTOR=FACTOR).to(device)
    model.load_state_dict(torch.load(args.model_path))

    loss_fn = torch.nn.L1Loss()
    loss = (loss_fn(data, model(data), reduction=None)).detach().cpu().numpy()
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
