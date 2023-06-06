import os
import numpy as np
import argparse

import torch
# from helper_functions import mae
from models import LSTM_AE
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (NUM_IFOS,
                    SEG_NUM_TIMESTEPS,
                    BOTTLENECK,
                    FACTOR,
                    DEVICE,
                    GPU_NAME)

from helper_functions import mae_torch
def quak_eval(data, model_path):
    # data required to be torch tensor at this point
    device = DEVICE
    model = LSTM_AE(num_ifos=NUM_IFOS,
                    num_timesteps=SEG_NUM_TIMESTEPS,
                    BOTTLENECK=BOTTLENECK,
                    FACTOR=FACTOR).to(device)

    # check if the evaluation has to be done for one model or for several
    loss = dict()

    for dpath in model_path:
        model.load_state_dict(torch.load(dpath, map_location=GPU_NAME))
        print("WARNING: change .strip() to .pt once model properly renamed!")
        loss[os.path.basename(dpath)[:-3]] = \
            mae_torch(data, model(data)).detach()
    return loss

def main(args):
    device = DEVICE

    # load the data
    data = torch.from_numpy(data).float().to(device)
    data = np.load(args.test_data)
    print(f'loaded data shape is {data.shape}')
    loss = quak_eval(data, args.model_path)

    # move to CPU
    for key in loss.keys():
        loss[key] = loss[key].cpu().numpy()


    if args.save_file: np.savez(args.save_file, **loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('test_data', help='Required path to the test data file')
    parser.add_argument('save_file', help='Required path to save the file to',
                        type=str)

    parser.add_argument('--model-path', help='Required path to trained model',
                        nargs='+', type=str)
    args = parser.parse_args()
    main(args)

