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
                    FACTOR)

def main(args):

    # load the data
    data = np.load(args.test_data)
    print(f'loaded data shape is {data.shape}')

    # pick a random GPU device to train model on
    N_GPUs = torch.cuda.device_count()
    chosen_device = np.random.randint(0, N_GPUs)
    device = torch.device(f"cuda:{chosen_device}")

    data = torch.from_numpy(data).float().to(device)

    model = LSTM_AE(num_ifos=NUM_IFOS,
                    num_timesteps=SEG_NUM_TIMESTEPS,
                    BOTTLENECK=BOTTLENECK,
                    FACTOR=FACTOR).to(device)

    # check if the evaluation has to be done for one model or for several
    if len(args.model_path) > 1:

        loss_fn = torch.nn.L1Loss(reduce=None)
        loss = dict()

        for dpath in args.model_path:
            model.load_state_dict(torch.load(dpath))
            loss[os.path.basename(dpath).strip('.pt')] = \
                loss_fn(data, model(data)).detach().cpu().numpy()

        if args.save_file: np.savez(args.save_file, **loss)

    else:

        model = load_model(args.model_path)
        preds = model.predict(data)

        loss = mae(data, preds)
        if args.save_file: np.save(args.save_file, loss)

    return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('test_data', help='Required path to the test data file')
    parser.add_argument('save_file', help='Required path to save the file to',
                        type=str)

    parser.add_argument('--model-path', help='Required path to trained model',
                        nargs='+', type=str)
    args = parser.parse_args()
    loss = main(args)

