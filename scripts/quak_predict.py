import os
import numpy as np
import argparse
from torchsummary import summary
import torch
# from helper_functions import mae
from models import LSTM_AE, LSTM_AE_ERIC, DUMMY_CNN_AE, FAT
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (NUM_IFOS,
                    SEG_NUM_TIMESTEPS,
                    BOTTLENECK,
                    FACTOR,
                    GPU_NAME,
                    RECREATION_LIMIT)
DEVICE = torch.device(GPU_NAME)


from helper_functions import mae_torch
def quak_eval(data, model_path, reduce_loss=True):
    # data required to be torch tensor at this point
    model = FAT(num_ifos=NUM_IFOS,
                    num_timesteps=SEG_NUM_TIMESTEPS,
                    BOTTLENECK=BOTTLENECK,
                    FACTOR=FACTOR).to(DEVICE)
    

    # check if the evaluation has to be done for one model or for several
    loss = dict()
    if not reduce_loss:
        loss['original'] = dict()
        loss['recreated'] = dict()
        loss['loss'] = dict()

    for dpath in model_path:
        model.load_state_dict(torch.load(dpath, map_location=GPU_NAME))
        print("WARNING: change .strip() to .pt once model properly renamed!")
        if reduce_loss:
            loss[os.path.basename(dpath)[:-3]] = \
                mae_torch(data, model(data).detach())
        else:
            loss['loss'][os.path.basename(dpath)[:-3]] = \
                mae_torch(data, model(data).detach()).cpu().numpy()
            loss['original'][os.path.basename(dpath)[:-3]] = data[:RECREATION_LIMIT].cpu().numpy()
            loss['recreated'][os.path.basename(dpath)[:-3]] = model(data[:RECREATION_LIMIT]).detach().cpu().numpy()
    return loss

def main(args):
    
    # load the data
    data = np.load(args.test_data)
    data = torch.from_numpy(data).float().to(DEVICE)
    print(f'loaded data shape is {data.shape}')
    loss = quak_eval(data, args.model_path, args.reduce_loss)

    if args.reduce_loss:
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
    parser.add_argument('reduce_loss', help='Whether to reduce to loss values or return recreation',
                        type=str, default="False")

    parser.add_argument('--model-path', help='Required path to trained model',
                        nargs='+', type=str)
    args = parser.parse_args()
    args.reduce_loss = args.reduce_loss == "True"
    main(args)

