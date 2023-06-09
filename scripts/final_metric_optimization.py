import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import (
    GPU_NAME,
    SVM_LR,
    N_SVM_EPOCHS
)
DEVICE = torch.device(GPU_NAME)

class LinearModel(nn.Module):
    def __init__(self, n_dims):
        super(LinearModel, self).__init__()
        self.layer = nn.Linear(n_dims, 1)
        
    def forward(self, x):
        return self.layer(x)

def optimize_hyperplane(signals, backgrounds):
    device=DEVICE
    sigs = torch.from_numpy(signals).float().to(device)
    bkgs = torch.from_numpy(backgrounds).float().to(device)

    network = LinearModel().to(device)
    optimizer = optim.SGD(network.parameters(), lr=SVM_LR)

    for epoch in range(N_SVM_EPOCHS):
        optimizer.zero_grad()
        background_MV = network(bkgs)
        signal_MV = network(signals)
        signal_MV = torch.min(signal_MV, dim=1)[0] #second index are the indicies
        zero = torch.tensor(0).to(device)
        background_loss = torch.maximum(
                            zero,
                            1-background_MV).mean()
        signal_loss = torch.maximum(
                            zero,
                            1+signal_MV).mean()
        loss = background_loss + signal_loss

        loss.backward()
        optimizer.step()

    return network.layer.weight.data.cpu().numpy()[0]

def main(args):
    '''
    Fit a hyperplane to distinguish background, signal classes

    signals: shape (N_samples, time_axis, 5)
    backgrounds: shape (time_axis, 5)
    '''
    signal_evals = []
    if type(args.signal_path) == "str":
        args.signal_path = [args.signal_path]
    for file_name in args.signal_path:
        signal_evals.append(np.load(f'{file_name}'))
    signal_evals = np.concatenate(signal_evals, axis=0)

    timeslide_evals = []
    if type(args.timeslide_path) == "str":
        args.timeslide_path = np.load(args.timeslide_path)
    for file_name in args.timeslide_path:
        timeslide_evals.append(np.load(f'{file_name}'))
    timeslide_evals = np.concatenate(timeslide_evals, axis=0)

    
    optimal_coeffs = optimize_hyperplane(signal_evals, timeslide_evals)
    np.save(args.save_file, optimal_coeffs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('timeslide_path', type=str,
        nargs='+', help='str or list[str] pointing to timeslide files ')
    parser.add_argument('signal_path_folder', type=str,
        nargs= '+', help='str or list[str] pointing to signal files')
    parser.add_argument('save_file', type=str,
        help='Where to save the best final metric parameters')
    args = parser.parse_args()
    main(args)
