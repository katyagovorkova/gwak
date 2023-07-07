import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    GPU_NAME,
    SVM_LR,
    N_SVM_EPOCHS
    )
DEVICE = torch.device(GPU_NAME)


class LinearModel(nn.Module):
    def __init__(self, n_dims):
        super(LinearModel, self).__init__()
        #self.layer = nn.Linear(n_dims, 9)
        #self.layer2 = nn.Linear(9, 1)
        self.layer = nn.Linear(n_dims, 1)
        
    def forward(self, x):
        x = (self.layer(x))
        #return self.layer2(x)
        #return self.layer()
        return x
       


def optimize_hyperplane(signals, backgrounds):
    # saved as a batch, which needs to be flattned out
    backgrounds = np.reshape(backgrounds, (backgrounds.shape[0]*backgrounds.shape[1], backgrounds.shape[2]))

    sigs = torch.from_numpy(signals).float().to(DEVICE)
    bkgs = torch.from_numpy(backgrounds).float().to(DEVICE)
    network = LinearModel(n_dims = sigs.shape[2]).to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=SVM_LR)

    for epoch in range(N_SVM_EPOCHS):
        optimizer.zero_grad()
        background_MV = network(bkgs)
        signal_MV = network(sigs)
        signal_MV = torch.min(signal_MV, dim=1)[0] #second index are the indicies
        zero = torch.tensor(0).to(DEVICE)
        background_loss = torch.maximum(
                            zero,
                            1-background_MV).mean()
        signal_loss = torch.maximum(
                            zero,
                            1+signal_MV).mean()
        print(network.layer.weight.data.cpu().numpy()[0])
        loss = background_loss + signal_loss
        #print(loss.item())
        loss.backward()
        optimizer.step()

    #torch.save(network.state_dict(), "./fm_model.pt")
    #return 0
    return network.layer.weight.data.cpu().numpy()[0]

def engineered_features(data):
    #print(data[0, :10, :])
    newdata = np.zeros(data.shape)

    for i in range(4):
        a, b = data[:, :, 2*i], data[:, :, 2*i+1]
        newdata[:, :, 2*i] = (a+b)/2
        newdata[:, :, 2*i+1] = abs(a-b)# / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    #print(newdata[0, :10, :])
    #assert 0
    return newdata

def main(args):
    '''
    Fit a hyperplane to distinguish background, signal classes

    signals: shape (N_samples, time_axis, 5)
    backgrounds: shape (time_axis, 5)
    '''
    #np.save(args.save_file, np.array([0, -1, 0, 0, 0]))
    #return None

    signal_evals = []
    if type(args.signal_path) == str:
        args.signal_path = [args.signal_path]
    for file_name in args.signal_path:
        signal_evals.append(np.load(f'{file_name}'))
    signal_evals = np.concatenate(signal_evals, axis=0)

    timeslide_evals = []
    timeslide_path = args.timeslide_path
    if type(args.timeslide_path) == str:
        timeslide_path = [args.timeslide_path]
    for file_name in timeslide_path:
        timeslide_evals.append(np.load(f'{file_name}'))

    norm_factors = []
    norm_factors_path = args.norm_factor_path
    if type(args.norm_factor_path) == str:
        norm_factors_path = [args.norm_factor_path]
    for file_name in norm_factors_path:
        norm_factors.append(np.load(f'{file_name}'))

    norm_factors = np.array(norm_factors)
    means = np.mean(norm_factors[:, 0, 0, :], axis=0)
    stds = np.mean(norm_factors[:, 1, 0, :], axis=0)

    np.save(args.norm_factor_save_file, np.stack([means, stds], axis=0))
    timeslide_evals = np.concatenate(timeslide_evals, axis=0)
    signal_evals = (signal_evals-means)/stds
    timeslide_evals = (timeslide_evals-means)/stds

    signal_evals = signal_evals[:, 1300:1550, :]
    print(signal_evals.shape)
    print(timeslide_evals.shape)

    #signal_evals = engineered_features(signal_evals)
    #print(signal_evals.shape)
    #timeslide_evals = engineered_features(timeslide_evals)
    #print(timeslide_evals.shape)
    #print("DONEODONEOENE")
    #assert 0
    

    optimal_coeffs = optimize_hyperplane(signal_evals, timeslide_evals)
    #return None
    np.save(args.save_file, optimal_coeffs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('save_file', type=str,
        help='Where to save the best final metric parameters')
    parser.add_argument('norm_factor_save_file', type=str,
        help='Where to save the normalization factors')
    parser.add_argument('--timeslide-path', type=str,
         nargs = '+', help='list[str] pointing to timeslide files ')
    parser.add_argument('--signal-path', type=str,
        nargs= '+', help='list[str] pointing to signal files')
    parser.add_argument('--norm-factor-path', type=str,
        nargs= '+', help='list[str] pointing to norm factors')
    
    args = parser.parse_args()
    main(args)
