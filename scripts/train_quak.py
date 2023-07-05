import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from scipy.linalg import dft

from models import (
    LSTM_AE,
    LSTM_AE_SPLIT,
    FAT)

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    BOTTLENECK,
    EPOCHS,
    BATCH_SIZE,
    LOSS,
    OPTIMIZER,
    VALIDATION_SPLIT,
    TRAINING_VERBOSE,
    NUM_IFOS,
    SEG_NUM_TIMESTEPS,
    GPU_NAME,
    LIMIT_TRAINING_DATA,
    CURRICULUM_SNRS)
DEVICE = torch.device(GPU_NAME)

def main(args):

    data_name = (args.data).split("/")[-1][:-4]
    #SEG_NUM_TIMESTEPS = 100
    if 1:
        if data_name in ['bbh', 'sg']:
            # curriculus learning scheme
            noisy_data = np.load(args.data)['noisy']
            clean_data = np.load(args.data)['clean']
            n_currics = len(noisy_data)
            # normalization scheme
            stds = np.std(noisy_data, axis=-1)[:, :, :, np.newaxis]
            noisy_data = noisy_data / stds
            clean_data = clean_data / stds

            #shuffle
            p = np.random.permutation(noisy_data.shape[1])
            noisy_data = noisy_data[:, p, :, :]
            clean_data = clean_data[:, p, :, :]

        else:
            assert data_name in ['background', 'glitches']
            noisy_data = np.load(args.data)['data'] # n_samples, ifo, timesteps

            n_currics = 1
            noisy_data = noisy_data[np.newaxis, :, :, :]
            stds = np.std(noisy_data, axis=-1)[:, :, :, np.newaxis]
            noisy_data = noisy_data / stds
            p = np.random.permutation(noisy_data.shape[1])
            noisy_data = noisy_data[:, p, :, :]
            clean_data = noisy_data

    if data_name in ['bbh', 'sg']:
        AE = LSTM_AE_SPLIT(num_ifos=NUM_IFOS,
                num_timesteps=SEG_NUM_TIMESTEPS,
                BOTTLENECK=BOTTLENECK[data_name]).to(DEVICE)

    elif data_name in ['background', 'glitches']:
        AE = FAT(num_ifos=NUM_IFOS,
                num_timesteps=SEG_NUM_TIMESTEPS,
                BOTTLENECK=BOTTLENECK[data_name]).to(DEVICE)

    optimizer = optim.Adam(AE.parameters())

    if LOSS == 'MAE':
        loss_fn = nn.L1Loss()
    else:
        # add in support for more losses?
        raise Exception("Unknown loss function")

    curriculum_master = []
    for c in range(n_currics):
        #noisy_data, clean_data
        data_x, data_y = noisy_data[c], clean_data[c]
        # create the dataset and validation set
        validation_split_index = int((1-VALIDATION_SPLIT) * len(data_x))

        train_data_x = data_x[:validation_split_index]
        train_data_x = torch.from_numpy(train_data_x).float().to(DEVICE)
        train_data_y = data_y[:validation_split_index]
        train_data_y = torch.from_numpy(train_data_y).float().to(DEVICE)

        validation_data_x = data_x[validation_split_index:]
        validation_data_x = torch.from_numpy(validation_data_x).float().to(DEVICE)
        validation_data_y = data_y[validation_split_index:]
        validation_data_y = torch.from_numpy(validation_data_y).float().to(DEVICE)


        dataloader = []
        N_batches = len(train_data_x) // BATCH_SIZE
        for i in range(N_batches-1):
            start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE
            dataloader.append([ train_data_x[start:end] ,
                               train_data_y[start:end] ])

        curriculum_master.append([
            dataloader,
            validation_data_x,
            validation_data_y
        ])
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'curric_step': []
    }
    # training loop
    epoch_count = 0
    for curric_num in range(n_currics):
        dataloader, validation_data_x, validation_data_y = curriculum_master[curric_num]
        for epoch_num in range(EPOCHS//n_currics):
            epoch_count += 1
            ts = time.time()
            epoch_train_loss = 0
            for batch in dataloader:
                train_data_x, train_data_y = batch
                optimizer.zero_grad()
                #print("shape 83", batch.shape)
                output = AE(train_data_x)
                loss = loss_fn(train_data_y, output)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_train_loss /= N_batches
            training_history['train_loss'].append(epoch_train_loss)
            validation_loss = loss_fn(validation_data_y,
                                    AE(validation_data_x))
            training_history['val_loss'].append(validation_loss.item())
            training_history['curric_step'].append(curric_num)
            #scheduler.step(validation_loss)

            if TRAINING_VERBOSE:
                elapsed_time = time.time() - ts

                print(f"data: {data_name}, epoch: {epoch_count}, train loss: {epoch_train_loss :.4f}, val loss: {validation_loss :.4f}, time: {elapsed_time :.4f}")
        # reset the optimizer after each curriculum iteration
        optimizer = optim.Adam(AE.parameters())

    # save the model
    torch.save(AE.state_dict(), f'{args.save_file}')

    # save training history
    np.save(f'{args.savedir}/loss_hist.npy',
            np.array(training_history['train_loss']))
    np.save(f'{args.savedir}/val_loss_hist.npy',
            np.array(training_history['val_loss']))

    # plot training history
    centers = CURRICULUM_SNRS

    fig, ax = plt.subplots(1, figsize=(8, 5))
    epochs = np.linspace(1, epoch_count, epoch_count)

    ax.plot(epochs, np.array(training_history['train_loss']), label = "Training loss")
    ax.plot(epochs, np.array(training_history['val_loss']), label = "Validation loss")


    ax.legend()
    ax.set_xlabel("Epochs", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.grid()

    #ax_1.plot(np.array(training_history['curric_step']), label = 'curric_step', c='red')
    if n_currics != 1:
        ax_1 = ax.twinx()
        for i in range(n_currics):
            low, high = centers[i]-centers[i]//4, centers[i] + centers[i]//2
            ax_1.fill_between(epochs[i*EPOCHS//n_currics:(i+1)*EPOCHS//n_currics+1], low, high, color="blue", alpha=0.2)
        #ax_1.legend()
        #ax_1.set_yscale("log")
        ax_1.set_ylabel("SNR range", fontsize=15)

    plt.savefig(f'{args.savedir}/loss.pdf', dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data', help='Input dataset',
        type=str)
    parser.add_argument('save_file', help='Where to save the trained model',
        type=str)
    parser.add_argument('savedir', help='Where to save the plots',
        type=str)

    parser.add_argument('--model', help='Required path to trained model',
                        type=str, choices=['lstm', 'dense', 'transformer'])

    args = parser.parse_args()
    main(args)