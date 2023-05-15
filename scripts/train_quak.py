import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import losses, callbacks, regularizers
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector


def main(args):
    # read the input data
    data = np.load(args.data)
    input_shape = data.shape[1:]
    print('Loading new model')
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    L1 = LSTM(64*args.factor, activation='tanh', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(args.bottleneck*args.factor, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(input_shape[0])(L2)
    L4 = LSTM(args.bottleneck*args.factor, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64*args.factor, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(input_shape[1]))(L5)
    AE = Model(inputs=inputs, outputs=output)
    AE.compile(optimizer=args.optimizer, loss=args.loss)

    # define callbacks
    callbacks=[]
        # EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
        # ]

    history = AE.fit(data, data,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_split=0.15,
                    callbacks=callbacks)
    AE.save(f'{args.savedir}/AE.h5', include_optimizer=False)

    loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    np.save(f'{args.savedir}/loss_hist.npy', loss_hist)
    np.save(f'{args.savedir}/val_loss_hist.npy', val_loss_hist)

    plt.figure(figsize=(15, 10))
    plt.plot(loss_hist, label='loss')
    plt.plot(val_loss_hist, label='val loss')
    plt.legend()
    plt.xlabel('Number of epochs', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    plt.title('Loss curve for training network', fontsize=17)
    plt.savefig(f'{args.savedir}/loss.pdf', dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data', help='Input dataset',
        type=str)
    parser.add_argument('savedir', help='Where to save the plots and the trained model',
        type=str)

    # Additional arguments
    parser.add_argument('--optimizer', help='Which optimizer to use',
        type=str, default='adam')
    parser.add_argument('--loss', help='Which loss function to use',
        type=str, default='MAE')
    parser.add_argument('--batch-size', help='What batch size to use',
        type=int, default=100)
    parser.add_argument('--epochs', help='For how many epochs to train?',
        type=int, default=100)
    parser.add_argument('--factor', help='Factor in LSTM architecture',
        type=int, default=2)
    parser.add_argument('--bottleneck', help='Size of the bottleneck of the autoencoder',
        type=int, default=2)
    args = parser.parse_args()
    main(args)