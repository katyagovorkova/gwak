import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau)
from keras import (
    losses,
    callbacks,
    regularizers)
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import (
    Input,
    Dense,
    LSTM,
    TimeDistributed,
    RepeatVector)

from constants import (
    BOTTLENECK,
    FACTOR,
    EPOCHS,
    BATCH_SIZE,
    LOSS,
    OPTIMIZER,
    VALIDATION_SPLIT)


def main(args):
    # read the input data
    data = np.load(args.data)
    input_shape = data.shape[1:]
    print('Loading new model')
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    L1 = LSTM(64*FACTOR, activation='tanh', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(BOTTLENECK*FACTOR, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(input_shape[0])(L2)
    L4 = LSTM(BOTTLENECK*FACTOR, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64*FACTOR, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(input_shape[1]))(L5)
    AE = Model(inputs=inputs, outputs=output)
    AE.compile(optimizer=OPTIMIZER, loss=LOSS)

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
        ]

    history = AE.fit(data, data,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=VALIDATION_SPLIT,
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
    args = parser.parse_args()
    main(args)