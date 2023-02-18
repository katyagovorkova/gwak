from keras import layers
from keras.models import Sequential, Model
import numpy as np

def dim_check(input_shape):
    assert type(input_shape) == tuple
    assert len(input_shape) == 3
    #assert input_shape[0] % 2 == 0 #odd stuff will mess up dimensions

def make_model(input_shape:tuple, bottleneck:int):
    #CNN network for either 2-d time data or the spectrogram data
    dim_check(input_shape)
    depth = input_shape[-1]

    encoder = Sequential()
    encoder.add(layers.Conv2D(filters=256, kernel_size=(3, 3), input_shape = input_shape , padding='same'))
    encoder.add(layers.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(layers.Conv2D(filters=128, kernel_size=(3, 3),  padding='same'))
    encoder.add(layers.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(layers.Conv2D(filters=5, kernel_size=(2, 2), padding='same'))
    pre_flat_shape = tuple(np.array(encoder.output.shape)[1:])
    pre_flat_size = 1
    for elem in pre_flat_shape:
        pre_flat_size *= elem
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(units=2048))
    encoder.add(layers.Dense(units=bottleneck))

    decoder_input_shape = tuple(np.array(encoder.output.shape)[1:])
    decoder = Sequential()
    decoder.add(layers.Dense(2048, input_shape=(bottleneck,)))
    decoder.add(layers.Dense(pre_flat_size))
    decoder.add(layers.Reshape(pre_flat_shape))
    
    decoder.add(layers.UpSampling2D(size=2))
    decoder.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    decoder.add(layers.Conv2DTranspose(filters=input_shape[0], kernel_size=(3, 3), padding="same"))
    decoder.add(layers.UpSampling2D(size=2))
    decoder.add(layers.Conv2D(filters=input_shape[-1], kernel_size=(3, 3), padding="same"))
    
    assert tuple(np.array(decoder.output.shape)[1:]) == input_shape #checking output size matches input
    
    autoencoder = Model(inputs=encoder.input, 
                    outputs=decoder(encoder.outputs))

    return autoencoder, encoder, decoder