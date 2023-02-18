from keras import layers
from keras.models import Sequential, Model
import numpy as np

def dim_check(input_shape):
    assert type(input_shape) == tuple
    assert len(input_shape) == 2
    assert input_shape[0] % 2 == 0 #odd stuff will mess up dimensions

def make_model(input_shape:tuple, bottleneck:int):
    #CNN network for either 2-d time data or the spectrogram data
    dim_check(input_shape)

    encoder = Sequential()
    encoder.add(layers.Conv1D(filters=256, kernel_size=3, input_shape = input_shape , padding='same'))
    encoder.add(layers.MaxPooling1D(pool_size=2))
    encoder.add(layers.Conv1D(filters=16, kernel_size=3,  padding='same'))
    pre_flat_shape = tuple(np.array(encoder.output.shape)[1:])
    pre_flat_size = 1
    for elem in pre_flat_shape:
        pre_flat_size *= elem
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(units=512))
    encoder.add(layers.Dense(units=bottleneck))

    decoder_input_shape = tuple(np.array(encoder.output.shape)[1:])
    
    #print(encoder.summary())

    decoder_input_shape = tuple(np.array(encoder.output.shape)[1:])
    decoder = Sequential()
    decoder.add(layers.Dense(512, input_shape=(bottleneck,)))
    decoder.add(layers.Dense(pre_flat_size))
    decoder.add(layers.Reshape(pre_flat_shape))
    decoder.add(layers.UpSampling1D(size=2))
    decoder.add(layers.Conv1D(filters=256, kernel_size=3))
    decoder.add(layers.Conv1DTranspose(filters=input_shape[1], kernel_size=3))
    
    #print(decoder.summary())

    if tuple(np.array(decoder.output.shape)[1:]) != input_shape: #checking output size matches input
        print("should be", input_shape)
        print("instead is", tuple(np.array(decoder.output.shape)[1:]))
        assert False
    
    autoencoder = Model(inputs=encoder.input, 
                    outputs=decoder(encoder.outputs))

    return autoencoder, encoder, decoder