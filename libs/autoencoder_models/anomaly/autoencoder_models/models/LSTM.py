from keras import layers
from keras.models import Sequential, Model

def dim_check(input_shape):
        assert type(input_shape) == tuple
        assert len(input_shape) == 2
        assert input_shape[0] >= 1
        assert input_shape[1] >= 1

def make_model(input_shape:tuple, bottleneck:int):
        #LSTM network specifically for n-d inputs with n >=1
        #but in input_shape, the second dimension should be some int >= 1
        dim_check(input_shape)

        encoder = Sequential()
        encoder.add(layers.LSTM(units=32, return_sequences=True, input_shape = input_shape ))
        encoder.add(layers.LSTM(bottleneck))

        decoder = Sequential()
        decoder.add(layers.RepeatVector(input_shape[0], input_dim=bottleneck))
        decoder.add(layers.LSTM(bottleneck, return_sequences=True))
        decoder.add(layers.LSTM(32, return_sequences=True))
        decoder.add(layers.TimeDistributed(layers.Dense(1)))

        autoencoder = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

        return autoencoder, encoder, decoder	
