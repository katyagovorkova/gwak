from keras import layers
from keras.models import Sequential, Model


def dim_check(input_shape):
	assert type(input_shape) == tuple
	assert len(input_shape) == 1
	assert input_shape[0] >= 1

def make_model(input_shape:tuple, bottleneck:int):
	#Dense network specifically for 1-d inputs
	dim_check(input_shape)

	encoder = Sequential()
	encoder.add(layers.Dense(units=2048, input_shape = input_shape))
	encoder.add(layers.Dense(units=1024))
	encoder.add(layers.Dense(units=bottleneck))
	
	decoder = Sequential()
	decoder.add(layers.Dense(units=1024, input_shape = (bottleneck,)))
	decoder.add(layers.Dense(units=2048))
	decoder.add(layers.Dense(units=input_shape[0]))

	autoencoder = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

	return autoencoder, encoder, decoder
