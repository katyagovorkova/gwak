import os

def main(model_choice:str, input_shape:tuple, bottleneck:int):
	# Take request for which model is desired
	# create the model
	# and return it
	#again, not the best way to do this, but whatever

	if model_choice == "LSTM":
		from anomaly.autoencoder_models.models.LSTM import make_model
	elif model_choice == "CNN":
		from anomaly.autoencoder_models.models.CNN import make_model
	elif model_choice == "dense":
		from anomaly.autoencoder_models.models.dense import make_model
	elif model_choice == "CNN3D":
		from anomaly.autoencoder_models.models.CNN3D import make_model
	elif model_choice == "eric": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM as make_model
	elif model_choice == "eric_conv_paper": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_Conv_paper as make_model
	elif model_choice == "eric_conv_LSTM": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_ConvLSTM as make_model
	elif model_choice == "eric_conv_dnn": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_ConvDNN as make_model
	elif model_choice == "eric_lstm_big": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_big as make_model
	elif model_choice == "eric_lstm_big_CNN": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_big_CNN as make_model
	elif model_choice == "eric_lstm_big_2channel": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_big_2channel as make_model
	elif model_choice == "eric_lstm_big_big_2channel": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_big_big_2channel as make_model

	elif model_choice == "eric_lstm_big_2channel_CNN": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_big_2channel_CNN as make_model

	elif model_choice == "eric_lstm_smaller_2channel": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_smaller_2channel as make_model
	elif model_choice == "eric_lstm_big_2channel_batchnorm": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_big_2channel_batchnorm as make_model
	elif model_choice == "eric_lstm_deep": #going to have to do this more or less manually from now on
		from anomaly.autoencoder_models.eric_models import autoencoder_LSTM_deep as make_model
	elif model_choice == "transformer":
		from anomaly.autoencoder_models.models.transformer import transformer as make_model
	
	
	else:
		assert False # invalid model specified
	
	AE, EN, DE = make_model(input_shape, bottleneck)
	return AE, EN, DE	
