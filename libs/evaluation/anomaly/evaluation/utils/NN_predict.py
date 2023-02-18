import numpy as np
from keras.models import load_model

def main(savedir, data):
    '''
    Function to evaluate the trained NN on data from the QUAK space

    assuming that data is in the shape (N_samples, sample_axis, 4)
    '''

    nn_quak_model = load_model(f"{savedir}/TRAINED_MODELS/QUAK_NN/quak_nn.h5")
    print("data going into NN QUAK", data.shape)
    data_r = np.reshape(data, (data.shape[0]*data.shape[1], 4))
    print("data after reshape", data_r.shape)
    preds = nn_quak_model.predict(data_r)
    print("after neural network prediction shape", preds.shape)

    return np.reshape(preds, (data.shape[0], data.shape[1], 4))


