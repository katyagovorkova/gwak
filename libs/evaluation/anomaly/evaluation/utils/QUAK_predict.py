from keras.models import load_model
import numpy as np
import os

def mae(a, b):
    '''
    compute MAE across a, b
    using first dimension as representing each sample
    '''
    norm_factor = a[0].size
    assert a.shape == b.shape
    diff = np.abs(a-b)
    N = len(diff)
    
    #sum across all axes except the first one
    return np.sum(diff.reshape(N, -1), axis=1)/norm_factor

def pre_processing_method(data): #for two detectors at once
    print("DATA SHAPE", data.shape)
	#individually normalize each segment
    #data -= np.average(data, axis=1)[:, np.newaxis] #doesn't seem right to do
    std_vals = np.std(data, axis=2)
    print("std vals shape", std_vals.shape)
    data /= std_vals[:,:, np.newaxis]

    
    print("now data shape:", data.shape)
    return data

def main(savedir, datae, preprocess=True):
    '''
    Function to get the QUAK losses on an input set of data
    '''
    print("into quak prediction", )
    print("datae in 33, ", datae.shape)
    datae = datae.swapaxes(2, 3)
    a, b = datae.shape[0], datae.shape[1]

    #load QUAK models
    QUAK_models = dict()
    QUAK_model_path = f"{savedir}/TRAINED_MODELS/QUAK/"
    for QUAK_class in os.listdir(QUAK_model_path):
        #print("QUAK class", QUAK_class)
        QUAK_models[QUAK_class] = load_model(f"{QUAK_model_path}/{QUAK_class}/AE.h5")
    
    all_results = np.empty((datae.shape[0], datae.shape[1], 4))
    #for i in range(len(datae)):
    data = datae.reshape(datae.shape[0]*datae.shape[1], 2, 100)
    #data = np.swapaxes(data, 1, 2)
    if preprocess:
        data = pre_processing_method(data)
    QUAK_evals = dict()
    for QUAK_class in QUAK_models:
        #print("95, data_class, QUAK_class", data_class, QUAK_class)
        #print("data", data)
        
        pred = QUAK_models[QUAK_class].predict(data)
        QUAK_evals[QUAK_class] = mae(data, pred)
        #print("99, mae from autoencoder", QUAK_evals[data_class][QUAK_class])
        #out_len = len(mae(data_dict[data_class], pred))


    #print("QUAK_classes", QUAK_models.keys())
    
    if "BBH" in QUAK_evals.keys():
        index_map = {'BBH': 0,
                    "BKG": 1,
                    "GLITCH":2,
                    "SG":3}
        index_map_inv = {0:'BBH',
            1:"BKG",
            2:"GLITCH",
            3:"SG"}

    else:
        index_map = {'bbh': 0,
                    "bkg": 1,
                    "glitches_new":2,
                    "injected":3}
        index_map_inv = {0:'bbh',
                    1:"bkg",
                    2:"glitches_new",
                    3:"injected"}

    QUAK_stack = np.zeros(shape=(len(QUAK_evals[list(index_map.keys())[0]]), 4))

    for val in index_map:
        QUAK_stack[:, index_map[val]] = QUAK_evals[val]

    return QUAK_stack.reshape(a, b, 4)
    
    #return all_results