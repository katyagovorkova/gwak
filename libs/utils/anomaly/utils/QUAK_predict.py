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

def main(savedir, data, preprocess=True):
    '''
    Function to get the QUAK losses on an input set of data
    '''

    if preprocess:
        data = pre_processing_method(data)

    #load QUAK models
    QUAK_models = dict()
    QUAK_model_path = f"{savedir}/TRAINED_MODELS/QUAK/"
    for QUAK_class in os.listdir(QUAK_model_path):
        #print("QUAK class", QUAK_class)
        QUAK_models[QUAK_class] = load_model(f"{QUAK_model_path}/{QUAK_class}/AE.h5")
    
    QUAK_evals = dict()
    for QUAK_class in QUAK_models:
        #print("95, data_class, QUAK_class", data_class, QUAK_class)
        pred = QUAK_models[QUAK_class].predict(data)
        QUAK_evals[QUAK_class] = mae(data, pred)
        #print("99, mae from autoencoder", QUAK_evals[data_class][QUAK_class])
        out_len = len(mae(data_dict[data_class], pred))


    #print("QUAK_classes", QUAK_models.keys())
    QUAK_stack = np.zeros(shape=(len(QUAK_evals[data_class]['bbh']), 4))
    index_map = {'bbh': 0,
                "BBH" : 0,
                "bkg": 1,
                "BKG" : 1,
                "glitches_new":2,
                "GLITCH" : 2,
                "injected":3,
                "SG":3}
    
    for val in index_map:
        QUAK_stack[:, index_map[val]] = QUAK_evals[data_class][val]

    return QUAK_stack