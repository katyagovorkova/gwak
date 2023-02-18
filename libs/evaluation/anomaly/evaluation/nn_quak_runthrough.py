from anomaly.evaluation.kde import KDE
from keras.models import load_model
from keras import layers
from keras.losses import CategoricalCrossentropy
from keras import Sequential
import pickle
import numpy as np
import os
import scipy.stats as st

#goal is to runthrough some background files, 
#evaluate QUAK models on them, 
#and save the KDE models for future use
index_map = {'bbh': 0,
                    "bkg": 1,
                    "glitches_new":2,
                    "injected":3}
index_map_inv = {0:'bbh',
            1:"bkg",
            2:"glitches_new",
            3:"injected"}

index_map = {'BBH': 0,
                    "BKG": 1,
                    "GLITCH":2,
                    "SG":3}
index_map_inv = {0:'BBH',
    1:"BKG",
    2:"GLITCH",
    3:"SG"}

def make_model(input_shape):
    model = Sequential([
    layers.Flatten(input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')
    ])

    return model

def main(savedir:str):

    comb_data = dict()
    for data_class_name in os.listdir(f"{savedir}/DATA_PREDICTION/TEST/"):
        #start with assumption that everything form the runthrough path is from the same class, at least by initial design
        #print("ACCESS SHAPE DEBUG")
        #print(np.load(f"{savedir}/DATA_PREDICTION/TEST/bbh/QUAK_evals.npy").shape)
        runthrough_path = f"{savedir}/DATA_PREDICTION/TEST/{data_class_name}/QUAK_evals.npy"
        QUAK_preds = np.load(runthrough_path)
        split = int(len(QUAK_preds)*0.8)
        QUAK_preds = QUAK_preds[:split]
        #print("DEBUG 19", QUAK_preds.shape)
        #print("DEBUG 20", data_class_name, QUAK_preds)
        #assert False

        #if KDE_model:
        #training KDE models
        #kde_model_trained = KDE(QUAK_preds)
        #comb_data.append(QUAK_preds)
        comb_data[data_class_name] = QUAK_preds

    #structuring the data
    data_structured = []
    y_data_structured = []
    if "BBH" in comb_data:
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


    for i in (index_map_inv):
        val = index_map_inv[i]
        data_structured.append(comb_data[val])
        y_data_structured.append(np.ones(shape=(len(comb_data[val])))*i)

    nn_x = np.concatenate(data_structured)
    nn_y = np.concatenate(y_data_structured)

    print("nn_y orig", nn_y)
    N_classes = 4
    nn_y = np.eye(N_classes)[nn_y.astype('int')]

    #train the simple model
    model = make_model(input_shape=(4, 1))
    print("MODEL SUMMARY MODEL SUMMARY")
    print(model.summary())

    model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

    model.fit(nn_x, nn_y, epochs=20)

    '''
    savedir is the usual big directory
    datapath should be the folder with /DATA_PREDICTION/TEST/(class).
    '''
    for data_class in os.listdir(f"{savedir}/DATA_PREDICTION/TEST/"):
        #load the data
        datapath = f"{savedir}/DATA_PREDICTION/TEST/{data_class}/QUAK_evals.npy"
        data = np.load(datapath)
        
        #splitting
        split = int(len(data)*0.8)
        data = data[split:]

        #ordering: bbh, bkg, glitches, injected

        NN_evals = model.predict(data)
        print(NN_evals)

       
        
        np.save(f"{savedir}/DATA_PREDICTION/TEST/{data_class}/NN_evals.npy", NN_evals)

    #save the NN model
    try:
        os.makedirs(f"{savedir}/TRAINED_MODELS/QUAK_NN/")
    except FileExistsError:
        None
    nn_savepath = f"{savedir}/TRAINED_MODELS/QUAK_NN/quak_nn.h5"
    model.save(nn_savepath, include_optimizer=False)

        
    return model
