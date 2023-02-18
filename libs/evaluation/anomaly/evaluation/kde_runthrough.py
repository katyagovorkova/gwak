from anomaly.evaluation.kde import KDE
from keras.models import load_model
import pickle
import numpy as np
import os
import scipy.stats as st

#goal is to runthrough some background files, 
#evaluate QUAK models on them, 
#and save the KDE models for future use


def main(savedir:str):
    kde_models = dict()
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
        kde_model_trained = st.gaussian_kde(QUAK_preds.T)
        kde_models[data_class_name] = kde_model_trained
        
        #make file path to be saved at
        saving = False
        if saving:
            try:
                os.makedirs(f"{savedir}/TRAINED_MODELS/KDES/")
            except FileExistsError:
                None

            with open(f"{savedir}/TRAINED_MODELS/KDES/{data_class_name}_KDE.pkl", 'wb') as f:
                pickle.dump(kde_model_trained, f, pickle.HIGHEST_PROTOCOL)


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

        KDE_evals = np.zeros(shape=(len(data), 4))
        index_map = {'bbh': 0,
                    "bkg": 1,
                    "glitches_new":2,
                    "injected":3}
        print("DATA CLASS:", data_class)
        for model_name in kde_models:
            model = kde_models[model_name]

            #with open(f"{savedir}/TRAINED_MODELS/KDES/{model}", 'rb') as f:
            #    KDE_class = pickle.load(f)
                
            #res = model.predict(data, convert_ln=True) #note: this is important for quak space, won't have to do log scaling, although perhaps that's easier with labels
            res = model(data.T)
            print("74 RES", res)

            #do some normalization
            KDE_evals[:, index_map[model_name]] = res
        print("     MODEL NAME", model_name)
        print("     BEFORE", KDE_evals[5])
        #KDE_evals = np.divide(KDE_evals, np.squeeze(np.sum(KDE_evals, axis=1), axis=1))
        for i, row in enumerate(KDE_evals):
            KDE_evals[i, :] /= np.sum(row)

        print("     AFTER", KDE_evals[5])
        #saveing
        try:
            os.makedirs(f"{savedir}/DATA_PREDICTION/TEST/{data_class}/")
        except FileExistsError:
            None
        
        np.save(f"{savedir}/DATA_PREDICTION/TEST/{data_class}/KDE_evals.npy", KDE_evals)
        
    return kde_models
