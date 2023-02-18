import os
import pickle
import numpy as np

'''
load the saved KDE files 
evalulate them on validation data, with the goal of plotting KDE probabilities 
instead of the regular autoencoder recreation loss in the QUAK space
'''

def main(savedir:str,
        datapath:str,
        class_name:str): 
    '''
    savedir is the usual big directory
    datapath should be the folder with /DATA_PREDICTION/TEST/(class).
    '''

    #load the data
    data = np.load(datapath)
    
    #splitting
    split = int(len(data)*0.8)
    data = data[split:]

    #ordering: bbh, bkg, glitches, injected

    KDE_evals = np.zeros(shape=(len(data), 4))
    index_map = {'bbh_KDE': 0,
                "bkg_KDE": 1,
                "glitches_new_KDE":2,
                "injected_KDE":3}
    for i, model in enumerate(os.listdir(f"{savedir}/TRAINED_MODELS/KDES/")):

        with open(f"{savedir}/TRAINED_MODELS/KDES/{model}", 'rb') as f:
            KDE_class = pickle.load(f)
            
            res = KDE_class.predict(data, convert_ln=True) #note: this is important for quak space, won't have to do log scaling, although perhaps that's easier with labels
            #print("FINALFINAL", res.shape)
            KDE_evals[:, index_map[model[:-4]]] = res #cutting off extension tag

    #saveing
    try:
        os.makedirs(f"{savedir}/DATA_PREDICTION/TEST/{class_name}/")
    except FileExistsError:
        None

    np.save(f"{savedir}/DATA_PREDICTION/TEST/{class_name}/KDE_evals.npy", KDE_evals)







