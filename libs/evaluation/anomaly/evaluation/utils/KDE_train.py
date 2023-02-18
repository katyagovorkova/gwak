from scipy.stats import gaussian_kde
import os
import numpy as np

class KDE_supermodel:
    def __init__(self, kde_trained_map, index_map, index_map_inv):
        self.kde_trained_map = kde_trained_map
        self.index_map = index_map
        self.index_map_inv = index_map_inv

    def eval(self, data):
        output = np.empty((data.shape[0], data.shape[1], len(self.index_map_inv)))
        print("on eval, output shape", output.shape)
        for i in range(len(self.index_map_inv)):
            kde_class = self.index_map_inv[i]
            print("15 shape", data.shape)
            orig_shape = data.shape
            #priunt()
            data_ = np.reshape(data, (orig_shape[0]*orig_shape[1], 4))
            preds = self.kde_trained_map[kde_class].evaluate(data_.T)
            preds = np.reshape(preds, (orig_shape[0], orig_shape[1]))
            output[:, :, i] = preds
        return output

def main(savedir, manual_bkg_data=None):
    '''
    Simple function that takes the QUAK evaluation on training data,
    trains KDE models, and returns them in a sorted way.
    R4 -> R4 (quak to probabilities)
    '''

    class_names = os.listdir(f"{savedir}/DATA_PREDICTION/TEST/")

    if "BBH" in class_names:
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
    
    kde_trained_map = dict()
    for CN in class_names:
        if CN in ['BKG', "bkg"] and manual_bkg_data is not None:
            #manually load background data, specifically from timeslides
            data = manual_bkg_data
        else:
            data = np.load(f"{savedir}/DATA_PREDICTION/TEST/{CN}/QUAK_evals.npy")
        kde_trained = gaussian_kde(data.T)
        kde_trained_map[CN] = kde_trained

    final_model = KDE_supermodel(kde_trained_map, index_map, index_map_inv)

    return final_model


def main_bkg(savedir):
    '''
    Similar implementation to the above, but it only creates 
    one KDE model based on the background data - (BKG + GLITCH)
    '''
    class_names = os.listdir(f"{savedir}/DATA_PREDICTION/TEST/")

    if "BBH" in class_names:
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
    
    kde_trained_map = dict()
    class_names = [index_map_inv[1], index_map_inv[2]]
    datae = []
    for CN in class_names:
        data = np.load(f"{savedir}/DATA_PREDICTION/TEST/{CN}/QUAK_evals.npy")
        datae.append(data)
        
    data = np.vstack(datae)

    print("100 DEBUG", data.shape)
    kde_trained = gaussian_kde(data.T)
    kde_trained_map["FULL_BKG"] = kde_trained

    new_index_map = {"FULL_BKG":0}
    new_index_map_inv = {0:"FULL_BKG"}

    final_model = KDE_supermodel(kde_trained_map, new_index_map, new_index_map_inv)

    return final_model