import os
import numpy as np
from keras.models import load_model
import scipy.signal as sig

#helper function
def MAE(a, b):
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

def error_STD(a, b):
    '''
    compute error across samples by calculating 
    std of difference
    '''
    return np.squeeze(np.std(a-b, axis=1),axis=1)

def edit(x):
    print("entering into edit", x.shape)
    final = []
    for row in x:
        final.append(np.array([max(1, elem) for elem in row]))
    #print(np.array(final).shape)
    return np.array(final)

def MAPE_edit(data, preds):
    diff = np.abs(data-preds)
    norm_factor = data[0].size
    diff = np.divide(diff, edit(data))
    N = len(diff)

    return np.sum(diff.reshape(N, -1), axis=1)/norm_factor

def LS_predict_data(data:np.ndarray, path: str):
    #only do the prediction for the encoder here
    model = load_model(path + "EN.h5")
    return model.predict(data)

def freq_domain_error(data):
    D = np.copy(data)
    norm_factor = np.std(D, axis=1)
    norm_factor_A = np.std(norm_factor, axis=1)
    
    D = np.array([elem/norm_factor_A[i] for i, elem in enumerate(D)])
    
    _, Pxxs = sig.welch(D)
    norm_factor = np.std(Pxxs, axis = 2)
    norm_factor = np.average(norm_factor, axis=1)   
    random_dir = "/home/ryan.raikman/s22/temp/"
    ind = np.random.randint(10, 20)
    np.save(f"{random_dir}data_{ind}.npy", data)
    np.save(f"{random_dir}data_normvals_{ind}.npy", norm_factor)
    return norm_factor

def QUAK_predict_data(data:np.ndarray, path: str, class_labels:list[str]):
    #this one a little more complicated,
    #want to load each QUAK autoencoder, 
    #calculate loss from each
    N_classes = len(os.listdir(path))
    N_samples = len(data)

    print("XXXX doing quak predict: class_labels, path", class_labels, path)

    QUAK_errors = np.zeros((N_samples, N_classes))
    for i in range(N_classes):
        #load model
        folder_name = f"{path}/{class_labels[i]}/"
        print("I got this for folder:", folder_name)
        AE = load_model(folder_name + "AE.h5")

        #run the autoencoder
        preds = AE.predict(data)
        #print("preds", preds.shape)
        if len(preds.shape)==3 and preds.shape[-1]==1 and len(data.shape)==2:
            preds = np.squeeze(preds, axis=2)
            #print("after", preds.shape)

        #error is the MAE between preds, data
        assert preds.shape == data.shape
        error = MAE(data, preds)

        #now trying out a thing to normalize the errors at the end
        norm_factor = np.std(data, axis=1)
        norm_factor_B = np.std(norm_factor, axis=1) #have this to calculate std across every dim except the first
        #print("DATA PRED DEBUG", error.shape, norm_factor.shape)
        print("SHAPE DEBUG 608", error.shape, norm_factor.shape)
        #norm_factor = freq_domain_error(data)
        QUAK_errors[:, i] = error #/ norm_factor_B#going to add this back in for now...can't hurt?

    return QUAK_errors

def QUAK_edit_predict_data(data:np.ndarray, path: str, class_labels:list[str]):
    #this one a little more complicated,
    #want to load each QUAK autoencoder, 
    #calculate loss from each
    N_classes = len(os.listdir(path))
    N_samples = len(data)

    QUAK_errors = np.zeros((N_samples, N_classes))
    for i in range(N_classes):
        #load model
        folder_name = f"{path}/{class_labels[i]}/"
        print("I got this for folder:", folder_name)
        AE = load_model(folder_name + "AE.h5")

        #run the autoencoder
        preds = AE.predict(data)

        #error is the MAE between preds, data
        assert preds.shape == data.shape
        #error = MAPE_edit(data, preds)
        error = error_STD(data, preds)
        QUAK_errors[:, i] = error

    return QUAK_errors

def main(testdata:list[np.ndarray],
        model_paths:str, 
        savedir:str,
        class_labels:list[str],
        do_LS:bool):

    #model_paths should be a folder with two folders in it:
    #AE_models, LS_models

    #when outputs are done, maintain the organization by class
    #however, going to put all the classes together to start
    #since the vectorization will make this process faster
    #just need to save the indicies so it can be split up later,
    #done by making a slice object for each class

    class_slices = []
    prev = 0
    for class_data in testdata:
        class_slices.append(slice(prev, prev+len(class_data)))
        prev += len(class_data)

    print("CLASS SLICES", class_slices)

    print("954 before vstack", len(testdata), testdata[0].shape, testdata[1].shape)
    testdata = np.vstack(testdata)
    print("955 after vstack", testdata.shape)

    do_std = False
    #do evaluations, make sure that the models are organized in the correct file structure
    QUAK_evals = QUAK_predict_data(testdata, model_paths + "/QUAK/", class_labels)

    if do_std: QUAK_evals_STD = QUAK_edit_predict_data(testdata, model_paths + "/QUAK/", class_labels)
    if do_LS: LS_evals = LS_predict_data(testdata, model_paths + "/LS/")

    assert len(QUAK_evals) == len(testdata)
    if do_LS: assert len(LS_evals) == len(testdata)

    #split them back up by class, and save each to a folder
    for i, class_slice in enumerate(class_slices):
        QUAK_eval_slice = QUAK_evals[class_slice]
        if do_LS: LS_eval_slice = LS_evals[class_slice]
        if do_std: QUAK_eval_STD_slice = QUAK_evals_STD[class_slice]

        folder_path = f"{savedir}/{class_labels[i]}/"
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            None

        np.save(folder_path + "QUAK_evals.npy", QUAK_eval_slice)
        if do_std:  np.save(folder_path + "QUAK_evals_STD.npy", QUAK_eval_STD_slice)
        if do_LS: np.save(folder_path + "LS_evals.npy", LS_eval_slice)