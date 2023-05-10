
import os
#import setGPU
import numpy as np
from keras import losses, callbacks
from keras.optimizers import Adam
from anomaly.training.train_perf_plot import main as perf_plot_main
from anomaly.evaluation import plotting_main
import tensorflow as tf

def split_data(data, frac = 0.9):
    split = int(len(data)*frac)

    train_data = data[:split]
    test_data = data[split:]

    return train_data, test_data

def train_model_main(models: tuple,
        data: np.ndarray,
        savedir: str,
        batch_size:int,
        epochs:int,
        valid_data_2:str=None,
        temp_direc:str=None,
        alpha:float=0):
    '''
    Function to train autoencoder based on some input data,
    save the models to some path
    '''
    assert alpha <= 0
    #global val2_datasets, class_labels
    if temp_direc is not None:
        val2_datasets = []
        if valid_data_2 is not None:
            #load the data
            class_labels = []
            for CL in os.listdir(valid_data_2):
                val2_datasets.append(np.load(f"{valid_data_2}/{CL}"))
                class_labels.append(CL[:-4]) #cut off .npy
    try: #note makedirs is recursive version or mkdir
        # should incorporate this elsewhere in the codebase
        os.makedirs(savedir + "/BY_EPOCH/")
    except FileExistsError:
        None

    if temp_direc is not None:
        try:
            os.makedirs(f"{temp_direc}/TEMP_PLOTS/")

            for CL in class_labels:
                os.makedirs(f"{temp_direc}/TEMP/{CL}/")

        except FileExistsError:
            None

        np.save(f"{temp_direc}/TEMP_PLOTS/n_epochs.npy", np.array([0]))

    #pull apart models
    AE, EN, DE = models

    #split data
    train_data, test_data = split_data(data)

    #hyperparameters
    optimizer = "adam"
    #optimizer = Adam(learning_rate=1e-2)
    #optimizer = None
    loss = losses.mean_absolute_error
    def weird_loss_fn(y_true, y_pred):
        return tf.math.reduce_mean( (tf.math.abs(y_true - y_pred) + tf.math.scalar_mul(alpha, tf.math.abs(y_pred) )) )
    loss = weird_loss_fn
    loss = losses.mean_absolute_error
    #train
    if optimizer is not None:
        AE.compile(optimizer = optimizer,
                    loss=loss)
    else:
        AE.compile(loss=loss)

    class CustomCallback(callbacks.Callback):
        def on_epoch_end(self, batch, logs=None):
            #save the model at each epoch
            num = len(os.listdir(f"{savedir}/BY_EPOCH/"))
            AE.save(f"{savedir}/BY_EPOCH/AE_{num}.h5")

            if temp_direc is None: return None
            make_procedural_plots = 1
            N_epochs = np.load(f"{temp_direc}/TEMP_PLOTS/n_epochs.npy")[0]
            if N_epochs % 20 == 0:
                if temp_direc is not None and make_procedural_plots:

                    for i, X in enumerate(val2_datasets):
                        np.save(f"{temp_direc}/TEMP/{class_labels[i]}/LS_evals.npy",EN.predict(X))
                    #num = len(os.listdir(f"{temp_direc}/TEMP_PLOTS/"))
                    num = N_epochs
                    plotting_main(f"{temp_direc}/TEMP/",
                                f"{temp_direc}/TEMP_PLOTS/{num}/",
                                class_labels,
                                make_QUAK=False,
                                do_LS=True)

            np.save(f"{temp_direc}/TEMP_PLOTS/n_epochs.npy", np.array([N_epochs+1]))


    history = AE.fit(train_data, train_data,
                    batch_size = batch_size,
                    epochs=epochs,
                    validation_data=(test_data, test_data),#),
                    callbacks = [CustomCallback()])

    try: #note makedirs is recursive version or mkdir
        # should incorporate this elsewhere in the codebase
        os.makedirs(savedir)
    except FileExistsError:
        None

    perf_plot_main(history.history, savedir)

    #outdir for saving the models and training history should be a folder
    AE.save(f"{savedir}/AE.h5",include_optimizer=False)

    loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    np.save(f"{savedir}/loss_hist.npy", loss_hist)
    np.save(f"{savedir}/val_loss_hist.npy", val_loss_hist)

    if EN is not None:
        EN.save(f"{savedir}/EN.h5",include_optimizer=False)
    if DE is not None:
        DE.save(f"{savedir}/DE.h5",include_optimizer=False)
    #np.save(f"{savedir}/history.npy", history)

import os
import numpy as np
from anomaly.training.train_model import main as train_model_main
from anomaly.autoencoder_models import main as make_AE_model
from keras.models import load_model

def main(data:list[np.ndarray], 
        model_choice:str,
        savedir:str,
        batch_size:int,
        epochs:int,
        bottleneck:int,
        class_labels:list[str]):

    #print("ALL CLASS LABELS HERE HEREH HERERHERERE", class_labels)

    input_shape = data[0].shape[1:]

    #train the QUAK networks on each class
    N_classes = len(data)
    for i, oneclass_data in enumerate(data):
        
        #Go through each class and train the QUAK networks
        oneclass_savedir = f"{savedir}/{class_labels[i]}/"
        try:
            os.makedirs(oneclass_savedir)
        except FileExistsError:
            None

        continue_where_left_off=True
        load_new_model = True
        num_epochs_trained = 0
        if continue_where_left_off:
            #check if models exist in the savedir
            try:
                os.listdir(f"{savedir}/{class_labels[i]}/BY_EPOCH/")
            except:
                load_new_model = True
            if not load_new_model:
                num_epochs_trained = len(os.listdir(f"{savedir}/{class_labels[i]}/BY_EPOCH/"))
                AE = load_model(f"{savedir}/{class_labels[i]}/BY_EPOCH/AE_{num_epochs_trained-1}.h5")
                print('original number of epochs:', epochs)
                print(f"loaded most recent model, {epochs - num_epochs_trained}")
                print(f"{epochs - num_epochs_trained} more epochs to train")
                load_new_model=False
        
        if load_new_model:
            print("loading new model")
            #don't care about latent space
            AE, _, __ = make_AE_model(model_choice, 
                            input_shape, bottleneck)
        print(f"training QUAK network,  {class_labels[i]}   ")

        #if class_labels[i] != "SG":
         #   print("skipping this one", class_labels[i])
         #   continue
        
        print("ive continued with the class", class_labels[i])
        
        if class_labels[i] == "BKG" or class_labels[i] == "bkg": #just a particular thing
            print("got to alternate argument")
            train_model_main(
            (AE, None, None),
            oneclass_data,
            oneclass_savedir,
            batch_size,
            epochs=epochs - num_epochs_trained, #usually setting this to 5
            alpha=0
            )
        '''
        if class_labels[i] == "BBH": #special case, already done
            print("already finished bbh, continuing (one time thing)")
            continue
        elif class_labels[i] == "BKG": #special case, already done
            print("already finished bkg, continuing (one time thing)")
            continue
        elif class_labels[i] == "GLITCH": #special case, already done
            print("already finished glitch, continuing (one time thing)")
            continue
        '''
        if class_labels[i] != "BKG":
            print('training model with', epochs-num_epochs_trained, "epochs")
            train_model_main(
                (AE, None, None),
                oneclass_data,
                oneclass_savedir,
                batch_size,
                epochs - num_epochs_trained,
                alpha=-0.2
            )
