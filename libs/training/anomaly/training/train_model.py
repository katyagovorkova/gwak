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

def main(models: tuple, 
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