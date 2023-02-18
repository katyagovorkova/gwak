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
