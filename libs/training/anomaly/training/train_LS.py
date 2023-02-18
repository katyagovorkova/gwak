import os
import numpy as np
from anomaly.training.train_model import main as train_model_main
from anomaly.autoencoder_models import main as make_AE_model

def main(data:list[np.ndarray], 
        model_choice:str,
        savedir:str,
        batch_size:int,
        epochs:int,
        bottleneck:int,
        valid_data_2:str=None):

    input_shape = data[0].shape[1:]

    #combine the data across classes
    combined_data = np.vstack(data)
    print("LS COMBINED DATA", combined_data.shape)
    #shuffle for better training
    p = np.random.permutation(len(combined_data))
    combined_data = combined_data[p]

    AE, EN, DE = make_AE_model(model_choice, input_shape, bottleneck)
    print("training LATENT SPACE network")
    train_model_main(
        (AE, EN, DE),
        combined_data,
        savedir,
        batch_size,
        epochs,
        valid_data_2,
        savedir
    )



