import numpy as np
import matplotlib.pyplot as plt
import os


def main(model_path:str):
    data_path = f"{model_path}/TRAINED_MODELS/QUAK/"
    plt.figure(figsize=(8, 5))
    color_index = {"BBH":"blue","BKG":"purple", "GLITCH":"green", "SG":"red"}
    name_index = {"BBH":"BBH","BKG":"Background", "GLITCH":"Glitch", "SG":"Sine Gaussian"}
    for class_name in os.listdir(data_path):
        loss = np.load(f"{data_path}/{class_name}/loss_hist.npy")
        val_loss = np.load(f"{data_path}/{class_name}/val_loss_hist.npy")

        #print("loss shape", loss.shape)
        #print("val loss shape", loss.shape)
        col = color_index[class_name]
        name = name_index[class_name]
        plt.plot(np.linspace(0, 100, 100), loss, "-", label = f"{name} ", c= col)
        plt.plot(np.linspace(0, 100, 100), val_loss, "--", c=col)#, label = f"{class_name} validation loss"

    plt.xlabel("Epoch")
    plt.grid()
    plt.ylabel("Mean Absolute Error")
    plt.title("Network Training Loss curve")
    plt.legend(loc=(0.75, 0.45))
    plt.savefig(f"/home/ryan.raikman/s22/temp8/loss_plot.png", dpi=400)

    

model_path = "/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/"
main(model_path)

from anomaly.evaluation import autoencoder_prediction_main_limited as APML

APML(model_path, False, override_savedir="/home/ryan.raikman/s22/temp8/", limit_data=True)