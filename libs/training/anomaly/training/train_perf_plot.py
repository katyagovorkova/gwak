import numpy as np
import matplotlib.pyplot as plt

def main(history, savedir):
    loss = history['loss']
    val_loss = history['val_loss']
    N_epochs = len(loss)
    assert len(val_loss) == N_epochs

    plt.figure(figsize=(15, 10))
    plt.plot(loss, label = "loss")
    plt.plot(val_loss, label = "val loss")
    plt.legend()
    plt.xlabel("number of epochs", fontsize=17)
    plt.ylabel("loss", fontsize=17)
    plt.title("Loss curve for training network", fontsize=17)
    #plt.yscale("log")
    plt.savefig(f"{savedir}/loss.png", dpi=300)
    plt.show()