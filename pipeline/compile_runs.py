import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

'''
Directed at a folder containing networks trained at different bandpass widths
The goal of this code is to "compile" these runs - look at the training performance at different widths
And decide which one is "best" to use for which class (based on some tbd scheme, although I have a general idea)
And create a new folder which contains the desired models for each
Note: maybe the code for running a model on some arbitrary data may need some polishing up
'''
path = "/home/ryan.raikman/s22/anomaly/bp_runs/" #add code for automatically doing this later

def valid_train_name(name):
    if len(name.split("_")) != 2:
        return -1
    label, width = name.split("_")
    if label == "RUNX" and width.isnumeric():
        return int(width)
    return -1

#parse through the files and pick the ones trained on certain bandwidth
valid_files = []
for name in os.listdir(path): 
    res = valid_train_name(name)
    if res != -1:
        valid_files.append( (name, res) )

classes = ['bbh', 'bkg', 'glitches_new', 'injected']

#fetch the loss (and validation loss) histories
loss_hists = dict()
for name, width in valid_files:
    #print("34, name, width", name, width)
    loss_hists[width] = dict()
    for cl in classes:
        loss_path = f"{path}/{name}/TRAINED_MODELS/QUAK/{cl}/loss_hist.npy"
        val_loss_path = f"{path}/{name}/TRAINED_MODELS/QUAK/{cl}/val_loss_hist.npy"

        loss_hists[width][f'{cl}_loss'] = np.load(loss_path)
        loss_hists[width][f'{cl}_val_loss'] = np.load(val_loss_path)
#print("42, loss_hists", loss_hists)    
#first treat the best case in each as the minimum loss per epoch
best_losses = dict()
for width in loss_hists.keys():
    best_losses[width] = dict()
    for cl in classes:
        #print("45, width", width)
        #have to save the locations for picking the best network later
        min_loss = min(loss_hists[width][f'{cl}_loss'])
        #print("49, min_loss", min_loss)
        min_loss_loc = np.where(loss_hists[width][f'{cl}_loss']==min_loss)
        min_val_loss = min(loss_hists[width][f'{cl}_val_loss'])
        min_val_loss_loc = np.where(loss_hists[width][f'{cl}_val_loss']==min_val_loss)[0][0]

        best_losses[width][cl] = {"loss": (min_loss, min_loss_loc),
                                "val_loss": (min_val_loss, min_val_loss_loc)}


#make a plot of the minimum validation losses
savedir = "/home/ryan.raikman/s22/forks/gw-anomaly/pipeline/outputs/graphs" # update this to be an argument
widths = list(best_losses.keys())

MVL = dict()
fig, axs = plt.subplots(2, 2, figsize=(16, 16))
for n, cl in enumerate(classes):
    #plt.figure(figsize = (14, 9)
    i, j = n//2, n%2
    #print("67, i, j", i, j)
    axs[i, j].set_title(f"training across bandwidth, {cl}")
    axs[i, j].set_ylabel("minimum validation loss")
    axs[i, j].set_xlabel("bottleneck width")

    min_val_losses = []
    #print("Class", cl)
    for width in widths:
        #print("Taking value:", width, best_losses[width][cl]['val_loss'][0])
        min_val_losses.append(best_losses[width][cl]['val_loss'][0])
    min_val_losses = np.array(min_val_losses)
    widths = np.array(widths)
    #print("79, widths", widths)
    p = widths.argsort()
    #print("terminated")
    MVL[cl]=min_val_losses[p]
    axs[i, j].plot(widths[p], min_val_losses[p])
    axs[i, j].scatter(widths[p], min_val_losses[p], c='black')
plt.savefig(f"{savedir}/min_val_loss.png", dpi=300)
    #plt.show()
widths = sorted(widths)


#metric: compare the derivative (gain) at each point, and see how it compares to the overall change 
#in validation loss

#for now, I'm just going to choose it by inspection
#widths are 2, 5, 10, 15, 20, 25, 32, 40, 52, 64
print(widths)
best_widths = {"bbh": 20, "bkg": 10, "glitches_new": 25, "injected": 25}

which_network = dict()
for cl in classes:
    epoch_index = best_losses[best_widths[cl]][cl]['val_loss'][1]
    which_network[cl] = (best_widths[cl], epoch_index)

print("105, which_network", which_network)

#now go and pull those networks, and rearrange them in a way that coheres with my standard approach
if not os.path.exists(f"{path}/COMPILED_RUN/TRAINED_MODELS/QUAK/"):
    os.makedirs(f"{path}/COMPILED_RUN/TRAINED_MODELS/QUAK/")
for cl in classes:
    if not os.path.exists(f"{path}/COMPILED_RUN/TRAINED_MODELS/QUAK/{cl}/"):
        os.mkdir(f"{path}/COMPILED_RUN/TRAINED_MODELS/QUAK/{cl}/")
    #put the desired network in 
    width, epoch = which_network[cl]
    network_path = f"{path}/RUNX_{width}/TRAINED_MODELS/QUAK/{cl}/BY_EPOCH/AE_{epoch}.h5"

    #copy it over
    shutil.copyfile(network_path, f"{path}/COMPILED_RUN/TRAINED_MODELS/QUAK/{cl}/AE.h5")

    #/home/ryan.raikman/s22/anomaly/bp_runs/RUNX_25/TRAINED_MODELS/QUAK/bbh/BY_EPOCH/AE_145.h5