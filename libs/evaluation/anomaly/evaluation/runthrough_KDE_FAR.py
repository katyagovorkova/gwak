import numpy as np
import os
from anomaly.evaluation import QUAK_predict, data_segment, NN_predict, KDE_train, KDE_train_BKG
import matplotlib.pyplot as plt

def make_kernel(N):
    return np.ones(N)/N

N_kernel=50
kernel = make_kernel(N_kernel)


def smooth_samples(data):
    new_len = max(data.shape[1], N_kernel) - min(data.shape[1], N_kernel) + 1
    data_smooth = np.empty((data.shape[0], new_len, data.shape[2]))
    for j in range(len(data)):
        for k in range(data.shape[2]):
            #valid mode takes care of cutting off the edge effects
            data_smooth[j, :, k] = np.convolve(data[j, :, k], kernel, mode='valid')

    return data_smooth


def KDE_plotting(data, savedir):
    plt.figure(figsize=(12, 7))
    sample=0
    plt.plot(data[sample, :, 0], label = 'BBH')
    plt.plot(data[sample, :, 1], label = 'BKG')
    plt.plot(data[sample, :, 2], label = 'GLITCH')
    plt.plot(data[sample, :, 3], label = 'SG')
    plt.legend()
    plt.xlabel("datapoints")
    plt.ylabel("KDE values")
    plt.ylim(0,200)
    plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_BBH_runthrough.png", dpi=300)
    plt.show()

def katya_metric(data):
    #sum of two signal losses minus bkg and glitch losses
    return data[:, 0] - data[:, 1] - data[:, 2] + data[:, 3]

def katya_metric2(data):
    #sum of two signal losses minus bkg and glitch losses
    return 2*data[:, 0] - data[:, 1] - data[:, 2] + 2*data[:, 3]

def main(savedir:str, eval_savedir:str):

    #KDE_model = KDE_train(savedir)

    data = np.load( "/home/ryan.raikman/s22/anomaly/generated_data_2_1/1242107200_1242117912/bbh_segs.npy")
    print("orig data shape", data.shape)
    data = np.swapaxes(data, 0, 1) #to make it samples, features, 2
    data = np.swapaxes(data, 1, 2)
    ind_ = 0
    data_strain = data[ind_:ind_+1, :, :]
    print("rearranged data shape,", data.shape)
    data_segs = data_segment(data_strain, 100, 2)
    print(data_segs.shape)
    
    QUAK_values = QUAK_predict(savedir, data_segs)
    
    #ssert 0
    eval_on_kde=False
    if eval_on_kde:
        KDE_values = KDE_model.eval(QUAK_values)

        
        KDE_values_smooth = smooth_samples(KDE_values)
        KDE_plotting(KDE_values_smooth, savedir)
        print("final", KDE_values_smooth.shape)
        KDE_values_smooth = KDE_values_smooth[0, :, :]


        #load the scatter values
        

        KDE_mod = np.maximum(KDE_values_smooth[:, 1], KDE_values_smooth[:, 2])
        print("67", KDE_mod.shape)
        data = KDE_mod
    else:
        QUAK_values_smooth = smooth_samples(QUAK_values)[0, :, :]
        print("78, QUAK_values_smooth", QUAK_values_smooth.shape)
        data = katya_metric2(QUAK_values_smooth)
        print("80, data.shape (after metric), ", data.shape)
    SCY = np.load(f"{savedir}/PLOTS/scatter_y_2.npy")
    SCX = np.load(f"{savedir}/PLOTS/scatter_x_2.npy")
    out = []
    #print("SCY", SCY)
    #print("SCX", SCX)
    for val in data:
    
        ind = np.searchsorted(SCX, val)
       # print("ind, val", ind, val)
        try:
            out.append(SCY[ind])
        except IndexError:
            try:
                out.append(SCY[ind-1])
            except IndexError:
                try:
                    out.append(SCY[ind+1])
                except IndexError:
                    None
                    print("got here!")

    out = np.array(out)
    if 0:
        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(len(out)), out)
        plt.xlabel("time, datapoints")
        plt.ylabel("FAR")
        plt.title("FAR(t) for simluated BBH signal")
        
        plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_BBH_FAR.png", dpi=300)
    if 1:
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        axs[0].plot(np.arange(len(data)), data)
        axs[0].set_xlabel("time, datapoints")
        axs[0].set_ylabel("metric value")
        axs[0].set_title("metric over time")

        axs[1].plot(np.arange(len(out)), out)
        axs[1].set_xlabel("time, datapoints")
        axs[1].set_ylabel("FAR")
        axs[1].set_title("FAR(t) for simluated BBH signal")
        axs[1].set_yscale("log")

        axs[2].plot(data_strain[0, :, 0], label = "H1")
        axs[2].plot(data_strain[0, :, 1], label = "L1")
        axs[2].set_xlabel("time, datapoints")
        axs[2].set_ylabel("detector strain")
        axs[2].set_title("detector readout of signal")

        axs[3].plot(np.arange(len(QUAK_values_smooth)), QUAK_values_smooth[:, 0], label = 'bbh kde')
        axs[3].plot(np.arange(len(QUAK_values_smooth)), QUAK_values_smooth[:, 1], label = 'bkg kde')
        axs[3].plot(np.arange(len(QUAK_values_smooth)), QUAK_values_smooth[:, 2], label = 'glitch kde')
        axs[3].plot(np.arange(len(QUAK_values_smooth)), QUAK_values_smooth[:, 3], label = 'sg kde')
        axs[3].legend()

        axs[3].set_xlabel("time, datapoints")
        axs[3].set_ylabel("QUAK loss")
        axs[3].set_title("QUAK over time")

        fig.tight_layout()
        
        fig.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_BBH_FAR_{ind_}_2.png", dpi=300)
    #now compute FARs...somehow

main("/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/", 
                "/home/ryan.raikman/s22/anomaly/TS_evals10/")
