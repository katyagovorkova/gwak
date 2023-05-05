import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

'''
Function which shows the original image, 
versus the autoencoder predicted image.
Useful tool for debugging with no learning going on
'''
def MAE(a, b):
    '''
    compute MAE across a, b
    using first dimension as representing each sample
    '''
    norm_factor = a[0].size
    assert a.shape == b.shape
    diff = np.abs(a-b)
    #N = len(diff)
    
    #sum across all axes except the first one
    return np.sum(diff)/len(diff)

def main(savedir, do_LS, override_savedir=None, limit_data=False):
    #assuming that the ordering of data stays the same, which it should
    N_samples = 3

    #going to do each folder, for now
    original_samples_path = f"{savedir}/DATA/TEST_PROCESS/"

    orig_samples = dict()
    for data_class in os.listdir(original_samples_path):
        loaded = np.load(original_samples_path + data_class)
        print(loaded.shape)
        if limit_data:
            loaded = loaded[:10, :, :]
        #assert 0
        #loaded = np.random.choice(loaded, N_samples, replace=False)
        loaded = loaded[np.random.permutation(len(loaded))][:N_samples]
        orig_samples[data_class[:-4]]=(loaded)

    AE_samples_QUAK = dict()
    AE_samples_LS = dict()

    #do latent space first

    if do_LS: 
        for data_class in orig_samples.keys():
            model_path = f"{savedir}/TRAINED_MODELS/LS/AE.h5"
            LS_model = load_model(model_path)

            AE_samples_LS[data_class]=LS_model.predict(orig_samples[data_class])

    #quak space
    for data_class in orig_samples.keys():
        #only doing quak model for each specific data class
        model_path = f"{savedir}/TRAINED_MODELS/QUAK/{data_class}/AE.h5"
        QUAK_model = load_model(model_path)

        AE_samples_QUAK[data_class]=QUAK_model.predict(orig_samples[data_class])

    #quak to quak
    AE_cross_QUAK = dict()
    for data_class in orig_samples.keys():
        temp = dict()
        for quak_class in orig_samples.keys():
            print("doing quak 2 quak, starting from", data_class, "and evaling with quak from", quak_class)
            model_path = f"{savedir}/TRAINED_MODELS/QUAK/{quak_class}/AE.h5"
            QUAK_model = load_model(model_path)

            temp[quak_class]=QUAK_model.predict(orig_samples[data_class])

        AE_cross_QUAK[data_class] = temp


    try:
        os.makedirs(f"{savedir}/PLOTS/AE_EVALS/")
    except FileExistsError:
        None

    #do the actual plotting
    if 0: #this is built for one detector
        for data_class in orig_samples.keys():
            orig = orig_samples[data_class]
            if do_LS: LS = AE_samples_LS[data_class]
            QUAK = AE_samples_QUAK[data_class]

            print("FOR QUAK2QUAK, SHAPE:", orig.shape, QUAK.shape)
            fig, axs = plt.subplots(N_samples, 1, figsize=(15, 3*N_samples))

            for i in range(N_samples):
                axs[i].plot(orig[i], label = "original")
                if do_LS: axs[i].plot(LS[i], label = "LS")
                axs[i].plot(QUAK[i], label = "QUAK")
                axs[i].legend()
                
                axs[i].set_ylabel("strain")
                axs[i].set_xlabel("time")


            fig.savefig(f"{savedir}/PLOTS/AE_EVALS/{data_class}_AE_evals.png", dpi=300)

    try:
        os.makedirs(f"{savedir}/PLOTS/AE_QUAK2QUAK/")
    except FileExistsError:
        None

    #plotting for cross-quak 
    for data_class in orig_samples.keys():
        if 0:
            None

        else:
            #plotting for 2 detectors
            orig = orig_samples[data_class]
            figshape_X = (17, 5 * N_samples)
            fig, axs = plt.subplots(N_samples, 2, figsize=figshape_X)

            time_axis = 2
            print("117, ORIG SHAPE", orig.shape)
            #assert 0
            if orig.shape[1] > orig.shape[2]:
                print("time axis is now 1")
                time_axis = 1

            for i in range(N_samples):
                if i == 0:
                    axs[i, 0].set_title("Hanford", fontsize = 20)
                #print("working on sample", i)
                
                if time_axis==2:
                    xs = np.arange(0, 1000/4096*len(orig[i, 0, :]), 1000/4096)
                    axs[i, 0].plot(xs, orig[i, 0, :], label = "original, H1")
                    orig_0 = orig[i, 0, :]
                elif time_axis==1:
                    xs = np.arange(0, 1000/4096*len(orig[i, :, 0]), 1000/4096)
                    axs[i, 0].plot(xs, orig[i, :, 0], label = "original, H1")
                    orig_0 = orig[i, :, 0]
                for quak_class in AE_cross_QUAK[data_class].keys():
                    #print("ok, doing the plotting for cross QUAK", quak_class)
                    #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                    if time_axis==2:
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 0, :]), 1000/4096)
                        pred = AE_cross_QUAK[data_class][quak_class][i, 0, :]
                        err = MAE(orig_0, pred)
                        #print("H1 orig, pred shapes:", orig_0.shape, pred.shape)
                        axs[i, 0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 0, :], label = f"{quak_class}, err: {err:.2f}")
                    elif time_axis==1:
                        assert time_axis==1
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 0]), 1000/4096)
                        pred = AE_cross_QUAK[data_class][quak_class][i, :, 0]
                        err = MAE(orig_0, pred)
                        #print("H1 orig, pred shapes:", orig_0.shape, pred.shape)
                        axs[i, 0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 0], label = f"{quak_class}, err: {err:.2f}")

                axs[i, 0].legend()
                axs[i, 0].set_ylabel("normalized strain", fontsize=15)
                axs[i, 0].set_xlabel("time, ms", fontsize=15)

            for i in range(N_samples):
                if i == 0:
                    axs[i, 1].set_title("Livingston", fontsize = 20)
                #print("working on sample", i)
                if time_axis==2:
                    xs = np.arange(0, 1000/4096*len(orig[i, 1, :]), 1000/4096)
                    axs[i, 1].plot(xs, orig[i, 1, :], label = "original, L1")
                    orig_1 = orig[i, 1, :]
                elif time_axis==1:
                    assert time_axis==1
                    xs = np.arange(0, 1000/4096*len(orig[i, 1, :]), 1000/4096)
                    axs[i, 1].plot(xs, orig[i, 1, :], label = "original, L1")
                    orig_1 = orig[i, :, 1]
                    
                for quak_class in AE_cross_QUAK[data_class].keys():
                    #print("ok, doing the plotting for cross QUAK", quak_class)
                    #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                    if time_axis==2:
                        pred = AE_cross_QUAK[data_class][quak_class][i, 1, :]
                        err = MAE(orig_1, pred)
                        #print("L1 orig, pred shapes:", orig_1.shape, pred.shape)
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 1, :]), 1000/4096)
                        axs[i, 1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 1, :], label = f"{quak_class}, err: {err:.2f}")
                    elif time_axis==1:
                        assert time_axis==1
                        pred = AE_cross_QUAK[data_class][quak_class][i, :, 1]
                        err = MAE(orig_1, pred)
                        #print("L1 orig, pred shapes:", orig_1.shape, pred.shape)
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 1]), 1000/4096)
                        axs[i, 1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 1], label = f"{quak_class}, err: {err:.2f}")

                axs[i, 1].legend()
                #axs[i, 1].set_ylabel("normalized strain", fontsize=15)
                axs[i, 1].set_xlabel("time, ms", fontsize=15)
            fig.suptitle(f"{data_class} recreation plots", fontsize=25)
            fig.tight_layout()
            
            if override_savedir is None:
                fig.savefig(f"{savedir}/PLOTS/AE_QUAK2QUAK/{data_class}_QUAK2QUAK.png", dpi=300)
            else:
                fig.savefig(f"{override_savedir}/{data_class}_QUAK2QUAK.png", dpi=300)

            #DOING THE SAME THING, BUT ONLY SHOWING 1-1
            #plotting for 2 detectors
            orig = orig_samples[data_class]
            fig, axs = plt.subplots(N_samples, 2, figsize=figshape_X)

            for i in range(N_samples):
                if i == 0:
                    axs[i, 0].set_title("Hanford", fontsize = 20)
                #print("working on sample", i)
                if time_axis==2:
                    xs = np.arange(0, 1000/4096*len(orig[i, 0, :]), 1000/4096)
                    axs[i, 0].plot(xs, orig[i, 0, :], label = "original, H1")
                elif time_axis==1:
                    assert time_axis==1
                    xs = np.arange(0, 1000/4096*len(orig[i, :, 0]), 1000/4096)
                    axs[i, 0].plot(xs, orig[i, :, 0], label = "original, H1")

                for quak_class in AE_cross_QUAK[data_class].keys():
                    if quak_class == data_class:
                    #print("ok, doing the plotting for cross QUAK", quak_class)
                    #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                        if time_axis==2:
                            xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 0, :]), 1000/4096)
                            axs[i, 0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 0, :], label = quak_class)
                        elif time_axis==1:
                            xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 0]), 1000/4096)
                            axs[i, 0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 0], label = quak_class)
                axs[i, 0].legend()
                axs[i, 0].set_ylabel("normalized strain", fontsize=15)
                axs[i, 0].set_xlabel("time, ms", fontsize=15)

            for i in range(N_samples):
                if i == 0:

                    axs[i, 1].set_title("Livingston", fontsize = 20)
                #print("working on sample", i)
                if time_axis==2:
                    xs = np.arange(0, 1000/4096*len(orig[i, 1, :]), 1000/4096)
                    axs[i, 1].plot(xs, orig[i, 1, :], label = "original, L1")
                elif time_axis==1:
                    xs = np.arange(0, 1000/4096*len(orig[i, :, 1]), 1000/4096)
                    axs[i, 1].plot(xs, orig[i, :, 1], label = "original, L1")

                for quak_class in AE_cross_QUAK[data_class].keys():
                    if quak_class == data_class:
                    #print("ok, doing the plotting for cross QUAK", quak_class)
                    #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                        if time_axis==2:
                            xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 1, :]), 1000/4096)
                            axs[i, 1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 1, :], label = quak_class)
                        elif time_axis==1:
                            xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 1]), 1000/4096)
                            axs[i, 1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 1], label = quak_class)
                axs[i, 1].legend()
                #axs[i, 1].set_ylabel("normalized strain", fontsize=15)
                axs[i, 1].set_xlabel("time, ms", fontsize=15)
            fig.suptitle(f"{data_class} recreation plots", fontsize=25)
            fig.tight_layout()
            if override_savedir is None:
                fig.savefig(f"{savedir}/PLOTS/AE_QUAK2QUAK/{data_class}_QUAK2QUAK_121.png", dpi=300)
            else:
                fig.savefig(f"{override_savedir}/{data_class}_QUAK2QUAK_121.png", dpi=300)

def main_limited(savedir, do_LS, override_savedir=None, limit_data=False):
    #assuming that the ordering of data stays the same, which it should
    N_samples = 5

    #going to do each folder, for now
    original_samples_path = f"{savedir}/DATA/TEST_PROCESS/"

    orig_samples = dict()
    for data_class in os.listdir(original_samples_path):
        loaded = np.load(original_samples_path + data_class)
        print(loaded.shape)
        if limit_data:
            loaded = loaded[:10, :, :]
        #assert 0
        #loaded = np.random.choice(loaded, N_samples, replace=False)

        #loaded = loaded[np.random.permutation(len(loaded))][:N_samples]
        loaded = loaded[:N_samples]
        orig_samples[data_class[:-4]]=(loaded)

    AE_samples_QUAK = dict()
    AE_samples_LS = dict()

    #do latent space first

    if do_LS: 
        for data_class in orig_samples.keys():
            model_path = f"{savedir}/TRAINED_MODELS/LS/AE.h5"
            LS_model = load_model(model_path)

            AE_samples_LS[data_class]=LS_model.predict(orig_samples[data_class])

    #quak space
    for data_class in orig_samples.keys():
        #only doing quak model for each specific data class
        model_path = f"{savedir}/TRAINED_MODELS/QUAK/{data_class}/AE.h5"
        QUAK_model = load_model(model_path)

        AE_samples_QUAK[data_class]=QUAK_model.predict(orig_samples[data_class])

    #quak to quak
    AE_cross_QUAK = dict()
    for data_class in orig_samples.keys():
        temp = dict()
        for quak_class in orig_samples.keys():
            print("doing quak 2 quak, starting from", data_class, "and evaling with quak from", quak_class)
            model_path = f"{savedir}/TRAINED_MODELS/QUAK/{quak_class}/AE.h5"
            QUAK_model = load_model(model_path)

            temp[quak_class]=QUAK_model.predict(orig_samples[data_class])

        AE_cross_QUAK[data_class] = temp


    try:
        os.makedirs(f"{savedir}/PLOTS/AE_EVALS/")
    except FileExistsError:
        None

    #do the actual plotting
    
    try:
        os.makedirs(f"{savedir}/PLOTS/AE_QUAK2QUAK/")
    except FileExistsError:
        None

    #plotting for cross-quak 
    for data_class in orig_samples.keys():
        #plotting for 2 detectors
        orig = orig_samples[data_class]
        figshape_X = (13, 5)
        fig, axs = plt.subplots(1, 2, figsize=figshape_X)

        time_axis = 2
        print("117, ORIG SHAPE", orig.shape)
        #assert 0
        if orig.shape[1] > orig.shape[2]:
            print("time axis is now 1")
            time_axis = 1

        #for i in range(N_samples):
        i = 1
        color_map = {
            "SG": "red",
            "GLITCH": "green",
            "BKG": "purple",
            "BBH": "blue"
        }
        if True:
            #if i == 0:
            axs[0].set_title("Hanford", fontsize = 20)
            #print("working on sample", i)
            
            if time_axis==2:
                xs = np.arange(0, 1000/4096*len(orig[i, 0, :]), 1000/4096)
                axs[0].plot(xs, orig[i, 0, :], label = "original", c="black", linewidth=1.5)
                orig_0 = orig[i, 0, :]
            elif time_axis==1:
                xs = np.arange(0, 1000/4096*len(orig[i, :, 0]), 1000/4096)
                axs[0].plot(xs, orig[i, :, 0], label = "original", c="black", linewidth=1.5)
                orig_0 = orig[i, :, 0]
            for quak_class in AE_cross_QUAK[data_class].keys():
                #print("ok, doing the plotting for cross QUAK", quak_class)
                #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                if time_axis==2:
                    xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 0, :]), 1000/4096)
                    pred = AE_cross_QUAK[data_class][quak_class][i, 0, :]
                    err = MAE(orig_0, pred)
                    #print("H1 orig, pred shapes:", orig_0.shape, pred.shape)
                    axs[0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 0, :], alpha=0.7, 
                                    label = f"{quak_class}, err: {err:.2f}",
                                    c=color_map[quak_class])
                elif time_axis==1:
                    assert time_axis==1
                    xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 0]), 1000/4096)
                    pred = AE_cross_QUAK[data_class][quak_class][i, :, 0]
                    err = MAE(orig_0, pred)
                    #print("H1 orig, pred shapes:", orig_0.shape, pred.shape)
                    axs[0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 0], alpha=0.7, 
                                    label = f"{quak_class}, err: {err:.2f}",
                                    c=color_map[quak_class])

            axs[0].legend()
            axs[0].grid()
            axs[0].set_ylabel("normalized strain", fontsize=15)
            axs[0].set_xlabel("time, ms", fontsize=15)
            axs[0].set_ylim(None, 3.1)
        if True:
            #if i == 0:
            axs[1].set_title("Livingston", fontsize = 20)
            #print("working on sample", i)
            if time_axis==2:
                xs = np.arange(0, 1000/4096*len(orig[i, 1, :]), 1000/4096)
                axs[1].plot(xs, orig[i, 1, :], label = "original", c="black")
                orig_1 = orig[i, 1, :]
            elif time_axis==1:
                assert time_axis==1
                xs = np.arange(0, 1000/4096*len(orig[i, 1, :]), 1000/4096)
                axs[1].plot(xs, orig[i, 1, :], label = "original", c="black")
                orig_1 = orig[i, :, 1]
                
            for quak_class in AE_cross_QUAK[data_class].keys():
                #print("ok, doing the plotting for cross QUAK", quak_class)
                #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                if time_axis==2:
                    pred = AE_cross_QUAK[data_class][quak_class][i, 1, :]
                    err = MAE(orig_1, pred)
                    #print("L1 orig, pred shapes:", orig_1.shape, pred.shape)
                    xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 1, :]), 1000/4096)
                    axs[1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 1, :], alpha=0.7, label = f"{quak_class}, err: {err:.2f}", c=color_map[quak_class])
                elif time_axis==1:
                    assert time_axis==1
                    pred = AE_cross_QUAK[data_class][quak_class][i, :, 1]
                    err = MAE(orig_1, pred)
                    #print("L1 orig, pred shapes:", orig_1.shape, pred.shape)
                    xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 1]), 1000/4096)
                    axs[1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 1], alpha=0.7, label = f"{quak_class}, err: {err:.2f}", c=color_map[quak_class])

            axs[1].legend()
            #axs[i, 1].set_ylabel("normalized strain", fontsize=15)
            axs[1].set_xlabel("time, ms", fontsize=15)
            axs[1].grid()
            axs[1].set_ylim(None, 3.1)
        fig.suptitle(f"{data_class} recreation plots", fontsize=25)
        fig.tight_layout()
        
        if override_savedir is None:
            fig.savefig(f"{savedir}/PLOTS/AE_QUAK2QUAK/{data_class}_QUAK2QUAK.png", dpi=300)
        else:
            fig.savefig(f"{override_savedir}/{data_class}_QUAK2QUAK.png", dpi=300)

        #DOING THE SAME THING, BUT ONLY SHOWING 1-1
        #plotting for 2 detectors
        orig = orig_samples[data_class]
        fig, axs = plt.subplots(1, 2, figsize=figshape_X)

        #for i in range(N_samples):
        i = 0
        if True:
            if i == 0:
                axs[0].set_title("Hanford", fontsize = 20)
            #print("working on sample", i)
            if time_axis==2:
                xs = np.arange(0, 1000/4096*len(orig[i, 0, :]), 1000/4096)
                axs[0].plot(xs, orig[i, 0, :], label = "original, H1")
            elif time_axis==1:
                assert time_axis==1
                xs = np.arange(0, 1000/4096*len(orig[i, :, 0]), 1000/4096)
                axs[0].plot(xs, orig[i, :, 0], label = "original, H1")

            for quak_class in AE_cross_QUAK[data_class].keys():
                if quak_class == data_class:
                #print("ok, doing the plotting for cross QUAK", quak_class)
                #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                    if time_axis==2:
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 0, :]), 1000/4096)
                        axs[0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 0, :], label = quak_class)
                    elif time_axis==1:
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 0]), 1000/4096)
                        axs[0].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 0], label = quak_class)
            axs[0].legend()
            axs[0].set_ylabel("normalized strain", fontsize=15)
            axs[0].set_xlabel("time, ms", fontsize=15)

        for i in range(N_samples):
            if i == 0:

                axs[1].set_title("Livingston", fontsize = 20)
            #print("working on sample", i)
            if time_axis==2:
                xs = np.arange(0, 1000/4096*len(orig[i, 1, :]), 1000/4096)
                axs[1].plot(xs, orig[i, 1, :], label = "original, L1")
            elif time_axis==1:
                xs = np.arange(0, 1000/4096*len(orig[i, :, 1]), 1000/4096)
                axs[1].plot(xs, orig[i, :, 1], label = "original, L1")

            for quak_class in AE_cross_QUAK[data_class].keys():
                if quak_class == data_class:
                #print("ok, doing the plotting for cross QUAK", quak_class)
                #print("the shape of data to be plotted:", AE_cross_QUAK[data_class][quak_class].shape)
                    if time_axis==2:
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, 1, :]), 1000/4096)
                        axs[1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, 1, :], label = quak_class)
                    elif time_axis==1:
                        xs = np.arange(0, 1000/4096*len(AE_cross_QUAK[data_class][quak_class][i, :, 1]), 1000/4096)
                        axs[1].plot(xs, AE_cross_QUAK[data_class][quak_class][i, :, 1], label = quak_class)
            axs[1].legend()
            #axs[i, 1].set_ylabel("normalized strain", fontsize=15)
            axs[1].set_xlabel("time, ms", fontsize=15)
        fig.suptitle(f"{data_class} recreation plots", fontsize=25)
        fig.tight_layout()
        if override_savedir is None:
            fig.savefig(f"{savedir}/PLOTS/AE_QUAK2QUAK/{data_class}_QUAK2QUAK_121.png", dpi=300)
        else:
            fig.savefig(f"{override_savedir}/{data_class}_QUAK2QUAK_121.png", dpi=300)

#main("/home/ryan.raikman/s22/anomaly/march23_nets/", False)