import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
import scipy.stats as st


'''
Code to take in segments longer than what was trained on,
procedurally run the autoencoders on it, and make a graph
of a loss metric over time
'''
'''
note:
it would be better to incorporate the data-preprocessing with this,
but I'm just going to copy it here for now 
'''
def density_plot(x, y):
    # Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

   # return ax.contour(xx, yy, f, colors='k')
    return xx, yy, f

def mae(a, b):
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

def split_data(data, overlap, seglen=100):
    assert data.shape[1] > seglen
    num_out = (data.shape[1]- seglen)//overlap
    final = np.zeros((num_out, 2, seglen))

    for i in range(num_out):
        a, b = i*overlap, i*overlap+seglen
        final[i, :] = data[:, a:b]
    print("after splitting 37 " , final.shape)
    return final

def pre_processing_method_NOT_IN_USE(data):
	#individually normalize each segment
    #data -= np.average(data, axis=1)[:, np.newaxis] #doesn't seem right to do
    std_vals = np.std(data, axis=1)
    data /= std_vals[:, np.newaxis]
    return data[:, :, np.newaxis] #for the LSTM stuff, the extra axis is needed

def pre_processing_method(data): #for two detectors at once
    print("DATA SHAPE", data.shape)
	#individually normalize each segment
    #data -= np.average(data, axis=1)[:, np.newaxis] #doesn't seem right to do
    std_vals = np.std(data, axis=2)
    print("std vals shape", std_vals.shape)
    data /= std_vals[:,:, np.newaxis]
    
    print("now data shape:", data.shape)
    #assert False

    #return (data, std_vals)
    #return (data[:, :, np.newaxis], std_vals) #for the LSTM stuff, the extra axis is needed
    return data #unless using 2 detector streams!
def main(data_dir:str,
        savedir:str,
        overlap:int,
        KDE_models:dict=None,
        NN_quak:bool=False):

    do_KDE = False
    if KDE_models is not None:
        do_KDE = True

    if NN_quak:
        #load the QUAK NN model
        nn_quak_model = load_model(f"{savedir}/TRAINED_MODELS/QUAK_NN/quak_nn.h5")
    
    
    # load data
    iden = data_dir.split("/")[-1]
    if iden == "" or iden == None:
        iden = data_dir.split("/")[-2]
    print('iden', iden)
    N_data_classes = len(os.listdir(data_dir))
    data_dict = dict()
    orig_data = dict()
    for i in range(N_data_classes):
        full_data_path = f"{data_dir}/{os.listdir(data_dir)[i]}"
        name = os.listdir(data_dir)[i][:-4] #cut off .npy
        data_dict[name] = np.load(full_data_path)
        orig_data[name] = np.load(full_data_path)

    #split and pre-process
    for key in data_dict:
        print("DEBUG 77", data_dict[key].shape, overlap)
        data_dict[key] = split_data(data_dict[key], overlap)
        data_dict[key] = pre_processing_method(data_dict[key])

    #load QUAK models
    QUAK_models = dict()
    QUAK_model_path = f"{savedir}/TRAINED_MODELS/QUAK/"
    for QUAK_class in os.listdir(QUAK_model_path):
        print("QUAK class", QUAK_class)
        QUAK_models[QUAK_class] = load_model(f"{QUAK_model_path}/{QUAK_class}/AE.h5")

    #do quak evaluation for each data class by each QUAK model
    QUAK_evals = dict()
    QUAK_preds = dict()
    KDE_evals = dict()
    if do_KDE: KDE_evals = dict()
    if NN_quak: NNQUAK_evals = dict()

    for data_class in data_dict:
        QUAK_evals[data_class] = dict()
        QUAK_preds[data_class] = dict()
        for QUAK_class in QUAK_models:
            print("95, data_class, QUAK_class", data_class, QUAK_class)
            pred = QUAK_models[QUAK_class].predict(data_dict[data_class])
            QUAK_preds[data_class][QUAK_class] = pred
            QUAK_evals[data_class][QUAK_class] = mae(data_dict[data_class], pred)
            print("99, mae from autoencoder", QUAK_evals[data_class][QUAK_class])
            out_len = len(mae(data_dict[data_class], pred))


        print("QUAK_classes", QUAK_models.keys())
        QUAK_stack = np.zeros(shape=(len(QUAK_evals[data_class]['bbh']), 4))
        index_map = {'bbh': 0,
                    "bkg": 1,
                    "glitches_new":2,
                    "injected":3}
        index_map_inv = {0:'bbh',
                    1:"bkg",
                    2:"glitches_new",
                    3:"injected"}
        
        for val in index_map:
            QUAK_stack[:, index_map[val]] = QUAK_evals[data_class][val]

        if NN_quak:
            NNQUAK_evals[data_class]=nn_quak_model.predict(QUAK_stack)

        if do_KDE:
            #stack the KDE predictions for input into the KDE model
            KDE_evals[data_class] = dict()
            for KDE_class in KDE_models:
                print("doing KDE for runthrough , 133")
                print("138", QUAK_stack.shape, QUAK_stack.T.shape)
                pred = KDE_models[KDE_class](QUAK_stack.T)

                KDE_evals[data_class][KDE_class] = pred
                len_pred = len(pred)

            #normalization scheme across datapoint
            for i in range(len_pred):
                #each data sample
                tot = 0
                for k in range(4):
                    #each class
                    tot += KDE_evals[data_class][index_map_inv[k]][i]

                for k in range(4):
                    #each class
                    KDE_evals[data_class][index_map_inv[k]][i] /= tot
            
    #break up the NN quak to adhere with scheme for other variables
    # NNQUAK_evals

    index_map_inv = {0:'bbh',
            1:"bkg",
            2:"glitches_new",
            3:"injected"}

    NN_QUAK_evals_split = dict()
    for data_class in data_dict:
        NN_QUAK_evals_split[data_class] = dict()
        for i in range(4):
            NN_QUAK_evals_split[data_class][index_map_inv[i]] = NNQUAK_evals[data_class][:, i]

    #print("196, NN_QUAK_evals_split", NN_QUAK_evals_split)


    #create save directory for plots
    try:
        os.makedirs(f"{savedir}/PLOTS/RUNTHROUGH/")
    except FileExistsError:
        None
    #plotting
    n_sample = len(os.listdir(f"{savedir}/PLOTS/RUNTHROUGH/"))
    try:
        os.makedirs(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/")
    except FileExistsError:
        None
    x_vals = np.arange(0, out_len, 1) * overlap
    print("keys", data_dict.keys())
    SN_plotting = True

    if SN_plotting:
        #do some extra plotting for the SN

        #plot over all of the files in the folder
        #so going over every key in QUAK_evals
        for key in QUAK_evals:
            sn_errs = QUAK_evals[key]
            sn_preds = QUAK_preds[key]
            sn_data = orig_data[key]
            sn_process = data_dict[key]
            
            #print("got past")
            #print("SN SHAPES", sn_errs, sn_preds, sn_data.shape)
            print("SN preds", sn_preds.keys())
            #print("SN PREDS shape", sn_preds.shape)
            try:
                os.makedirs(f"{savedir}/PLOTS/SN/{iden}/")
            except FileExistsError:
                None

            #first just a plot showing the supernova waveform
            plt.figure(figsize=(13, 8))
            plt.title(f"{key} GW Signature", fontsize = 25)
            plt.xlabel("Time, ms", fontsize=15)
            plt.ylabel("Whitened strain", fontsize=15)
            xs = np.arange(0, 1000/4096*len(sn_data[0, :]), 1000/4096)
            #print("SHAPE 131", xs.shape, sn_data[0, :].shape)
            plt.plot(xs, sn_data[0, :], label = "H1")
            plt.plot(xs, sn_data[1, :], label = "L1")
            plt.legend()
            width = 15*2  
            #plt.xlim(len(xs)/2-width/2, len(xs)/2+width/2)
            plt.savefig(f"{savedir}/PLOTS/SN/{iden}/strain_plots_{key}.png", dpi=300)

            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            detecs = ["Hanford", "Livingston"]
            axs[0].set_ylabel("Whitened strain", fontsize=15)

            #offset by 5 datapoints - 5 * 1000/4096 = 1.22 milliseconds per step
            time_per_step = 5*1000/4096
            #ind = int(700/time_per_step)
            #get one from the middle
            ind = len(sn_process)//2
            time_start = time_per_step * ind
            for i in range(2): #each detector
                #plt.figure(figsize=(13, 8))
                axs[i].set_title(detecs[i], fontsize=16)
                axs[i].set_xlabel("Time, ms", fontsize=15)
                
                xs = np.arange(time_start, time_start+len(sn_process[ind][i, :]) * 1000/4096, 1000/4096)

                axs[i].plot(xs, sn_process[ind][i, :], label = "original")
                axs[i].plot(xs, sn_preds["injected"][ind][i, :], label = "sine-gaussian recreation")
                #print("SHAPE 131", xs.shape, sn_data[0, :].shape)
                #axs[0, i].plot(xs, sn_data[i, :], label = "original")
                #plt.plot(xs, sn_data[1, :], label = "L1")
                axs[i].legend()      
                #axs[0, i].xlim(700, 730)
            fig.suptitle(f"{key} GW Recreation", fontsize = 25)
            fig.tight_layout()
            fig.savefig(f"{savedir}/PLOTS/SN/{iden}/QUAK_recreation_{key}.png", dpi=300)

    #assert False #just not to run the stuff below for now
    if 0:
        for data_class in QUAK_evals:
            plt.figure(figsize = (14, 7.5))

            for QUAK_class in QUAK_evals[data_class]:
                plt.plot(x_vals, QUAK_evals[data_class][QUAK_class], label = QUAK_class)

            plt.legend()
            plt.xlabel("time, datapoints", fontsize=14)
            plt.ylabel("autoencoder recreation loss", fontsize=14)
            plt.title(f"procedural AE loss, {data_class} sample", fontsize=17)

            plt.savefig(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/{data_class}_PROCEDURAL.png", dpi=300)
            plt.show()


    #want to extract the "interesting" points as samples of the QUAK space to make a plot of the distribution
    #of SN event points in the QUAK space

    #two strategies: one, just take the minimum points for the SG and BBH network respectively, 
    #but this seems rather arbirtrary, so instead, take all the points where at least one of the 
    #QUAK models has a loss of below 0.65 (or some other value, but this is at least a place to start)
    #in reality, this number should be coded as some std away from the background distribution in the autoencoders

    QUAK_barring_method=False
    if QUAK_barring_method:
        y_bar = 0.65 #arbitrary choice, for future calibrate as above
        #data_class labels the sample, QUAK_class are the actual classes
        QUAK_samples = dict()
        for data_sample in QUAK_evals:
            times_of_interest = set()
            for QUAK_class in QUAK_evals[data_sample]:
                class_data = QUAK_evals[data_class][QUAK_class]
                #now, pick out points below the bar
                for ind in np.where(class_data<y_bar)[0]:
                    times_of_interest.add(ind)
            #SG, glitch, bkg, bbh
            QUAK_space_samples = np.zeros((len(times_of_interest), 4))
            print("FOUND POINTS OF INTEREST", times_of_interest)
            for i, t in enumerate(times_of_interest):
                point = np.zeros(4)
                #check ordering of classes, especially important for plotting in QUAK space
                point[3] = QUAK_evals[data_sample]['injected'][t]
                point[2] = QUAK_evals[data_sample]['glitches_new'][t]
                point[1] = QUAK_evals[data_sample]['bkg'][t]
                point[0] = QUAK_evals[data_sample]['bbh'][t]

                QUAK_space_samples[i, :] = point

            QUAK_samples[data_sample] = QUAK_space_samples
    else:
        QUAK_samples = dict()
        #barring method based on NN output, barred at 0.8
        for data_sample in NN_QUAK_evals_split:
            y_bar = 0.55
            NN_samples = dict()
            bkg_vals = NN_QUAK_evals_split[data_sample]['bkg']

            #KW = 25
            #dseg = bkg_vals
            #kernel = np.ones((KW,))/KW
            #conv_dseg = np.convolve(dseg, kernel, mode = 'valid')
            conv_dseg = bkg_vals

            #times_of_interest = np.where(conv_dseg<y_bar)[0]
            times_of_interest = np.where(conv_dseg == min(conv_dseg))[0]
    
            QUAK_space_samples = np.zeros((len(times_of_interest), 4))
            print("LEN SAMPLE", len(QUAK_evals[data_sample]['bbh']))
            print("FOUND POINTS OF INTEREST", times_of_interest)
            for i, t in enumerate(times_of_interest):
                point = np.zeros(4)
                #check ordering of classes, especially important for plotting in QUAK space
                point[3] = QUAK_evals[data_sample]['injected'][t]
                point[2] = QUAK_evals[data_sample]['glitches_new'][t]
                point[1] = QUAK_evals[data_sample]['bkg'][t]
                point[0] = QUAK_evals[data_sample]['bbh'][t]

                QUAK_space_samples[i, :] = point
                QUAK_samples[data_sample] = QUAK_space_samples      

    make_SN_QUAK_space = True
    if make_SN_QUAK_space:
        CPH = np.load(f"{savedir}/PLOTS/CPH.npy", allow_pickle=True)

        labels = CPH[0]
        CPH = CPH[1:]
        N = len(labels)
        N=4
        def make_QUAK_space_dist(quak_space_points, data_sample):
            N=4
            fig, axs = plt.subplots(N, N, figsize=(20, 20))
            for i in range(N):
                for j in range(i, N-1):
                    axs[i, j].axis('off')

            for i in range(N):
                axs[i, -1].axis("off")

            cmaps = ["Blues",
            "Purples",
            "Greens",
            "Reds",
            "Purples"]
            #if time_index != None:
            #    fig.suptitle(f"{data_class} QUAK animation \n Elapsed Time: {time_index:.0f} ms", fontsize = 35)
            for i in range(N):
                for j in range(i):
                        contour = True
                        for k, label in enumerate(labels):
                            for elem in CPH: #the worst possible way to do this, but I'm running out of time
                                i_, j_, k_, yy, xx, f = elem
                                if i_ == i and j_ == j and k_ == k:
                                    break
                                #print("I found the right set")
                            cset = axs[i, j].contour(yy, xx, f, cmap = cmaps[k])#, labels=labels)
                            axs[i, j].clabel(cset, inline=1, fontsize=10)
                            axs[i, j].set_xlim(0, 1.2)
                            axs[i, j].set_ylim(0, 1.2)
                            #axs[i, j].legend()

                            #plot the actual point
                            #AXIS HERE MAY BE FLIPPED CHECK THIS AFTERWARDS
                        #for quak_space_point in quak_space_points:
                        #    axs[i, j].scatter(quak_space_point[j], quak_space_point[i],c="black", s=50)

                        #make the actual distribution for QUAK space points, instead of scatterplotting
                        smooth = False
                        if smooth:
                            QSP = quak_space_points.T
                            A, B = QSP[:, i], QSP[:, j]
                            xx, yy, f = density_plot(A, B)
                            #cset = axs[i, j].contour(xx, yy, f, cmap='Blues', label=f"class {j}")
                            #cset = axs[i, j].contour(xx, yy, f, cmap = cmaps[k])
                            cset = axs[i, j].contour(yy, xx, f, cmap = "Greys")
                            axs[i, j].clabel(cset, inline=1, fontsize=10)
                            axs[i, j].set_xlim(0, 1.2)
                            axs[i, j].set_ylim(0, 1.2)
                        else:
                            for quak_space_point in quak_space_points:
                                axs[i, j].scatter(quak_space_point[j], quak_space_point[i],c="black", s=30)
                        

            for i in range(N):
                print("setting labels!")
                if labels[i] == "glitches_new":
                    LBL = "Glitches"
                elif labels[i] == "injected":
                    LBL = "Sine Gaussian"
                elif labels[i] == "bbh":
                    LBL = "BBH"
                elif labels[i] == "bkg":
                    LBL = "Background"
                else:
                    LBL = labels[i]
                print("setting label here")
                axs[i, 0].set_ylabel(LBL, fontsize=15)
                axs[-1, i].set_xlabel(LBL, fontsize=15)
            print("SAVED PLOT for ", iden, "GO CHECK IT!!")

            fig.savefig(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/QUAK_dist_{data_sample}.png", dpi=150)
            plt.close(fig)
            
        compiled = True

        #just doing both here
        if compiled:
            for data_sample in QUAK_samples:
                make_QUAK_space_dist(QUAK_samples[data_sample], data_sample)
        
        if compiled:
            all_samples = []
            for data_sample in QUAK_samples:
                all_samples.append(QUAK_samples[data_sample])
                print(QUAK_samples[data_sample].shape)

            #print("ALL SAMPLES", all_samples)
            all_samples = np.concatenate(all_samples)
            print(all_samples.shape)

            make_QUAK_space_dist(all_samples, "all")
            #assert 0
    QUAK_colormap = {"bbh" : "blue",
                    "bkg" : "purple",
                    "glitches_new": "green",
                    "injected" : "red"}  
    class_map = {"bbh" : "BBH",
                    "bkg" : "Background",
                    "glitches_new": "Glitches",
                    "injected" : "SG Injection"}               
    for data_class in QUAK_evals:
        toggle = True #working on the stuff below for now
        if toggle:
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            
                
            for QUAK_class in QUAK_evals[data_class]:
                LEN = len(QUAK_evals[data_class][QUAK_class])
                xs = np.arange(0, 1000/4096 * len(QUAK_evals[data_class][QUAK_class])*overlap,1000/4096 * overlap)
                #axs[0].plot(xs, QUAK_evals[data_class][QUAK_class], label = class_map[QUAK_class], c=QUAK_colormap[QUAK_class])
            
                smoothing=True
                if not smoothing:
                    axs[0].plot(xs, QUAK_evals[data_class][QUAK_class], label = class_map[QUAK_class], c=QUAK_colormap[QUAK_class])
                else:
                    KW = 25
                    xs = xs[KW-1:]
                    dseg = QUAK_evals[data_class][QUAK_class]   
                    kernel = np.ones((KW,))/KW
                    conv_dseg = np.convolve(dseg, kernel, mode = 'valid')
                    axs[0].plot(xs, conv_dseg, label = class_map[QUAK_class], c=QUAK_colormap[QUAK_class])

            np_save = True
            if np_save:
                build = np.zeros((4, LEN))
                for i, QUAK_class in enumerate(QUAK_evals[data_class]):
                    build[i, :] = QUAK_evals[data_class][QUAK_class]

            axs[0].legend()
            axs[0].set_xlabel("time, ms", fontsize=14)
            axs[0].set_ylabel("autoencoder recreation loss", fontsize=14)
            axs[0].set_title(f"procedural AE loss, {data_class} sample", fontsize=17)

            if 1:
                xs = np.arange(0, 1000/4096 * len(orig_data[data_class][0, :]),1000/4096)
                axs[1].plot(xs, orig_data[data_class][0, :], label = "original signal, H1")
                axs[1].plot(xs, orig_data[data_class][1, :], label = "original signal, L1")
                axs[1].legend()
                axs[1].set_xlabel("time, ms", fontsize=14)
                axs[1].set_ylabel("whitened strain", fontsize=14)
                axs[1].set_title("original signal", fontsize=17)
            plt.savefig(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/{data_class}_PROCEDURAL.png", dpi=300)
            print("SAVING, ", iden, data_class)
            np.save(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/{data_class}_np_data.np", build)
            plt.show()

        if do_KDE:
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            
            for KDE_class in KDE_evals[data_class]:
                LEN = len(KDE_evals[data_class][QUAK_class])
                xs = np.arange(0, 1000/4096 * len(KDE_evals[data_class][KDE_class])*overlap,1000/4096 * overlap)
                axs[0].plot(xs, KDE_evals[data_class][KDE_class], label = class_map[KDE_class], c=QUAK_colormap[KDE_class])
            
            np_save = True
            if np_save:
                build = np.zeros((4, LEN))
                for i, KDE_class in enumerate(KDE_evals[data_class]):
                    build[i, :] = KDE_evals[data_class][KDE_class]

            axs[0].legend()
            axs[0].set_xlabel("time, ms", fontsize=14)
            axs[0].set_ylabel("autoencoder recreation loss", fontsize=14)
            axs[0].set_title(f"procedural AE loss, {data_class} sample", fontsize=17)

            if 1:
                xs = np.arange(0, 1000/4096 * len(orig_data[data_class][0, :]),1000/4096)
                axs[1].plot(xs, orig_data[data_class][0, :], label = "original signal, H1")
                axs[1].plot(xs, orig_data[data_class][1, :], label = "original signal, L1")
                axs[1].legend()
                axs[1].set_xlabel("time, ms", fontsize=14)
                axs[1].set_ylabel("whitened strain", fontsize=14)
                axs[1].set_title("original signal", fontsize=17)

            plt.savefig(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/KDE_{data_class}_PROCEDURAL.png", dpi=300)
            print("SAVING, ", iden, data_class)
            np.save(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/KDE_{data_class}_np_data.np", build)
            plt.show()

        if NN_quak:
            NN_evals = NN_QUAK_evals_split
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            smoothing = True
            for NN_class in NN_evals[data_class]:
                LEN = len(NN_evals[data_class][NN_class])
                xs = np.arange(0, 1000/4096 * len(NN_evals[data_class][NN_class])*overlap,1000/4096 * overlap)
                if not smoothing:
                    axs[0].plot(xs, NN_evals[data_class][NN_class], label = class_map[NN_class], c=QUAK_colormap[NN_class])
                else:
                    KW = 25
                    xs = xs[KW-1:]
                    dseg = NN_evals[data_class][NN_class]
                    kernel = np.ones((KW,))/KW
                    conv_dseg = np.convolve(dseg, kernel, mode = 'valid')
                    axs[0].plot(xs, conv_dseg, label = class_map[NN_class], c=QUAK_colormap[NN_class])
            np_save = True
            if np_save:
                build = np.zeros((4, LEN))
                for i, NN_class in enumerate(NN_evals[data_class]):
                    build[i, :] = NN_evals[data_class][NN_class]

            axs[0].legend()
            axs[0].set_xlabel("time, ms", fontsize=14)
            axs[0].set_ylabel("NN softmax output", fontsize=14)
            axs[0].set_title(f"procedural NN output, {data_class} sample", fontsize=17)

            if 1:
                xs = np.arange(0, 1000/4096 * len(orig_data[data_class][0, :]),1000/4096)
                axs[1].plot(xs, orig_data[data_class][0, :], label = "original signal, H1")
                axs[1].plot(xs, orig_data[data_class][1, :], label = "original signal, L1")
                axs[1].legend()
                axs[1].set_xlabel("time, ms", fontsize=14)
                axs[1].set_ylabel("whitened strain", fontsize=14)
                axs[1].set_title("original signal", fontsize=17)

            plt.savefig(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/NN_QUAK_{data_class}_PROCEDURAL.png", dpi=300)
            print("SAVING, ", iden, data_class)
            np.save(f"{savedir}/PLOTS/RUNTHROUGH/{iden}/NN_QUAK_{data_class}_np_data.np", build)
            plt.show()
        
        #KDE space plots
        #note this is still under the above loop


        if 0: #disabling the animation thing for now, don't need it
            #do the plotting of the point moving around in QUAK space
            CPH = np.load(f"{savedir}/PLOTS/CPH.npy", allow_pickle=True)

            labels = CPH[0]
            CPH = CPH[1:]
            N = len(labels)
        

            def make_quakker(quak_space_point, save_index, time_index=None):
                fig, axs = plt.subplots(N, N, figsize=(20, 20))
                for i in range(N):
                    for j in range(i, N-1):
                        axs[i, j].axis('off')

                for i in range(N):
                    axs[i, -1].axis("off")

                cmaps = ["Blues",
                "Purples",
                "Greens",
                "Reds",
                "Purples"]
                if time_index != None:
                    fig.suptitle(f"{data_class} QUAK animation \n Elapsed Time: {time_index:.0f} ms", fontsize = 35)
                for i in range(N):
                    for j in range(i):
                            contour = True
                            for k, label in enumerate(labels):
                                for elem in CPH: #the worst possible way to do this, but I'm running out of time
                                    i_, j_, k_, yy, xx, f = elem
                                    if i_ == i and j_ == j and k_ == k:
                                        break
                                    #print("I found the right set")
                                cset = axs[i, j].contour(yy, xx, f, cmap = cmaps[k])#, labels=labels)
                                axs[i, j].clabel(cset, inline=1, fontsize=10)
                                axs[i, j].set_xlim(0, 1.2)
                                axs[i, j].set_ylim(0, 1.2)
                                #axs[i, j].legend()

                                #plot the actual point
                                #AXIS HERE MAY BE FLIPPED CHECK THIS AFTERWARDS
                            axs[i, j].scatter(quak_space_point[j], quak_space_point[i],c="black", s=100)
        

                for i in range(N):
                        if labels[i] == "glitches_new":
                            LBL = "Glitches"
                        elif labels[i] == "injected":
                            LBL = "SG Injection"
                        elif labels[i] == "bbh":
                            LBL = "BBH"
                        elif labels[i] == "bkg":
                            LBL = "Background"
                        else:
                            LBL = labels[i]
                        axs[i, 0].set_ylabel(LBL, fontsize=15)
                        axs[-1, i].set_xlabel(LBL, fontsize=15)

                fig.savefig(f"{savedir}/PLOTS/WIGGLE_2/{data_class}/img_{save_index}.png", dpi=150)
                plt.close(fig)
        # make_quakker(np.array([0.5, 0.5, 0.5, 0.5]))
            #now have to make the data into a set of these "quakker" points
            #I really need to fix these variable names later...

            #this is the stuff being plotted
            #for QUAK_class in QUAK_evals[data_class]:
            #    axs[0].plot(x_vals, QUAK_evals[data_class][QUAK_class], label = QUAK_class)
            #print("original ordering of classes", labels)
            #HAVE TO PERMUTE THE ORDERING HERE

            DAT = dict()
            for QUAK_class in QUAK_evals[data_class]: 
                #print("QUAK class", QUAK_class)
                #print("shape 221", QUAK_evals[data_class][QUAK_class].shape)
                DAT[QUAK_class] = QUAK_evals[data_class][QUAK_class]

            RT_all = np.stack([DAT[elem] for elem in labels]) # do things according to the initial ordering of the classes
            RT_all = RT_all.transpose()
            print("RT ALL", RT_all.shape)
            #just look at the pieces in the middle
            mid = len(RT_all)//2
            bar = 200
            slx = slice(mid-bar//2, mid + bar//2)
            time_indicies = np.arange(0, 1000/4096 * len(QUAK_evals[data_class][QUAK_class])*overlap,1000/4096 * overlap)
            RT_all = RT_all[slx]
            TI = time_indicies[slx]
            #now do the plotting
            #clear the pre-existing images
            #fig.savefig(f"{savedir}/PLOTS/WIGGLE/{data_class}/img_{save_index}.png", dpi=150)
            toggle2=True
            if toggle2:
                try:
                    os.makedirs(f"{savedir}/PLOTS/WIGGLE_2/{data_class}/")
                except FileExistsError:
                    None
                step = 5
                print("this many images:", len(RT_all)//step)
                for i in range(0, len(RT_all), step): #go 50 by 50 evaluations
                    make_quakker(RT_all[i], i//step, TI[i])