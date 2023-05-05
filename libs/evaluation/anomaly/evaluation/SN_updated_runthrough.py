import numpy as np
import matplotlib.pyplot as plt
from anomaly.evaluation import QUAK_predict, data_segment, smooth_data, pQUAK_build_model_from_save
import time
import os

def discriminator(QUAK, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([QUAK, pearson])
    return np.dot(datum, param_vec)

def pearson_vectorized(data):
    #X is of shape (N_samples, N_features, 2)
    X, Y = data[:, :, 0], -data[:, :, 1] #inverting livingston
    X = X - np.mean(X, axis=1)[:, np.newaxis]
    Y = Y - np.mean(Y, axis=1)[:, np.newaxis]

    return np.multiply(X, Y).sum(axis=1) / np.sqrt(np.multiply(X, X).sum(axis=1)*np.multiply(Y, Y).sum(axis=1) )

def samples_pearson(data):
    '''
    Input of size (N_samples, feature_length, 2)

    '''
    full_seg_len=data.shape[1]
    seg_len=100
    seg_step = 5

    centres = np.arange(seg_len//2, full_seg_len-seg_len//2-seg_step, seg_step)#.shape
    maxshift = int(10e-3*4096) 
    maxshift=10
    step = 2

    #cut by maxshift
    edge_cut = slice(maxshift//seg_step, -maxshift//seg_step) #return this at end
    centres = centres[edge_cut]
    ts = time.time()
    center_maxs = np.zeros((len(data), len(centres)))
    for shift_amount in np.arange(-maxshift, maxshift, step):
        shifted_data = np.empty((len(data), len(centres), 100, 2))
        for i, center in enumerate(centres):
            
            shifted_data[:, i, :, 0] = data[:, center-50:center+50, 0]
            shifted_data[:, i, :, 1] = data[:, center-50+shift_amount:center+50+shift_amount, 1]
            
        #compute the pearson correlation
        shifted_data = shifted_data.reshape(shifted_data.shape[0]*shifted_data.shape[1], 100, 2)
        
        pearson_vals = pearson_vectorized(shifted_data).reshape(len(data), len(centres))
        center_maxs = np.maximum(center_maxs, pearson_vals)#element wise maximum
            
    print(f"{time.time()-ts:.3f}")   

    return center_maxs, edge_cut

def main(data, name):
    weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    model_savedir = "/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/"
    if 0:
        scatter_x = np.load(f"{model_savedir}/PLOTS/scatter_x_BBH_WEIGHTS.npy")
        scatter_y = np.load(f"{model_savedir}/PLOTS/scatter_y_BBH_WEIGHTS.npy")

    #data = np.load(None)
    print("loaded data shape", data.shape)
    if len(data.shape) == 2:
        #since data sample
        data = data[:, np.newaxis, :]
        print("after adding axis", data.shape)
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)

    print("72 shape", data.shape)

    data_orig = data[:]
    data = data_segment(data_orig, 100, 5)
    print("74 SHAPE", data.shape)

    QUAK_vals = QUAK_predict(model_savedir, data)

    pearson_evals, edge_cut= samples_pearson(data_orig)
    QUAK_vals = QUAK_vals[:, edge_cut, :]
    print("155", pearson_evals.shape, QUAK_vals.shape)

    

    do_smooth=True
    if do_smooth:
        KERNEL_LEN = 20
        print("BEFORE SMOOTH", pearson_evals.shape, QUAK_vals.shape)
        pearson_evals = smooth_data(pearson_evals[:, :, np.newaxis], KERNEL_LEN)
        QUAK_vals = smooth_data(QUAK_vals, KERNEL_LEN)
        print("AFTER SMOOTH", pearson_evals.shape, QUAK_vals.shape)

    if 0:
        path__ = "/home/ryan.raikman/s22/anomaly/ES_savedir_15SNR/"
        try:
            os.makedirs(path__)
        except FileExistsError:
            None
        np.save(f"{path__}signal_QUAK.npy", QUAK_vals)
        np.save(f"{path__}signal_pearson.npy", pearson_evals)
        assert 0

    #do the plotting
    plot_savedir = "/home/ryan.raikman/s22/temp6/"
    try:
        os.makedirs(plot_savedir)
    except FileExistsError:
        None
    print("77", data_orig.shape)
    
    sample=0
    QUAK_samp = QUAK_vals[sample, :, :]
    pearson_samp = pearson_evals[sample, :]
    metric_values = discriminator(QUAK_samp, pearson_samp, weights)
    
    if 0:
        far_vals = []
        for elem in metric_values:
            #print("elem", elem)
            far_ind = np.searchsorted(scatter_x, elem)
            #print("far ind", far_ind)
            if far_ind == len(scatter_x):
                far_ind = len(scatter_x) - 1 #doesn't really matter at this point
            far_vals.append(scatter_y[far_ind])
        far_vals = np.array(far_vals)
    do_4_plot = False
    if not do_4_plot:
        if 1:
            min_metric_vals = []
            for sample in range(len(QUAK_vals)):
                #sample=0
                QUAK_samp = QUAK_vals[sample, :, :]
                pearson_samp = pearson_evals[sample, :]
                metric_values = discriminator(QUAK_samp, pearson_samp, weights)
                min_metric_vals.append(min(metric_values))
            return min_metric_vals

        else:
            return min(metric_values)
    if do_4_plot:
        

        #STRAIN PLOT
        plt.figure(figsize=(8, 5))
        ts = np.linspace(0, len(data_orig[sample, :, 0])/4096 * 1000, len(data_orig[sample, :, 0]))
        plt.plot(ts, data_orig[sample, :, 0], label = "Hanford", linewidth=1)
        plt.plot(ts, data_orig[sample, :, 1], label = "Livingston", linewidth=1)
        plt.legend()
        plt.xlabel("Time, ms")
        plt.ylabel("Strain")
        plt.title("Supernova GW signature")
        plt.grid()
        plt.savefig(f"{plot_savedir}/ex1_{name}.png", dpi=300)

        #STRAIN PLOT, windowed
        plt.figure(figsize=(8, 5))
        ts = np.linspace(0, len(data_orig[sample, :, 0])/4096 * 1000, len(data_orig[sample, :, 0]))
        plt.plot(ts, data_orig[sample, :, 0], label = "Hanford", linewidth=1)
        plt.plot(ts, data_orig[sample, :, 1], label = "Livingston", linewidth=1, alpha=0.8)
        plt.legend()
        plt.xlim(700, 1450)
        plt.xlabel("Time, ms")
        plt.ylabel("Strain")
        plt.title("Supernova GW signature")
        plt.grid()
        plt.savefig(f"{plot_savedir}/ex1.5_{name}.png", dpi=300)

        #QUAK EVAL + PEARSON PLOT
        #plt.figure(figsize=(8, 5))
        fig, ax = plt.subplots(figsize=(8, 5))
        
        QUAK_labels = ["BBH", "BKG", "GLITCH", "SG"]
        ts_QUAK = np.linspace(0, len(data_orig[sample, :, 0])/4096 * 1000, len(pearson_samp))
        for i in range(4):
            ax.plot(ts_QUAK, QUAK_samp[:, i], label = QUAK_labels[i])
        
        #plt.plot(ts_QUAK, pearson_samp, label = "PEARSON")
        ax.set_xlabel("Time, ms")
        ax.set_ylabel("Recreation loss")
        ax.set_title("Supernova QUAK projection")
        
        ax.tick_params(axis='y', labelcolor="black")

        ax2 = ax.twinx()
        ax2.plot(ts_QUAK, pearson_samp, label = "Pearson", c="purple", alpha=0.8)
        ax2.set_ylabel("Pearson Correlation")
        ax2.tick_params(axis="y", labelcolor="black")
        #ax2.legend()
        plt.grid()
        ax2.set_ylim(0.1, 0.8)
        ax.plot([1, 0], [0.15, 0.15], color="purple", alpha=1, label = "Pearson")
        ax.legend(loc="center right")
        #ax.set_xlim(700, 1450)

        plt.savefig(f"{plot_savedir}/ex2_{name}.png", dpi=300)

        #plotting final discrimating values
        
        #plt.figure(figsize=(8, 5))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ts_QUAK, metric_values, label = "metric value")
        ax.tick_params(axis='y', labelcolor="blue")
        ax.set_xlabel("Time, ms")
        ax.set_ylabel("Metric value")

        #ax.set_xlim(700, 1450)
        
        
        #ax.set_title("Metric evaluation")
        ax.legend(loc=(0.01, 0.09))

        

        ax2 = ax.twinx()
        ax2.plot(ts_QUAK, far_vals, c="orange", label = "FAR")
        ax2.set_yscale("log")
        ax2.set_ylabel("False Alarm Rate, Hz")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.legend()
        plt.savefig(f"{plot_savedir}/ex3_{name}.png", dpi=300)

        of_interest = []
        of_interest_pearson = []
        for i, elem in enumerate(far_vals):
            if elem < 1e-3:
                of_interest.append(QUAK_samp[i])
                of_interest_pearson.append(pearson_samp[i])

        #print("of interest", of_interest)
        of_interest = np.array(of_interest)
        of_interest_pearson = np.array(of_interest_pearson)
        count = len(os.listdir(f"{plot_savedir}/numpy/"))
        np.save(f"{plot_savedir}/numpy/element_{count}.npy", of_interest)
        np.save(f"{plot_savedir}/numpy/element_{count}_pearson.npy", of_interest_pearson)


if 1:
    #WNB eval

    path = "/home/ryan.raikman/s23/data/SN_strains_long/Powell_2021/"
    path = "/home/ryan.raikman/s23/data/WNB_strains/0.01_400_1000/"
    datae = []
    SNRs = []
    for file in os.listdir(path):
        data = np.load(f"{path}/{file}")
        SNR = float(file.split("_")[-1][:-4])
        print(data.shape)
        datae.append(data)
        SNRs.append(SNR)

    datae = np.hstack(datae)
    SNRs = np.vstack(SNRs)

    print("datae, SNRs", datae.shape, SNRs.shape)
    #assert 0
    min_vals = main(datae, "WNB")
    min_vals = np.array(min_vals)[:, np.newaxis]
    print("min vals", min_vals.shape, SNRs.shape)
    np.save("/home/ryan.raikman/s23/min_vals_WNB_0.01_400_1000.npy", np.hstack([min_vals, SNRs]))



file_SN = "/home/ryan.raikman/s22/generated_SNs_noise_3/Powell_2020/y20_phi6.283_theta0.680_16384Hz_SN_BP.npy"

if 0:
    path = "/home/ryan.raikman/s22/ES_files/signals_snr_15/"
    loaded = []
    for file in os.listdir(path):
        try:
            BBH = np.load(f"{path}/{file}/bbh_segs.npy")
            loaded.append(BBH)
        except: None
        try:
            SG = np.load(f"{path}/{file}/injected_segs.npy")
            loaded.append(SG)
        except: None
    full_data = np.hstack(loaded)
    print("SHAPE", full_data.shape)

    main(full_data, "SIGNALS")
    

if 0:
    #main(np.load("/home/ryan.raikman/s22/WNB_inject/WNB_SNR_16.1.npy"), "WNB")
    min_vals = []
    path_ = "/home/ryan.raikman/s22/more_WNB_injects2/480_1000/"
    for file in os.listdir(path_):
        data = np.load(f"{path_}/{file}")
        SNR = float(file.split("_")[-1][:-4])
        min_val = main(data, "WNB")
        print("XXXX file, SNR, min_val", file, SNR, min_val)
        min_vals.append([SNR, min_val])
        np.save("/home/ryan.raikman/s22/temp8/MV_SNR_480_1000.npy", np.array(min_vals))
if 0:
    for file in os.listdir("/home/ryan.raikman/s22/generated_SNs_noise_4/Powell_2020/"):
        main(np.load(f"/home/ryan.raikman/s22/generated_SNs_noise_4/Powell_2020/{file}"), "not_chosen")

if 0: main(np.load("/home/ryan.raikman/s22/generated_SNs_noise_4/Powell_2020/s18_phi5.585_theta0.000_16384Hz_SN_BP.npy"), "chosen")
if 0:
    for name in os.listdir("/home/ryan.raikman/s22/generated_SNs_noise_5/Powell_2020"):
        data = np.load(f"/home/ryan.raikman/s22/generated_SNs_noise_5/Powell_2020/{name}")
        main(data, name)