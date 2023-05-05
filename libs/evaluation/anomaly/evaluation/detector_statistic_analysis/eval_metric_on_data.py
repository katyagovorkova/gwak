import numpy as np
import matplotlib.pyplot as plt
import os
from anomaly.evaluation import QUAK_predict, data_segment, smooth_data, pQUAK_build_model_from_save
from scipy.stats import boxcox
import time
from scipy.stats import pearsonr

def discriminator(QUAK, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([QUAK, pearson])
    return np.dot(datum, param_vec)
    
def metric_katyav2(data):
    return 2* data[:, 0] - data[:, 1] - data[:, 2] + 2* data[:, 3]

def metric_katyav3(data):
    return 6* data[:, 0] - data[:, 1] - 3 * data[:, 2] + 6* data[:, 3]

def metric_v4(data):
    #BBH, BKG, GLITCH, SG = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    #np.log(np.exp(BBH) + np.exp(BKG) + np.exp(GLITCH) + np.exp(SG))
    centres = np.array([0.76, 0.73, 0.87, 0.76])
    data_cent = data - centres
    w = [11, -6, -11, 6] #weights
    return w[0] * data_cent[:, 0] + w[1] * data_cent[:, 1] + w[2] * data_cent[:, 2] + w[3] * data_cent[:, 3]

def metric_v5(data):
    def expabs(x):
        return np.exp(np.abs(x))
    centres = np.array([0.76, 0.73, 0.87, 0.76])
    data_cent = data - centres
    w = [1, 1, 1, 1] #weights

    D =  expabs(w[0] * data_cent[:, 0]) + \
    expabs(w[1] * data_cent[:, 1]) + \
    expabs(w[2] * data_cent[:, 2]) + \
    expabs(w[3] * data_cent[:, 3])

    return np.log(D)

def metric_v6(data, corrs):
    centres = np.array([0.76, 0.73, 0.87, 0.76])
    data_cent = data - centres
    w = [2, -1.5, -2.5, 2] #weights
    quak_part = w[0] * data_cent[:, 0] + w[1] * data_cent[:, 1] + w[2] * data_cent[:, 2] + w[3] * data_cent[:, 3]
    print("279", quak_part.shape, corrs.shape)
    return np.subtract(quak_part, 3*corrs[:, 0])

def samples_pearson_(data):
    '''
    Find maximum pearson correlation per sample
    in shape: (N_samples, 100, 2)
    '''
    print("into pearson", data.shape)
    assert data.shape[1] == 100 # if not, switch around
    step = 1
    maxshift = int(10e-3*4096)//5 #10 ms at 4096 Hz
    best_pearsons = np.zeros((len(data), 2*maxshift//step))
    for shift in np.arange(0, maxshift, step):
        data_H = data[:, shift:, 0]
        data_L = data[:, :100-shift, 1]
        for i in range(len(data)):
            best_pearsons[i, shift//step] = (pearsonr(data_H[i], -data_L[i])[0])
            
        #augment the other way
        data_H = data[:, :100-shift, 0]
        data_L = data[:, shift: , 1]
        for i in range(len(data)):
            best_pearsons[i, shift//step+maxshift//step] = (pearsonr(data_H[i], -data_L[i])[0])
        
    return np.amax(best_pearsons, axis=1)

def metric_nball(data, center, lmbdas):
    #find the distance from the center based on the boxcox transformation

    dists = np.zeros(len(data))
    for axis in range(4):
        xt = boxcox(data[:, axis], lmbdas[axis])
        dists += (xt-center[axis])**2
    return dists**0.5

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

def calculate_ROC(scatter_x, min_vals, scatter_y):
    TPRs = []
    order = "lessthan"
    for cutoff in scatter_x:
        #count the number of samples that pass through(FPR)
        if order == "lessthan":
            TPRs.append((min_vals<cutoff).sum()/len(min_vals))
        else:
            assert order == "greaterthan"
            TPRs.append((min_vals>cutoff).sum()/len(min_vals))
            
    TPRs=np.array(TPRs)
    FPRs = scatter_y

    return TPRs, FPRs

def main(signal_bank_path:str, model_savedir:str, iden:str, tag:str="BBH", data_manual=None):
    if data_manual is None:
        if tag == "BBH":
            data = np.load(signal_bank_path + "bbh_segs.npy")
            data_SNRS = np.load(signal_bank_path + "bbh_SNRS.npy")
        elif tag == "SG":
            data = np.load(signal_bank_path + "injected_segs.npy")
            data_SNRS = np.load(signal_bank_path + "SG_SNRS.npy")
    else:
        data, data_SNRS = data_manual
    #comes in as detectors, samples, datapoints
    print("loaded data shape", data.shape)
    if len(data.shape) == 2:
        #since data sample
        data = data[:, np.newaxis, :]
        print("after adding axis", data.shape)
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)
    #switch to samples, datapoints, detectors
    limit_data = False
    if limit_data:
        data_orig = data[:5] #cutting them down for now
        data_SNRS = data_SNRS[:5]
    else:
        data_orig = data[:]
    print("loaded signal shape,", data.shape)
    data = data_segment(data_orig, 100, 5)
    print("after slice,", data.shape)

    QUAK_vals = QUAK_predict(model_savedir, data)
    print("20, QUAK_vals shape", QUAK_vals.shape)
    #QUAK_vals = smooth_data(QUAK_vals, 50)
    print("22, after smoothing", QUAK_vals.shape)

    do_pearson=True
    if do_pearson:
        pearson_evals, edge_cut= samples_pearson(data_orig)
        QUAK_vals = QUAK_vals[:, edge_cut, :]
        print("155", pearson_evals.shape, QUAK_vals.shape)
        if 0:
            ts = time.time()
            pearson_evals = []
            for x in range(len(data)):
                pearson_evals.append(samples_pearson(data[x]))
            pearson_evals = np.vstack(pearson_evals)[:, :, np.newaxis]
            print("pearson shape before smoothing", pearson_evals.shape)
            pearson_evals = smooth_data(pearson_evals, 50)
            print(f"pearson stuff took {time.time()-ts:.3f} seconds")
            print("pearson evals final shape", pearson_evals.shape)
    min_vals = []
    do_nball = False

    do_smooth=True
    if do_smooth:
        KERNEL_LEN = 20
        print("BEFORE SMOOTH", pearson_evals.shape, QUAK_vals.shape)
        pearson_evals = smooth_data(pearson_evals[:, :, np.newaxis], KERNEL_LEN)
        QUAK_vals = smooth_data(QUAK_vals, KERNEL_LEN)
        print("AFTER SMOOTH", pearson_evals.shape, QUAK_vals.shape)

    if 0:
        #save the QUAK and pearson values
        try:
            os.makedirs(f"{model_savedir}/DATA/ES_TRAIN")
        except FileExistsError:
            None


        order = len(os.listdir(f"{model_savedir}/DATA/ES_TRAIN/"))
        np.save(f"{model_savedir}/DATA/ES_TRAIN/QUAK_vals_{tag}_{order}.npy", QUAK_vals)
        np.save(f"{model_savedir}/DATA/ES_TRAIN/pearson_vals_{tag}_{order}.npy", pearson_evals)
        return None
    #metric_select = pQUAK_build_model_from_save(model_savedir)
    weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    weights = np.array([0.48613072, -0.17497687, -0.01590199,  0.1017961,  -0.14085923])
    weights = np.array([-0.01666499,  0.29371879, -0.23844285, -0.01687215, -0.05543631])
    weights = np.array([-0.01488957,  0.30431612, -0.25321185, -0.01635141, -0.06630924])
    weights = np.array([-0.01666499,  0.29371879, -0.23844285, -0.01687215, -0.05543631])
    weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    metric_select = lambda QUAK_vals, pearson_vals : discriminator(QUAK_vals, pearson_vals, 
                                                    weights)
    order='lessthan'
    for i in range(len(QUAK_vals)):
        #sample = data[i]
        if do_nball:
            values = metric_nball(QUAK_vals[i], boxcox_center, lmbdas)
            min_val = np.max(values) #lol
        else:   
            print("188,", QUAK_vals[i].shape, pearson_evals[i].shape)
            values = metric_select(QUAK_vals[i], pearson_evals[i])
            min_val = np.min(values)
        #print("min val:", min_val)
        min_vals.append(min_val)
    
    min_vals = np.array(min_vals)
    np.save(f"{model_savedir}/PLOTS/min_vals_{iden}.npy", min_vals)

    #load the FPR vs metric data
    #scatter_x = np.load(f"{model_savedir}/PLOTS/scatter_x_FAR_OPT_BBH_SG_BIG.npy")
    #scatter_y = np.load(f"{model_savedir}/PLOTS/scatter_y_FAR_OPT_BBH_SG_BIG.npy")

    scatter_x = np.load(f"{model_savedir}/PLOTS/scatter_x_BBH_WEIGHTS.npy")
    scatter_y = np.load(f"{model_savedir}/PLOTS/scatter_y_BBH_WEIGHTS.npy")
    #print("X", scatter_x)
    #print("Y", scatter_y)

    make_4panel = True
    if make_4panel:
        #4 panel plot 
        fig, axs = plt.subplots(5, figsize=(8, 16))
        ind=1
        np.save(f"/home/ryan.raikman/s22/data_{tag}.npy", data_orig[ind, :, :])
        axs[0].plot(data_orig[ind, :, 0], label = "H1")
        axs[0].plot(data_orig[ind, :, 1], label = "L1")
        axs[0].legend()
        axs[0].set_title("Detector strain")

        axs[1].plot(QUAK_vals[ind, :, 0], label = "BBH")      
        axs[1].plot(QUAK_vals[ind, :, 1], label = "BKG")     
        axs[1].plot(QUAK_vals[ind, :, 2], label = "GLITCH")     
        axs[1].plot(QUAK_vals[ind, :, 3], label = "SG")     

        axs[1].legend()
        axs[1].set_title("Smoothed QUAK values")

        if do_nball:
            axs[2].plot(metric_nball(QUAK_vals[ind], boxcox_center, lmbdas))
        else:
            axs[2].plot(metric_select(QUAK_vals[ind], pearson_evals[ind]))
        axs[2].set_title("Metric values")


        #corresponding FAR values
        far_vals = []
        
        metric_values = metric_select(QUAK_vals[ind], pearson_evals[ind])
        for elem in metric_values:
            #print("elem", elem)
            far_ind = np.searchsorted(scatter_x, elem)
            #print("far ind", far_ind)
            if far_ind == len(scatter_x):
                far_ind = len(scatter_x) - 1 #doesn't really matter at this point
            far_vals.append(scatter_y[far_ind])
        far_vals = np.array(far_vals)

        axs[3].plot(far_vals)
        axs[3].set_yscale("log")
        axs[3].set_title("FAR values")
        #axs[3].set_yscale()

        axs[4].plot(pearson_evals[ind, :])
        axs[4].set_title("pearson values")
        axs[4].set_xlabel("time, datapoints")
        axs[4].set_ylabel("abs(pearson value)")

        fig.savefig(f"{model_savedir}/PLOTS/4panel_{iden}.png", dpi=300)
        None

    #ROC curve but for the SNRs individually

    SNR_bins = np.linspace(0, 100, 11)
    SNR_min_vals = []
    for i in range(len(SNR_bins)):
        SNR_min_vals.append([])
    #min_vals, data_SNRS

    try:
        os.makedirs(f"{model_savedir}/DATA/BBH_EVALS/")
    except FileExistsError:
        None

    saveidx = len(os.listdir(f"{model_savedir}/DATA/BBH_EVALS/"))
    np.save(f"{model_savedir}/DATA/BBH_EVALS/SNRS_{saveidx}.npy", data_SNRS)
    np.save(f"{model_savedir}/DATA/BBH_EVALS/MIN_VALS_{saveidx}.npy", min_vals)

    for i, elem in enumerate(data_SNRS):
        idx = np.searchsorted(SNR_bins, elem)
        if idx == len(SNR_bins):
            idx -= 1

        SNR_min_vals[idx].append(min_vals[i])

    plt.figure(figsize=(15, 10))
    for i, min_val_list in enumerate(SNR_min_vals):
        if i ==0 : continue
        if len(min_val_list)==0:continue
        TPR, FPR = calculate_ROC(scatter_x, np.array(min_val_list), scatter_y)

        plt.plot(FPR, TPR, label = f"SNR: {SNR_bins[i-1]:.1f} : {SNR_bins[i]:.1f}")

    plt.xscale("log")
    plt.legend()
    plt.xlabel("FAR, Hz")
    plt.ylabel("TPR")
    plt.title("ROC curve BBH signals")
    plt.savefig(f"{model_savedir}/PLOTS/roc_on_SNR_{iden}.png", dpi=300)



    #use the same cutoffs as scatter_x
    TPRs = []
    for cutoff in scatter_x:
        #count the number of samples that pass through(FPR)
        if order == "lessthan":
            TPRs.append((min_vals<cutoff).sum()/len(min_vals))
        else:
            assert order == "greaterthan"
            TPRs.append((min_vals>cutoff).sum()/len(min_vals))

    TPRs=np.array(TPRs)
    FPRs = scatter_y
    #make ROC curve

    plt.figure(figsize=(12, 7))
    plt.plot(FPRs, TPRs) #should be ordered the same way
    plt.xlabel("FAR, Hz")
    plt.ylabel("TPR")
    plt.xscale("log")
    plt.title("ROC curve, katya metric v2")
    plt.savefig(f"{model_savedir}/PLOTS/real_roc_{iden}.png", dpi=300)

    #make a plot showing the FAR statistics on a certain SNR bin

    #FPRs, data_SNRS
    #going to manually write histogram code
    minSNR = 0
    maxSNR = 100
    snrNUM = 50
    SNR_bins = np.linspace(minSNR, maxSNR, snrNUM)
    SNR_hist = []
    for elem in SNR_bins:
        SNR_hist.append([])
    #print("scatter_x", scatter_x)
    #print("SHAPES", scatter_x.shape, scatter_y.shape)
    for i, elem in enumerate(data_SNRS):
        #convert the element into an index based on SNR_bins
        #equivalent to search sorted, but this is faster
        ind = int( (elem - minSNR)/ (maxSNR-minSNR) * snrNUM )
        if ind >= len(SNR_bins): #everything that goes over get grouped into the last one
            ind = len(SNR_bins)-1
        FAR_ind = np.searchsorted(scatter_x, min_vals[i], side='right')
        #if FAR_ind == len()
        FAR_elem = scatter_y[FAR_ind]
        if FAR_elem == 0:
            FAR_elem = 1e-10
        SNR_hist[ind].append(np.log10(FAR_elem))

    FAR_means = np.zeros(snrNUM)
    FAR_stds = np.zeros(snrNUM)
    FAR_nums = np.zeros(snrNUM)
    slice_array = np.empty(snrNUM).astype("bool")
    #now reduce do a mean and std value
    for i in range(snrNUM):
        elements = np.array(SNR_hist[i])
        if len(elements) == 0:
            slice_array[i] = False
        else:
            FAR_means[i] = elements.mean()
            FAR_stds[i] = elements.std()
            slice_array[i] = True
        FAR_nums[i] = len(elements)
    #plotting
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].errorbar(SNR_bins[slice_array], FAR_means[slice_array], FAR_stds[slice_array], fmt='o')
    axs[0].set_xlabel("SNR value")
    axs[0].set_ylabel("Mean log(FAR)")
    axs[0].set_title("SNR vs. recovery")
    #axs[0].set_yscale("log")

    axs[1].scatter(SNR_bins[slice_array], FAR_nums[slice_array])
    axs[1].set_xlabel("SNR value")
    axs[1].set_ylabel("Number of events")
    axs[1].set_title("Events per SNR histogram")

    fig.savefig(f"{model_savedir}/PLOTS/FAR_vs_SNR_{iden}.png", dpi=300)




model_savedir = "/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/"
model_savedir = "/home/ryan.raikman/s22/anomaly/march23_nets/double_glitch/"
data_path = "/home/ryan.raikman/s22/anomaly/generated_data_2_1/1238561172_1238591457/bbh_segs.npy"
data_path ="/home/ryan.raikman/s22/anomaly/2_25_datagen/1252619166_1252651566/bbh_segs.npy"
#data_path = "/home/ryan.raikman/s22/BNS_waveforms/0_BNS.npy"
#data_path = "/home/ryan.raikman/s22/anomaly/2_26_datagen/1239038789_1239039812/bkg_segs.npy"
data_path = "/home/ryan.raikman/s22/anomaly/3_5_further_bp/1250152305_1250175764/"

data_path = "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix/1245180365_1245182425/"
data_path = "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix/1246079821_1246084359/"
#data_path + "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix/1249871825_1249881718/"


data_path = "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix_v2/1239449412_1239458462/"
data_path = "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix_v2/1242302779_1242314885/"
data_path = "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix_v2/1244698008_1244700937/"
data_path = "/home/ryan.raikman/s22/anomaly/3_24_bbh_fix_v2/1250447698_1250453546/"
data_path = "/home/ryan.raikman/s22/training_files/3_27_datagen_bbh_500_further/1238561172_1238591457/"
#data_path = "/home/ryan.raikman/s22/training_files/3_26_datagen_bbh_500/1238561172_1238591457/"
model_savedir = "/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/"
data_paths = ["/home/ryan.raikman/s22/training_files/3_31_datagen/1239449412_1239458462/", 
            "/home/ryan.raikman/s22/training_files/3_31_datagen/1240335696_1240342595/", 
            "/home/ryan.raikman/s22/training_files/3_31_datagen/1245180365_1245182425/",    
            "/home/ryan.raikman/s22/training_files/3_31_datagen/1245180365_1245182425/",
            "/home/ryan.raikman/s22/training_files/3_31_datagen/1252150173_1252152348/"]

#manually load all samples
loaded_datae = []
loaded_SNRS = []
for path in data_paths:
    loaded_datae.append(np.load(f"{path}bbh_segs.npy"))
    loaded_SNRS.append(np.load(f"{path}bbh_SNRS.npy"))
    #print(np.load(f"{path}bbh_segs.npy").shape)
    #print(np.load(f"{path}bbh_SNRS.npy").shape)
loaded_datae = np.hstack(loaded_datae)
loaded_SNRS = np.hstack(loaded_SNRS)

main(data_path, model_savedir, "bbh_4_13", "BBH", (loaded_datae, loaded_SNRS))


if 0:
    #for data_path in data_paths:
    data_path = data_paths[4]
    print("working on data_path", data_path)
    main(data_path, model_savedir, "bbh_500", "BBH")
    print("finished BBH")
    #main(data_path, model_savedir, "SG_500", "SG")
    print("finished SG")