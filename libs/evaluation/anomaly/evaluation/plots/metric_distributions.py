import numpy as np
import matplotlib.pyplot as plt
from anomaly.evaluation import QUAK_predict, data_segment, smooth_data, pQUAK_build_model_from_save
import time
import os
import scipy.stats as st

def discriminator(QUAK, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([QUAK, pearson])
    return np.dot(datum, param_vec)

def discriminator_values(QUAK, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([QUAK, pearson])
    return np.multiply(datum, param_vec)

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

def main(data, name):
    weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    model_savedir = "/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/"
    scatter_x = np.load(f"{model_savedir}/PLOTS/scatter_x_BBH_WEIGHTS.npy")
    scatter_y = np.load(f"{model_savedir}/PLOTS/scatter_y_BBH_WEIGHTS.npy")

    print("DATA in shape", data.shape)
    if len(data.shape) == 2:
        #since data sample
        data = data[:, np.newaxis, :]

    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)

    print("DATA after mod", data.shape)
    #assert 0
    data_orig = data[:]
    data = data_segment(data_orig, 100, 5)

    QUAK_vals = QUAK_predict(model_savedir, data)

    pearson_evals, edge_cut= samples_pearson(data_orig)
    QUAK_vals = QUAK_vals[:, edge_cut, :]
    #print("155", pearson_evals.shape, QUAK_vals.shape)

    do_smooth=True
    if do_smooth:
        KERNEL_LEN = 20
        #print("BEFORE SMOOTH", pearson_evals.shape, QUAK_vals.shape)
        pearson_evals = smooth_data(pearson_evals[:, :, np.newaxis], KERNEL_LEN)
        QUAK_vals = smooth_data(QUAK_vals, KERNEL_LEN)
        #print("AFTER SMOOTH", pearson_evals.shape, QUAK_vals.shape)

    #sample=0
    
    min_vals = []
    SVs = []
    NVs = []
    for sample in range(len(QUAK_vals)):
        QUAK_samp = QUAK_vals[sample, :, :]
        pearson_samp = pearson_evals[sample, :]
        metric_values = discriminator(QUAK_samp, pearson_samp, weights)
        disc_vals = (discriminator_values(QUAK_samp, pearson_samp, weights))
        signal_vals = disc_vals[:, 0] + disc_vals[:, 3] + disc_vals[:, 4] 
        noise_vals = disc_vals[:, 1] + disc_vals[:, 2]
        #print(signal_vals.shape)
        point = np.argmin(metric_values)
        #print(point)
        #assert 0
        print("metric values min", np.amin(metric_values))
        min_vals.append(np.amin(metric_values))
        SVs.append(signal_vals[point])
        NVs.append(noise_vals[point])

    savepath = "/home/ryan.raikman/s22/temp9/"
    try:
        os.makedirs(savepath)
    except FileExistsError:
        None
    
    return min_vals, SVs, NVs

    
base = "/home/ryan.raikman/s22/training_files/4_24_loud2_datagen/1239134846_1239140924/"
min_val_dict = dict()
name_map = {"bbh_segs.npy":"BBH", 
            "bkg_segs.npy":"Background", 
            "glitch_segs.npy":"Glitches", 
            "injected_segs.npy":"Sine-Gaussian"}
min_bin = 100
max_bin = -100
XV_dict = dict()
for class_name in ["bbh_segs.npy", "bkg_segs.npy", "glitch_segs.npy", "injected_segs.npy"]:
    data = np.load(f"{base}/{class_name}")
    min_vals, SVs, NVs = main(data[:, :, :], None)
    min_, max_ = np.amin(min_vals), np.amax(min_vals)
    min_bin = min(min_bin, min_)
    max_bin = max(max_bin, max_)
    min_val_dict[name_map[class_name]] = min_vals
    XV_dict[name_map[class_name]] = [SVs, NVs]

#print("min_val_dict", min_val_dict)
cols = {"BBH":"blue","Background":"purple", "Glitches":"green", "Sine-Gaussian":"red"}

plt.figure(figsize=(8, 5))
for key in min_val_dict:
    plt.hist(min_val_dict[key], bins=20, range=(min_bin, max_bin), 
            label = key, density=True, alpha=0.5, color=cols[key])

plt.legend()
plt.xlabel("Metric value")
plt.ylabel("Density")
plt.title("Metric Distribution on Trained Classes")
plt.savefig(f"/home/ryan.raikman/s22/temp9/metric_dist.png", dpi=300)

cmaps = {"BBH":"Blues","Background":"Purples", "Glitches":"Greens", "Sine-Gaussian":"Reds"}
plt.figure(figsize=(8, 5))
for key in XV_dict:

    s, n = XV_dict[key]
    s, n = np.array(s), np.array(n)
    print("182", key, np.mean(s), np.mean(n), np.std(s), np.std(n))
    print("176", s.shape, n.shape)
    xx, yy, f = density_plot(s, n)
    plt.contour(xx, yy, f, cmap=cmaps[key])

plt.xlabel("Signal-like score")
plt.ylabel("Noise-like score")
plt.title("Metric decomposition")
plt.savefig(f"/home/ryan.raikman/s22/temp9/metric_decomp_V3.png", dpi=300)