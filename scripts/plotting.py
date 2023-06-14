import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper_functions import stack_dict_into_numpy, stack_dict_into_numpy_segments, compute_fars, far_to_metric
from config import (
    SEG_NUM_TIMESTEPS,
    SAMPLE_RATE,
    CLASS_ORDER,
    SPEED,
    NUM_IFOS,
    IFO_LABELS,
    RECREATION_WIDTH,
    RECREATION_HEIGHT_PER_SAMPLE,
    RECREATION_SAMPLES_PER_PLOT,
    SNR_VS_FAR_BAR,
    SNR_VS_FAR_HORIZONTAL_LINES,
    SNR_VS_FAR_HL_LABELS,
    SEGMENT_OVERLAP,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN
)

def calculate_means(metric_vals, snrs, bar):
    # helper function for SNR vs FAR plot
    means, stds = [], []
    snr_plot = []

    for i in range(10, 100, bar):
      
        points = []
        for shift in range(bar):
            for elem in np.where(((snrs-shift).astype(int))==i)[0]:
                points.append(elem)
        if len(points) == 0:
            continue
            
        snr_plot.append(i+bar/2)
        MV = []
        for point in points:
            MV.append(metric_vals[point])
        MV = np.array(MV)
        means.append(np.mean(MV))
        stds.append(np.std(MV))

    return snr_plot, means, stds

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

    return xx, yy, f


def corner_plotting(
    data:list[np.ndarray],
    labels:list[str],
    plot_savedir:str,
    enforce_lim:bool=True,
    contour:bool=True,
    loglog:bool=False,
    do_cph:bool=False,
    save_1d_hist:bool=False):

    # corner plot, BIL, LAL
    N = len(labels)
    fig, axs = plt.subplots(N, N, figsize=(20, 20))
    oneD_hist_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    # hide all of the ones not used
    for i in range(N):
        for j in range(i+1, N):
            axs[i, j].axis('off')

    cmaps = [
        'Purples',
        'Blues',
        'Greens',
        'Reds',
        'Purples']

    one_D_colors = [
        'purple', 
        'blue',
        'green',
        'red'
    ]

    # do the 1-d plots
    for i in range(N):
        norm_factor = 0
        for j, class_data in enumerate(data):
            norm_factor = min(norm_factor, class_data[:, i].min())

            if labels[j] == 'glitch':
                LBL = 'Glitches'
            elif labels[j] == 'sg':
                LBL = 'SG Injection'
            elif labels[j] == 'bbh':
                LBL = 'BBH'
            elif labels[j] == 'background':
                LBL = 'Background'
            else:
                LBL = labels[i]
            axs[i, i].hist(class_data[:, i], color=one_D_colors[j], **oneD_hist_kwargs, label = LBL)
            if save_1d_hist:
                np.save(f'{plot_savedir}/one_d_hist_{i}_{j}.npy', class_data[:, i])
            if enforce_lim:
                axs[i, i].set_xlim(0, 1.2)

    log_scaling = False

    corner_plot_hist = [labels]
    # do 2-d plots
    for i in range(N):
        for j in range(i):
            for k, class_data in enumerate(data):

                A, B = class_data[:, i], class_data[:, j]

                if contour:
                    xx, yy, f = density_plot(A, B)
                    cset = axs[i, j].contour(yy, xx, f, cmap=cmaps[k])
                    axs[i, j].clabel(cset, inline=1, fontsize=10)
                    if enforce_lim:
                        axs[i, j].set_xlim(0, 1.2)
                        axs[i, j].set_ylim(0, 1.2)
                    if loglog:
                        axs[i, j].loglog()
                    # save these values somehow
                    corner_plot_hist.append([i, j, k, yy, xx, f])


                else:
                    axs[i, j].scatter(B, A, s=15, c=cmaps[k][:-1])
                    if enforce_lim:
                        axs[i, j].set_xlim(0, 1.2)
                        axs[i, j].set_ylim(0, 1.2)
                    if loglog:
                        axs[i, j].loglog()


    # axis labels
    for i in range(N):
        if labels[i] == 'glitch':
            lbl = 'Glitches'
        elif labels[i] == 'sg':
            lbl = 'SG Injection'
        elif labels[i] == 'bbh':
            lbl = 'BBH'
        elif labels[i] == 'background':
            lbl = 'Background'
        else:
            lbl = labels[i]
        axs[i, 0].set_ylabel(lbl, fontsize=15)
        axs[-1, i].set_xlabel(lbl, fontsize=15)

    fig.legend()
    fig.savefig(plot_savedir+'/quak_plot.pdf')

    # save the corner plot hist
    corner_plot_hist = np.array(corner_plot_hist, dtype='object')
    if do_cph:
        np.save(f'{plot_savedir}/cph.npy', corner_plot_hist)

def recreation_plotting(data_original, data_recreated, savedir):
    ts = np.linspace(0, 1000*SEG_NUM_TIMESTEPS/SAMPLE_RATE, SEG_NUM_TIMESTEPS)
    colors = [
        'purple', 
        'blue',
        'green',
        'red'
    ]
    for i, class_name in enumerate(CLASS_ORDER):
        try:
            os.makedirs(f"{savedir}/recreation/{CLASS_ORDER[i]}/")
        except FileExistsError:
            None
        orig_samps = data_original[i][:RECREATION_SAMPLES_PER_PLOT, i, :, :]
        recreated_samps = data_recreated[i][:RECREATION_SAMPLES_PER_PLOT, :, :, :]

        # make the plot showing only original, recreated for that class
        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT*RECREATION_HEIGHT_PER_SAMPLE))
        
        for j in range(RECREATION_SAMPLES_PER_PLOT):
            for k in range(NUM_IFOS):
                axs[j, k].plot(ts, orig_samps[j, k, :], label = "Original", c='black')
                axs[j, k].plot(ts, recreated_samps[j, i, k, :], label = f"Recreated, {class_name}", c=colors[i])

                axs[j, k].grid()
                axs[j, k].set_title(IFO_LABELS[k])
                axs[j, k].legend()
                if k ==0:
                    axs[j, k].set_ylabel("Whitened Strain")
                axs[j, k].set_xlabel("Time (ms)")
                
        plt.tight_layout()
        fig.savefig(f"{savedir}/recreation/{CLASS_ORDER[i]}/one_to_one.pdf", dpi=300)

        # make the plot showing original, recreated for all classes
        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT*RECREATION_HEIGHT_PER_SAMPLE))
        
        for j in range(RECREATION_SAMPLES_PER_PLOT):
            for k in range(NUM_IFOS):
                axs[j, k].plot(ts, orig_samps[j, k, :], label = "Original", c='black')
                for l in range(len(CLASS_ORDER)):
                    axs[j, k].plot(ts, recreated_samps[j, l, k, :], label = f"Recreated, {CLASS_ORDER[l]}", c=colors[l])

                axs[j, k].grid()
                axs[j, k].set_title(IFO_LABELS[k])
                axs[j, k].legend()
                if k ==0:
                    axs[j, k].set_ylabel("Whitened Strain")
                axs[j, k].set_xlabel("Time (ms)")
                
        plt.tight_layout()
        fig.savefig(f"{savedir}/recreation/{CLASS_ORDER[i]}/one_to_all.pdf", dpi=300)

    # one general plot showcasing a sample from each class 
    fig, axs = plt.subplots(4, 2, figsize=(RECREATION_WIDTH, 4*RECREATION_HEIGHT_PER_SAMPLE))

    chosen_index = 0
    for l in range(len(CLASS_ORDER)):
        for k in range(NUM_IFOS):
            axs[l, k].plot(ts, data_original[l][chosen_index, l, k, :], label = "Original", c='black')
            for m in range(len(CLASS_ORDER)):
                axs[l, k].plot(ts, data_recreated[l][chosen_index, m, k, :], label = f"Recreated, {CLASS_ORDER[m]}", c=colors[m])
            
            axs[l, k].set_title(f"{IFO_LABELS[k]}, {CLASS_ORDER[l]}")
            axs[l, k].grid()
            axs[l, k].legend()

    plt.tight_layout()
    fig.savefig(f"{savedir}/general_recreation.pdf", dpi=300)
    
def snr_vs_far_plotting(data, snrs, metric_coefs, far_hist, tag, savedir):
    fm_vals = np.dot(data, metric_coefs)
    fm_vals = np.min(fm_vals, axis=1)
    far_vals = compute_fars(fm_vals, far_hist=far_hist)
    print("far_vals", far_vals)
    snr_plot, means_plot, stds_plot = calculate_means(fm_vals, snrs, bar=SNR_VS_FAR_BAR)

    plt.figure(figsize=(12, 8))
    plt.xlabel(f"{tag} SNR")
    plt.ylabel("Minimum metric value")
    #plt.ylim(-0.5, 0.5) 
    #plt.xlim(0,110) 
    plt.grid()
    plt.errorbar(snr_plot, means_plot, yerr=stds_plot, xerr=SNR_VS_FAR_BAR//2, fmt="o", color='goldenrod', label = f"{tag}")

    for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
        metric_val_label = far_to_metric(SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
        if metric_val_label is not None:
            plt.axhline(y=metric_val_label, alpha=0.8**i, label = f"1/{label}", c='black')

    plt.legend()

    plt.savefig(f'{savedir}/snr_vs_far_{tag}.pdf', dpi=300)
    plt.show()

def fake_roc_plotting(far_hist, savedir):
    datapoint_to_seconds = SEGMENT_OVERLAP / SAMPLE_RATE
    total_datapoints = far_hist.sum()
    total_seconds = total_datapoints * datapoint_to_seconds
    x_plot = []
    y_plot = []
    for i in range(len(far_hist)):
        total_below = np.sum(far_hist[:i])
        x_plot.append(i*HISTOGRAM_BIN_DIVISION-HISTOGRAM_BIN_MIN)
        y_plot.append(total_below/total_seconds)

    plt.figure()
    plt.plot(x_plot, y_plot)
    plt.yscale("log")
    plt.xlabel("Metric value")
    plt.ylabel("Corresponding FAR, Hz")

    plt.savefig(f'{savedir}/fake_roc.pdf', dpi=300)
        
def three_panel_plotting(strain, data, snr, metric_coefs, far_hist, tag, plot_savedir):
    # doing only one sample, for now
    print("Warning: three panel plot has incorrect x-axis, implement this!")
    fig, axs = plt.subplots(3, figsize=(8, 14))

    axs[0].set_title(f"Strain, SNR: {snr}")
    axs[0].plot(strain[0, :], label = "Hanford")
    axs[0].plot(strain[1, :], label = "Livingston")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Whitened strain")
    axs[0].legend()
    axs[0].grid()

    colors = [
        'purple', 
        'blue',
        'green',
        'red',
        'black'
    ]
    labels = [
        "background",
        "bbh",
        "glitch",
        "sine-gaussian",
        "pearson"
    ]

    for i in range(5):
        axs[1].plot(data[:, i], color=colors[i], label = labels[i])
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Significance (sigma)")
    axs[1].grid()
    axs[1].set_title("QUAK + Pearson significances")
    axs[1].legend()

    fm_vals = np.dot(data, metric_coefs)
    far_vals = compute_fars(fm_vals, far_hist=far_hist)

    axs[2].set_title("Metric value and corresponding FAR")
    axs[2].set_xlabel("Time (ms)")
    color = "blue"
    axs[2].set_ylabel("Metric value")
    axs[2].plot(fm_vals, label = "metric value")
    axs[2].tick_params(axis='y', labelcolor=color)
    axs[2].legend()

    axs2_2 = axs[2].twinx()

    color2 = "orange"
    axs2_2.set_ylabel("False Alarm Rate")
    axs2_2.plot(far_vals, label = "FAR", color=color2)
    axs2_2.legend()
    axs2_2.tick_params(axis='y', labelcolor=color2)
    axs2_2.set_yscale("log")

    fig.tight_layout()
    plt.savefig(f"{plot_savedir}/{tag}_3_panel_plot.pdf", dpi=300)
    
    
def main(args):
    # temporary
    do_corner = True
    do_recreation = True
    do_snr_vs_far = True
    do_fake_roc = True
    do_3_panel_plot = True

    if do_corner:
        corner_plot_data = [0] * 4 #4, N_samples, 4
        for i in range(len(args.class_labels)):
            class_label = args.class_labels[i] 
            class_index = CLASS_ORDER.index(class_label)
            data_dict = np.load(f'{args.data_predicted_path}/evaluated/quak_{class_label}.npz', allow_pickle=True)
            stacked_data = stack_dict_into_numpy(data_dict['loss'].flatten()[0])
            if SPEED:
                stacked_data = stacked_data[:500, :]

            corner_plot_data[class_index] = stacked_data

        corner_plotting(corner_plot_data, CLASS_ORDER, args.plot_savedir)

    if do_recreation:
        data_original = [0] * 4
        data_recreated = [0] * 4 
        for i in range(len(args.class_labels)):
            class_label = args.class_labels[i] 
            class_index = CLASS_ORDER.index(class_label)
            data_dict = np.load(f'{args.data_predicted_path}/evaluated/quak_{class_label}.npz', allow_pickle=True)
            stacked_data_original = stack_dict_into_numpy_segments(data_dict['original'].flatten()[0])
            stacked_data_recreated = stack_dict_into_numpy_segments(data_dict['recreated'].flatten()[0])

            data_original[class_index] = stacked_data_original
            data_recreated[class_index] = stacked_data_recreated

        recreation_plotting(data_original, data_recreated, args.plot_savedir)

    if do_snr_vs_far:
        # I will fix this later
        tags = ['bbh', 'sg']
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        metric_coefs = np.load(f"{args.data_predicted_path}/trained/final_metric_params.npy")
        for tag in tags:
            data = np.load(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy")
            snrs = np.load(f"{args.data_predicted_path}/generated/{tag}_varying_snr_injections_SNR.npy")

            snr_vs_far_plotting(data, snrs, metric_coefs, far_hist, tag, args.plot_savedir)

    if do_fake_roc:
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        fake_roc_plotting(far_hist, args.plot_savedir)

    if do_3_panel_plot:
        tags = ['bbh', 'sg']
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        metric_coefs = np.load(f"{args.data_predicted_path}/trained/final_metric_params.npy")

        ind = 0
        for tag in tags:
            strains = np.load(f"{args.data_predicted_path}/generated/{tag}_varying_snr_injections.npy", mmap_mode="r")[ind]
            data = np.load(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy", mmap_mode="r")[ind]
            snrs = np.load(f"{args.data_predicted_path}/generated/{tag}_varying_snr_injections_SNR.npy", mmap_mode="r")[ind]

            three_panel_plotting(strains, data, snrs, metric_coefs, far_hist, tag, args.plot_savedir)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_predicted_path', help='Required output directory of QUAK evaluations on data',
        type=str)
    parser.add_argument('plot_savedir', help='Required output directory for saving plots',
        type=str)

    # Additional arguments
    parser.add_argument('--class-labels', help='Labels for the QUAK axes',
        type=list[str], default=['bbh', 'sg', 'background', 'glitch'])
    args = parser.parse_args()
    main(args)