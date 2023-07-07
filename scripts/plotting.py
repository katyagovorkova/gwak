import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper_functions import (
    stack_dict_into_numpy,
    stack_dict_into_numpy_segments,
    compute_fars,
    far_to_metric
    )
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
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
    plt.xlim(-50, 50)

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

    axs[0].set_xlim(4.5e4, 5e4)
    axs[1].set_xlim(0.9e4, 1e4)
    axs[2].set_xlim(0.9e4, 1e4)

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
        "glitches",
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
    do_snr_vs_far = True
    do_fake_roc = True
    do_3_panel_plot = True


    if do_snr_vs_far:
        # I will fix this later
        tags = ['bbh', 'sg', 'wnb', 'supernova']
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        metric_coefs = np.load(f"{args.data_predicted_path}/trained/final_metric_params.npy")
        norm_factors = np.load(f"{args.data_predicted_path}/trained/norm_factor_params.npy")
        means, stds = norm_factors[0], norm_factors[1]
        for tag in tags:
            mod = ""
            for elem in args.data_predicted_path.split("/")[:-2]:
                print(elem)
                mod += elem + "/"
            print("mod", mod)
            data = np.load(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy")
            data = (data-means)#/stds
            snrs = np.load(f"output/data/{tag}_varying_snr_SNR.npz.npy")

            snr_vs_far_plotting(data, snrs, metric_coefs, far_hist, tag, args.plot_savedir)

    if do_fake_roc:
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        fake_roc_plotting(far_hist, args.plot_savedir)

    if do_3_panel_plot:
        tags = ['bbh', 'sg', 'wnb', 'supernova']
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        metric_coefs = np.load(f"{args.data_predicted_path}/trained/final_metric_params.npy")
        norm_factors = np.load(f"{args.data_predicted_path}/trained/norm_factor_params.npy")
        means, stds = norm_factors[0], norm_factors[1]

        ind = 1
        for tag in tags:
            mod = ""
            for elem in args.data_predicted_path.split("/")[:-2]:
                mod += elem + "/"
            strains = np.load(f"output/data/{tag}_varying_snr.npz")['data'][ind]
            data = np.load(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy", mmap_mode="r")[ind]
            data = (data-means)#/stds
            snrs = np.load(f"output/data/{tag}_varying_snr_SNR.npz.npy", mmap_mode="r")[ind]

            three_panel_plotting(strains, data, snrs, metric_coefs, far_hist, tag, args.plot_savedir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_predicted_path', help='Path to model directory',
        type=str)
    # Required arguments
    parser.add_argument('plot_savedir', help='Required output directory for saving plots',
        type=str)

    # Additional arguments
    parser.add_argument('--class-labels', help='Labels for the QUAK axes',
        type=list[str], default=['bbh', 'sg', 'background', 'glitch'])
    args = parser.parse_args()
    main(args)