import time
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from labellines import labelLines
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper_functions import (
    stack_dict_into_numpy,
    stack_dict_into_numpy_segments,
    compute_fars,
    far_to_metric
    )
from final_metric_optimization import LinearModel

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
    HISTOGRAM_BIN_MIN,
    VARYING_SNR_LOW,
    VARYING_SNR_HIGH,
    GPU_NAME,
    RETURN_INDIV_LOSSES
)

DEVICE = torch.device(GPU_NAME)


model = LinearModel(21).to(DEVICE)
model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))
weight = (model.layer_normal.weight.data.cpu().numpy()[0])
bias = model.layer_normal.bias.data.cpu().numpy()[0]
print("bias!:", bias)
weights = []
for i in range(5):
    arr = np.zeros(weight.shape)
    arr[4*i] = weight[4*i]
    arr[4*i+1] = weight[4*i+1]
    arr[4*i+3] = weight[4*i+3]
    weights.append(arr)

#shared, original -> original coefficient
arr = np.zeros(weight.shape)
for i in range(5):
    arr[4*i+2] = weight[4*i+2]
weights.append(arr)

#pearson coefficient
arr = np.zeros(weight.shape)
arr[-1] = weight[-1]
weights.append(arr)



def engineered_features(data):
    #print(data[0, :10, :])
    newdata = np.zeros(data.shape)

    for i in range(4):
        a, b = data[:, :, 2*i], data[:, :, 2*i+1]
        newdata[:, :, 2*i] = (a+b)/2
        newdata[:, :, 2*i+1] = abs(a-b)# / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    #print(newdata[0, :10, :])
    #assert 0
    return newdata

def engineered_features_torch(data):
    #print(data[0, :10, :])
    newdata = torch.zeros(data.shape).to(DEVICE)

    for i in range(4):
        a, b = data[:, :, 2*i], data[:, :, 2*i+1]
        newdata[:, :, 2*i] = (a+b)/2
        newdata[:, :, 2*i+1] = abs(a-b)# / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    #print(newdata[0, :10, :])
    #assert 0
    return newdata

class LinearModel(nn.Module):
        def __init__(self, n_dims):
            super(LinearModel, self).__init__()
            self.layer = nn.Linear(21, 4)
            self.layer1_5 = nn.Linear(4, 2)
            self.layer2 = nn.Linear(2, 1)
            self.layer_normal = nn.Linear(21, 1)

        def forward(self, x):

            return self.layer_normal(x)

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

def snr_vs_far_plotting(datas, snrss, metric_coefs, far_hist, tags, savedir, special=None):
    fig, axs = plt.subplots(1, figsize=(12, 8))
    colors = {
        "bbh":"blue",
        "sg":"red",
        "sglf":"red",
        "sghf":"orange",
        "wnb": "darkviolet",
        'wnblf': "deeppink",
        "supernova": "goldenrod"
    }

    axs.set_xlabel(f"SNR", fontsize=20)
    axs.set_ylabel("Final metric value, a.u.", fontsize=20)
    #axs.grid()



    for k in range(len(datas)):
        data = datas[k]
        snrs = snrss[k]
        tag = tags[k]

        if RETURN_INDIV_LOSSES:
            fm_vals = metric_coefs(torch.from_numpy(data).float().to(DEVICE)).detach().cpu().numpy()
        else:
            fm_vals = np.dot(data, metric_coefs)

        fm_vals = np.min(fm_vals, axis=1)
        #far_vals = compute_fars(fm_vals, far_hist=far_hist)


        snr_plot, means_plot, stds_plot = calculate_means(fm_vals, snrs, bar=SNR_VS_FAR_BAR)
        means_plot, stds_plot = np.array(means_plot), np.array(stds_plot)
        #axs.errorbar(snr_plot, means_plot, yerr=stds_plot, xerr=SNR_VS_FAR_BAR//2, fmt="o", color=colors[tag], label = f"{tag}")
        rename_map = {
        "background":"Background",
        "bbh":"BBH",
        "glitch":"Glitch",
        "sglf":"SG 64-512 Hz",
        "sghf":"SG 512-1024 Hz",
        "wnblf": "WNB 40-400 Hz",
        "wnb": "WNB 400-1000 Hz",
        "supernova": "Supernova"
        }
        tag_ = rename_map[tag]
        print("tag", tag_)
        axs.plot(snr_plot, means_plot-bias, color=colors[tag], label = f"{tag_}", linewidth=2)
        axs.fill_between(snr_plot, means_plot-bias - stds_plot/2, means_plot-bias + stds_plot/2, alpha=0.15, color=colors[tag])

    for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
        metric_val_label = far_to_metric(SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
        if metric_val_label is not None:
            axs.axhline(y=metric_val_label-bias, alpha=0.8**i, label = f"1/{label}", c='black')

    labelLines(axs.get_lines(), zorder=2.5, xvals=(30, 30, 30, 50, 60, 50, 50, 57, 64, 71, 78))
    #axs.legend()
    #axs.grid()
    axs.set_title(special, fontsize=20)
    #axs.set_xscale("log")

    axs.set_xlim(VARYING_SNR_LOW+SNR_VS_FAR_BAR/2, VARYING_SNR_HIGH-SNR_VS_FAR_BAR/2)
    fig.tight_layout()
    plt.savefig(f'{savedir}/{special}.pdf', dpi=300)
    plt.show()
    plt.close()

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



    #axs[0].set_xlim(4.5e4, 5e4)
    #axs[1].set_xlim(0.9e4, 1e4)
    #axs[2].set_xlim(0.9e4, 1e4)

    colors = [
        'purple',
        'blue',
        'green',
        'red',
        'orange',
        'sienna',
        'black'
    ]
    labels = [
        "Background",
        "BBH",
        "Glitch",
        "SG 64-512 Hz",
        "SG 512-1024 Hz",
        "Freq domain corr.",
        "Pearson"
    ]



    #fm_vals = np.dot(data, metric_coefs)
    if RETURN_INDIV_LOSSES:
        fm_vals = metric_coefs(torch.from_numpy(data).float().to(DEVICE)).detach().cpu().numpy()
    else:
        fm_vals = np.dot(data, metric_coefs)
    far_vals = compute_fars(fm_vals, far_hist=far_hist)

    ts_farvals = np.linspace(0, 5/4096*len(far_vals), len(far_vals))
    axs[2].set_title("Final metric")
    axs[2].set_xlabel("Time (ms)")
    color = "black"
    axs[2].set_ylabel("Value, a.u.")
    axs[2].plot(ts_farvals*1000, fm_vals-bias, label = "metric value")
    axs[2].tick_params(axis='y', labelcolor=color)
    axs[2].legend()
    axs[2].set_ylim(-50, 10)
    if 0:
        #this is broken, just going to draw lines as with detection efficiency
        axs2_2 = axs[2].twinx()

        color2 = "orange"
        axs2_2.set_ylabel("False Alarm Rate")
        axs2_2.plot(ts_farvals*1000, far_vals, label = "FAR", color=color2)
        axs2_2.legend()
        axs2_2.tick_params(axis='y', labelcolor=color2)
        axs2_2.set_yscale("log")

    else:
        for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
            if i%2 == 0:
                metric_val_label = far_to_metric(SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
                if metric_val_label is not None:
                    axs[2].axhline(y=metric_val_label-bias, alpha=0.8**i, label = f"1/{label}", c='black')


    strain = strain[:, 100+3*5:-(100+4*5)]

    ts_strain = np.linspace(0, len(strain[0, :])/4096, len(strain[0, :]))
    axs[0].set_title(f"{tag} strain, SNR = {snr:.1f}")
    axs[0].plot(ts_strain*1000, strain[0, :], label = "Hanford")
    axs[0].plot(ts_strain*1000, strain[1, :], label = "Livingston")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Whitened strain")
    axs[0].legend()
    axs[0].grid()

    print("245", strain.shape)
    print("246", len(far_vals))

    for k in range(len(weights)):
        extracted = np.dot(data, weights[k])

        axs[1].plot(ts_farvals*1000, extracted, color=colors[k], label = labels[k])
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Contribution")
    axs[1].grid()
    axs[1].set_title("Per autoencoder final metric contribution + coherence features")
    axs[1].legend()

    xlims = {"bbh":(1550, 1550+300), "sg":(1550, 1550+300), "wnb":(2100, 2100+300), "wnblf":(2100, 2100+300), "supernova":(2000,2900)}
    for i in range(3):
        axs[i].set_xlim(xlims[tag])
    a, b = xlims[tag]
    c = b-a
    step = c/10

    labelLines(axs[2].get_lines(), zorder=2.5, xvals=(300, a+step*(1), a+step*(2), a+step*(3),))

    fig.tight_layout()
    axs[0].grid()
    axs[1].grid()
    for i in range(3):
        axs[i].set_xlim(xlims[tag])
    #axs[2].grid()
    plt.savefig(f"{plot_savedir}/{tag}_3_panel_plot.pdf", dpi=300)

def main(args):
    # temporary
    do_snr_vs_far = True
    do_fake_roc = True
    do_3_panel_plot = True

    #model = LinearModel(9).to(DEVICE)
    #model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))
    #final_values = model(final_values).detach()


    if do_snr_vs_far:
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        metric_coefs = np.load(f"{args.data_predicted_path}/trained/final_metric_params.npy")
        means, stds = np.load(f"{args.data_predicted_path}/trained/norm_factor_params.npy")
        tags = ['bbh', 'wnb', 'supernova', 'wnblf', 'sglf', 'sghf']
        if RETURN_INDIV_LOSSES:
            model = LinearModel(17).to(DEVICE)#
            model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))

        data_dict = {}
        snrs_dict = {}
        for tag in tags:
            mod = ""
            for elem in args.data_predicted_path.split("/")[:-2]:

                mod += elem + "/"
            print(f"loading {tag}")
            ts = time.time()
            data = np.load(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy")
            print(f"{tag} loaded in {time.time()-ts:.3f} seconds")
            #print(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy", data.shape)
            data = (data-means)/stds
            data = data[1000:]
            snrs = np.load(f"{mod}/data/{tag}_varying_snr_SNR.npy")[1000:]

            data_dict[tag] = data
            snrs_dict[tag] = snrs

        #if 0:
        # do one for the GWAK signal classes
        X1 = ["bbh", "sglf", "sghf"]
        snr_vs_far_plotting([data_dict[elem] for elem in X1], [snrs_dict[elem] for elem in X1], model, far_hist, X1, args.plot_savedir, special = "Known Signals Detection Efficiency")

        # and for the anomalous classes
        X2 = ["wnb", "supernova", "wnblf"]
        snr_vs_far_plotting([data_dict[elem] for elem in X2], [snrs_dict[elem] for elem in X2], model, far_hist, X2, args.plot_savedir, special = "Anomaly Detection Efficiency")

        #if 0:
        #everything in one plot
        X3 = ["bbh", "sglf", "sghf", "wnb", "supernova", "wnblf"]
        snr_vs_far_plotting([data_dict[elem] for elem in X3], [snrs_dict[elem] for elem in X3], model, far_hist, X3, args.plot_savedir, special = "Detection Efficiency")


            #final_values = model(final_values).detach()
            #    snr_vs_far_plotting(data[1000:], snrs[1000:], model, far_hist, tag, args.plot_savedir)
            #else:
            #    snr_vs_far_plotting(data[1000:], snrs[1000:], metric_coefs, far_hist, tag, args.plot_savedir)
            #

    if do_fake_roc:
        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        fake_roc_plotting(far_hist, args.plot_savedir)

    if do_3_panel_plot:

        far_hist = np.load(f"{args.data_predicted_path}/far_bins.npy")
        metric_coefs = np.load(f"{args.data_predicted_path}/trained/final_metric_params.npy")
        norm_factors = np.load(f"{args.data_predicted_path}/trained/norm_factor_params.npy")
        means, stds = norm_factors[0], norm_factors[1]

        tags = ['bbh', 'sg', 'wnb', 'supernova', 'wnblf']
        ind = 1
        for tag in tags:
            mod = ""
            for elem in args.data_predicted_path.split("/")[:-2]:
                mod += elem + "/"
            strains = np.load(f"{mod}/data/{tag}_varying_snr.npy", mmap_mode="r")[ind]
            data = np.load(f"{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy", mmap_mode="r")[ind]
            data = (data-means)/stds
            snrs = np.load(f"{mod}/data/{tag}_varying_snr_SNR.npy", mmap_mode="r")[ind]

            ##print("DATA, SNRS, strains", data.shape, snrs, strains.shape)
            #assert 0

            if RETURN_INDIV_LOSSES:
                model = LinearModel(17).to(DEVICE)#
                model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))
                #final_values = model(final_values).detach()
                #snr_vs_far_plotting(data[1000:], snrs[1000:], model, far_hist, tag, args.plot_savedir)
                three_panel_plotting(strains, data, snrs, model, far_hist, tag, args.plot_savedir)
            else:
                #snr_vs_far_plotting(data[1000:], snrs[1000:], metric_coefs, far_hist, tag, args.plot_savedir)
                three_panel_plotting(strains, data, snrs, metric_coefs, far_hist, tag, args.plot_savedir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_predicted_path', help='Path to model directory',
        type=str)

    parser.add_argument('plot_savedir', help='Required output directory for saving plots',
        type=str)

    parser.add_argument('fm_model_path', help='Path to the final model',
        type=str)

    # Additional arguments
    parser.add_argument('--class-labels', help='Labels for the QUAK axes',
        type=list[str], default=['bbh', 'sg', 'background', 'glitch'])
    args = parser.parse_args()
    main(args)