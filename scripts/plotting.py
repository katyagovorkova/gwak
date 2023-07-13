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


def engineered_features(data):

    newdata = np.zeros(data.shape)

    for i in range(4):
        a, b = data[:, :, 2*i], data[:, :, 2*i+1]
        newdata[:, :, 2*i] = (a+b)/2
        newdata[:, :, 2*i+1] = abs(a-b)# / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    return newdata

def engineered_features_torch(data):

    newdata = torch.zeros(data.shape).to(DEVICE)

    for i in range(4):
        a, b = data[:, :, 2*i], data[:, :, 2*i+1]
        newdata[:, :, 2*i] = (a+b)/2
        newdata[:, :, 2*i+1] = abs(a-b)# / (a+b + 0.01)

    newdata[:, :, -1] = data[:, :, -1]

    return newdata

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

def snr_vs_far_plotting(datas, snrss, metric_coefs, far_hist, tags, savedir, special, bias):
    fig, axs = plt.subplots(1, figsize=(12, 8))
    colors = {
        'bbh':'blue',
        'sg':'red',
        'sglf':'red',
        'sghf':'orange',
        'wnbhf': 'darkviolet',
        'wnblf': 'deeppink',
        'supernova': 'goldenrod'
    }

    axs.set_xlabel(f'SNR', fontsize=20)
    axs.set_ylabel('Final metric value, a.u.', fontsize=20)

    for k in range(len(datas)):
        data = datas[k]
        snrs = snrss[k]
        tag = tags[k]

        if RETURN_INDIV_LOSSES:
            fm_vals = metric_coefs(torch.from_numpy(data).float().to(DEVICE)).detach().cpu().numpy()
        else:
            fm_vals = np.dot(data, metric_coefs)

        fm_vals = np.min(fm_vals, axis=1)

        snr_plot, means_plot, stds_plot = calculate_means(fm_vals, snrs, bar=SNR_VS_FAR_BAR)
        means_plot, stds_plot = np.array(means_plot), np.array(stds_plot)
        rename_map = {
        'background':'Background',
        'bbh':'BBH',
        'glitch':'Glitch',
        'sglf':'SG 64-512 Hz',
        'sghf':'SG 512-1024 Hz',
        'wnblf': 'WNB 40-400 Hz',
        'wnbhf': 'WNB 400-1000 Hz',
        'supernova': 'Supernova'
        }
        tag_ = rename_map[tag]
        print('tag', tag_)
        axs.plot(snr_plot, means_plot-bias, color=colors[tag], label = f'{tag_}', linewidth=2)
        axs.fill_between(snr_plot, means_plot-bias - stds_plot/2, means_plot-bias + stds_plot/2, alpha=0.15, color=colors[tag])

    for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
        metric_val_label = far_to_metric(SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
        if metric_val_label is not None:
            axs.axhline(y=metric_val_label-bias, alpha=0.8**i, label = f'1/{label}', c='black')

    labelLines(axs.get_lines(), zorder=2.5, xvals=(30, 30, 30, 50, 60, 50, 50, 57, 64, 71, 78))
    axs.set_title(special, fontsize=20)

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
    plt.yscale('log')
    plt.xlabel('Metric value')
    plt.ylabel('Corresponding FAR, Hz')
    plt.xlim(-50, 50)

    plt.savefig(f'{savedir}/fake_roc.pdf', dpi=300)

def three_panel_plotting(strain, data, snr, metric_coefs, far_hist, tag, plot_savedir, bias, weights):
    # doing only one sample, for now
    print('Warning: three panel plot has incorrect x-axis, implement this!')
    fig, axs = plt.subplots(3, figsize=(8, 14))

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
        'Background',
        'BBH',
        'Glitch',
        'SG 64-512 Hz',
        'SG 512-1024 Hz',
        'Freq domain corr.',
        'Pearson'
    ]



    #fm_vals = np.dot(data, metric_coefs)
    if RETURN_INDIV_LOSSES:
        fm_vals = metric_coefs(torch.from_numpy(data).float().to(DEVICE)).detach().cpu().numpy()
    else:
        fm_vals = np.dot(data, metric_coefs)
    far_vals = compute_fars(fm_vals, far_hist=far_hist)

    ts_farvals = np.linspace(0, 5/4096*len(far_vals), len(far_vals))
    axs[2].set_title('Final metric')
    axs[2].set_xlabel('Time (ms)')
    color = 'black'
    axs[2].set_ylabel('Value, a.u.')
    axs[2].plot(ts_farvals*1000, fm_vals-bias, label = 'metric value')
    axs[2].tick_params(axis='y', labelcolor=color)
    axs[2].legend()
    axs[2].set_ylim(-50, 10)
    if 0:
        #this is broken, just going to draw lines as with detection efficiency
        axs2_2 = axs[2].twinx()

        color2 = 'orange'
        axs2_2.set_ylabel('False Alarm Rate')
        axs2_2.plot(ts_farvals*1000, far_vals, label = 'FAR', color=color2)
        axs2_2.legend()
        axs2_2.tick_params(axis='y', labelcolor=color2)
        axs2_2.set_yscale('log')

    else:
        for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
            if i%2 == 0:
                metric_val_label = far_to_metric(SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
                if metric_val_label is not None:
                    axs[2].axhline(y=metric_val_label-bias, alpha=0.8**i, label = f'1/{label}', c='black')


    strain = strain[:, 100+3*5:-(100+4*5)]

    ts_strain = np.linspace(0, len(strain[0, :])/4096, len(strain[0, :]))
    axs[0].set_title(f'{tag} strain, SNR = {snr:.1f}')
    axs[0].plot(ts_strain*1000, strain[0, :], label = 'Hanford')
    axs[0].plot(ts_strain*1000, strain[1, :], label = 'Livingston')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Whitened strain')
    axs[0].legend()
    axs[0].grid()

    print('245', strain.shape)
    print('246', len(far_vals))

    for k in range(len(weights)):
        extracted = np.dot(data, weights[k])

        axs[1].plot(ts_farvals*1000, extracted, color=colors[k], label = labels[k])
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Contribution')
    axs[1].grid()
    axs[1].set_title('Per autoencoder final metric contribution + coherence features')
    axs[1].legend()

    xlims = {
        'bbh':(1550, 1550+300),
        'sglf':(1550, 1550+300),
        'sghf':(1550, 1550+300),
        'wnbhf':(2100, 2100+300),
        'wnblf':(2100, 2100+300),
        'supernova':(2000, 2900)}

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
    plt.savefig(f'{plot_savedir}/{tag}_3_panel_plot.pdf', dpi=300)

def main(args):


    model = LinearModel(21).to(DEVICE)
    model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))
    weight = (model.layer.weight.data.cpu().numpy()[0])
    bias = model.layer.bias.data.cpu().numpy()[0]
    print('bias!:', bias)
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

    # temporary
    do_snr_vs_far = True
    do_fake_roc = True
    do_3_panel_plot = True

    if do_snr_vs_far:
        far_hist = np.load(f'{args.data_predicted_path}/far_bins.npy')
        metric_coefs = np.load(f'{args.data_predicted_path}/trained/final_metric_params.npy')
        means, stds = np.load(f'{args.data_predicted_path}/trained/norm_factor_params.npy')
        tags = ['bbh', 'wnbhf', 'supernova', 'wnblf', 'sglf', 'sghf']
        if RETURN_INDIV_LOSSES:
            model = LinearModel(21).to(DEVICE)#
            model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))

        data_dict = {}
        snrs_dict = {}
        for tag in tags:

            print(f'loading {tag}')
            ts = time.time()
            data = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy')

            print(f'{tag} loaded in {time.time()-ts:.3f} seconds')

            data = (data-means)/stds
            data = data[1000:]
            snrs = np.load(f'output/data/{tag}_varying_snr_SNR.npz.npy')[1000:]

            data_dict[tag] = data
            snrs_dict[tag] = snrs

        # do one for the GWAK signal classes
        X1 = ['bbh', 'sglf', 'sghf']
        snr_vs_far_plotting([data_dict[elem] for elem in X1],
            [snrs_dict[elem] for elem in X1],
            model,
            far_hist,
            X1,
            args.plot_savedir,
            'Known Signals Detection Efficiency',
            bias)

        # and for the anomalous classes
        X2 = ['wnbhf', 'supernova', 'wnblf']
        snr_vs_far_plotting([data_dict[elem] for elem in X2],
            [snrs_dict[elem] for elem in X2],
            model,
            far_hist,
            X2,
            args.plot_savedir,
            'Anomaly Detection Efficiency',
            bias)

        X3 = ['bbh', 'sglf', 'sghf', 'wnbhf', 'supernova', 'wnblf']
        snr_vs_far_plotting([data_dict[elem] for elem in X3],
            [snrs_dict[elem] for elem in X3],
            model,
            far_hist,
            X3,
            args.plot_savedir,
            'Detection Efficiency',
            bias)

    if do_fake_roc:
        far_hist = np.load(f'{args.data_predicted_path}/far_bins.npy')
        fake_roc_plotting(far_hist, args.plot_savedir)

    if do_3_panel_plot:

        far_hist = np.load(f'{args.data_predicted_path}/far_bins.npy')
        metric_coefs = np.load(f'{args.data_predicted_path}/trained/final_metric_params.npy')
        norm_factors = np.load(f'{args.data_predicted_path}/trained/norm_factor_params.npy')
        means, stds = norm_factors[0], norm_factors[1]

        tags = ['bbh', 'sghf', 'sglf', 'wnbhf', 'supernova', 'wnblf']
        ind = 1
        for tag in tags:
            strains = np.load(f'output/data/{tag}_varying_snr.npz')['data'][ind]
            data = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy')[ind]
            data = (data-means)/stds
            snrs = np.load(f'output/data/{tag}_varying_snr_SNR.npz.npy')[ind]

            if RETURN_INDIV_LOSSES:
                model = LinearModel(21).to(DEVICE)#
                model.load_state_dict(torch.load(args.fm_model_path, map_location=GPU_NAME))
                three_panel_plotting(strains, data, snrs, model, far_hist, tag, args.plot_savedir, bias, weights)
            else:
                three_panel_plotting(strains, data, snrs, metric_coefs, far_hist, tag, args.plot_savedir, bias, weights)


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