import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


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
    do_cph:bool=True):

    # corner plot, BIL, LAL
    N = len(labels)
    fig, axs = plt.subplots(N, N, figsize=(20, 20))
    oneD_hist_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    # hide all of the ones not used
    for i in range(N):
        for j in range(i+1, N):
            axs[i, j].axis('off')

    cmaps = [
        'Blues',
        'Purples',
        'Greens',
        'Reds',
        'Purples']

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
            axs[i, i].hist(class_data[:, i], **oneD_hist_kwargs, label = LBL)
            np.save(f'{plot_savedir}/one_d_hist_{i}_{j}.npy', class_data[:, i])
            if loglog:
                axs[i, i].loglog()
            if enforce_lim:
                axs[i, i].set_xlim(0, 1.2)

    log_scaling = False

    corner_plot_hist = [labels]
    # do 2-d plots
    for i in range(N):
        for j in range(i):
            norm_factor_i = 0
            norm_factor_j = 0
            for k, class_data in enumerate(data): # find the additive normalizing factor, has to the same by class
                norm_factor_i = min(norm_factor_i, class_data[:, i].min())
                norm_factor_j = min(norm_factor_j, class_data[:, j].min())

                if log_scaling: # modify the data being plotted
                    epsilon = 1 # so we can take log10 of min value
                    A = class_data[:, i] - norm_factor_i + epsilon
                    A = np.log10(A)
                    B = class_data[:, j] - norm_factor_j + epsilon
                    B = np.log10(B)
                else:
                    A, B = class_data[:, i], class_data[:, j]


                if contour:
                    xx, yy, f = density_plot(A, B)
                    cset = axs[i, j].contour(yy, xx, f, cmap = cmaps[k])
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

    fig.savefig(plot_savedir+'/quak_plot.pdf')

    # save the corner plot hist
    corner_plot_hist = np.array(corner_plot_hist, dtype='object')
    if do_cph:
        np.save(f'{plot_savedir}/cph.npy', corner_plot_hist)


def main(args):
    corner_plot_data = []
    for i in range(len(args.class_labels)):
        class_data = []
        for j in range(len(args.class_labels)):
        # class data is the loss across all autoencoders for a given dataclass
            data = np.load(f'{args.data_predicted_path}/model_{args.class_labels[j]}/{args.class_labels[i]}.npy')
            class_data.append(data)
        class_data = np.stack(class_data, axis=1)
        corner_plot_data.append(class_data)

    corner_plotting(corner_plot_data, args.class_labels, args.plot_savedir)


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