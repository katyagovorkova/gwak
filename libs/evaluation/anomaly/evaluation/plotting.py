import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def scale_norm(scale):
    return 6/ (1 + np.exp(-(scale-1)))

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

def corner_plotting(data:list[np.ndarray], 
                    labels:list[str], 
                    savedir:str, 
                    class_labels:list[str], 
                    plot_savedir:str, 
                    enforce_lim:bool=True, 
                    contour:bool=True,
                    loglog:bool=False,
                    do_CPH:bool=True):
    #corner plot, BIL, LAL
    N = len(labels)
    fig, axs = plt.subplots(N, N, figsize=(20, 20))
    #oneD_hist_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="black", bins=40)#,
                       # bins = np.logspace(-5, 8, num=100))
    oneD_hist_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    #hide all of the ones not used
    for i in range(N):
        for j in range(i+1, N):
            axs[i, j].axis('off')
    
    cmaps = ["Blues",
        "Purples",
        "Greens",
        "Reds",
        "Purples"]

    cmaps2 = ['b', 'p', 'g', 'r', 'p']

    #do the 1-d plots
    for i in range(N):
        norm_factor = 0
        for j, class_data in enumerate(data):
            norm_factor = min(norm_factor, class_data[:, i].min())

        for j, class_data in enumerate(data):
            #axs[i, i].hist(np.log10(class_data[:, i] - norm_factor+1), **oneD_hist_kwargs, label = class_labels[j])
            if labels[j] == "glitches_new" or labels[j]== "GLITCH":
                LBL = "Glitches"
            elif labels[j] == "injected" or labels[j] == "SG":
                LBL = "SG Injection"
            elif labels[j] == "bbh" or labels[j] == "BBH":
                LBL = "BBH"
            elif labels[j] == "bkg" or labels[j] == "BKG":
                LBL = "Background"
            else:
                LBL = labels[i]
            axs[i, i].hist(class_data[:, i], **oneD_hist_kwargs, label = LBL)
            np.save(f"{plot_savedir}/one_d_hist_{i}_{j}.npy", class_data[:, i])
            if loglog:
                axs[i, i].loglog()
            if enforce_lim:
                axs[i, i].set_xlim(0, 1.5)
        #axs[i, i].legend()

    
    
    log_scaling = False

    corner_plot_hist = [labels]
    #do 2-d plots
    for i in range(N):
        for j in range(i):
            #sns.kdeplot(x=BIL[i], y=BIL[j], cmap="Reds", shade=True, bw_adjust=.5)
            norm_factor_i = 0
            norm_factor_j = 0
            for k, class_data in enumerate(data): #find the additive normalizing factor, has to the same by class
                norm_factor_i = min(norm_factor_i, class_data[:, i].min())
                norm_factor_j = min(norm_factor_j, class_data[:, j].min())

            for k, class_data in enumerate(data):

                if log_scaling: #modify the data being plotted
                    epsilon = 1 #so we can take log10 of min value
                    A = class_data[:, i] - norm_factor_i + epsilon
                    A = np.log10(A)
                    B = class_data[:, j] - norm_factor_j + epsilon
                    B = np.log10(B)
                else:
                    A, B = class_data[:, i], class_data[:, j]

                
                if contour:
                    xx, yy, f = density_plot(A, B)
                    #cset = axs[i, j].contour(xx, yy, f, cmap='Blues', label=f"class {j}")
                    #cset = axs[i, j].contour(xx, yy, f, cmap = cmaps[k])
                    cset = axs[i, j].contour(yy, xx, f, cmap = cmaps[k])
                    axs[i, j].clabel(cset, inline=1, fontsize=10)
                    if enforce_lim:
                        axs[i, j].set_xlim(0, 1.2)
                        axs[i, j].set_ylim(0, 1.2)
                    if loglog:
                        axs[i, j].loglog()
                    #else:
                    #    axs[i, j].set_xlim(-50, 10)
                    #    axs[i, j].set_ylim(-50, 10)
                    

                    #save these values somehow
                    corner_plot_hist.append([i, j, k, yy, xx, f])


                else:
                    #print(A.shape)
                    #print("DEBUG 97", cmaps[k])
                    axs[i, j].scatter(B, A, s = 15, c=cmaps[k][:-1])
                    if enforce_lim:
                        axs[i, j].set_xlim(0, 1.2)
                        axs[i, j].set_ylim(0, 1.2)
                    if loglog:
                        axs[i, j].loglog()
                    #else:
                    #    axs[i, j].set_xlim(-50, 10)
                    #    axs[i, j].set_ylim(-50, 10)

            #axs[i, j].legend()

    #axis labels
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
        
    if type(savedir) == str:
        fig.savefig(savedir, dpi=300)
    else:
        for path in savedir:
            fig.savefig(path, dpi=300)

    #save the corner plot hist
    corner_plot_hist = np.array(corner_plot_hist, dtype="object")
    if do_CPH:
        np.save(f"{plot_savedir}/CPH.npy", corner_plot_hist)

def QUAK_plotting(data_predicted_path, plot_savedir, class_labels, QUAK_scaling=False):
    N_classes = len(os.listdir(data_predicted_path))

    corner_plot_data = []
    for i in range(N_classes):
        folder_name = f"{data_predicted_path}/{class_labels[i]}/"
        class_data = np.load(folder_name + "QUAK_evals.npy")
        if 0:
            QUAK_scale = np.load(folder_name + f"../../../DATA/EXTRAS/{class_labels[i]}.npy")
            #print("SHAPES", class_data.shape, QUAK_scale.shape)
            class_data = class_data * scale_norm(QUAK_scale[:, np.newaxis])
        #class data is the loss across all autoencoders for a given dataclass
        corner_plot_data.append(class_data)

    corner_plotting(corner_plot_data, class_labels, [f"{plot_savedir}/QUAK_plot.pdf", 
                                             f"{plot_savedir}/QUAK_plot.png"], class_labels, plot_savedir)

def KDE_plotting(data_predicted_path, plot_savedir, class_labels): 
    '''
    VERY similar in concept to the QUAK plotting, but instead of the autoencoedr recreation on each axis
    we have the ln(probability) of belonging to that class on each axis
    working with the data from DATA_PREDICTION/(classes)/KDE_evals.npy

    most of the stuff in the pipeline/pipeline_main.py for this section is very similar as well
    '''
    N_classes = len(os.listdir(data_predicted_path))

    corner_plot_data = []
    for i in range(N_classes):
        folder_name = f"{data_predicted_path}/{class_labels[i]}/"
        class_data = np.load(folder_name + "KDE_evals.npy")
    
        corner_plot_data.append(class_data)

        debug_185=True
        if debug_185:
            #bit of debuggint the plots here
            print("186 DEBUG DEBUG")

            print("class_labels[i]", class_labels[i])
            print("class_data", class_data)
    print("212, getting to plotting")
    corner_plotting(corner_plot_data, class_labels, [f"{plot_savedir}/KDE_QUAK_plot.pdf", 
                                             f"{plot_savedir}/KDE_QUAK_plot.png"], class_labels, plot_savedir,
                                            enforce_lim=False, contour=True, loglog=False, do_CPH=False)


def QUAK_STD_plotting(data_predicted_path, plot_savedir, class_labels, QUAK_scaling=False):
    N_classes = len(os.listdir(data_predicted_path))

    corner_plot_data = []
    for i in range(N_classes):
        folder_name = f"{data_predicted_path}/{class_labels[i]}/"
        class_data = np.load(folder_name + "QUAK_evals_STD.npy")
        if QUAK_scaling:
            QUAK_scale = np.load(folder_name + f"../../../DATA/EXTRAS/{class_labels[i]}.npy")
            #print("SHAPES", class_data.shape, QUAK_scale.shape)
            class_data = class_data * scale_norm(QUAK_scale[:, np.newaxis])
        #class data is the loss across all autoencoders for a given dataclass
        corner_plot_data.append(class_data)

    corner_plotting(corner_plot_data, class_labels, [ 
                                             f"{plot_savedir}/QUAK_STD_plot.png"], class_labels)

def LS_plotting(data_predicted_path, plot_savedir, class_labels):
    N_classes = len(os.listdir(data_predicted_path))

    corner_plot_data = []
    labels = []
    for i in range(N_classes):
        folder_name = f"{data_predicted_path}/{class_labels[i]}/"
        class_data = np.load(folder_name + "LS_evals.npy")
        #class data is the representation in the latent space of the certain class

        #handle the case where the latent space is not a vector, but perhaps a matrix
        #just flatten here
        class_data = class_data.reshape(len(class_data), -1)
       

        N_latentspace_dims = class_data.shape[1]
        if N_latentspace_dims > 10:
            print(f"Too many latent space dims to plot! ({N_latentspace_dims})")
            return None

        corner_plot_data.append(class_data)


    for i in range(N_latentspace_dims):
        labels.append(f"latent dim {i}")

    corner_plotting(corner_plot_data, labels, [f"{plot_savedir}/LS_plot.png"], class_labels)

def QUAK_3Dscatter(data_predicted_path, plot_savedir, class_labels, QUAK_scaling=False):
    N_classes = len(os.listdir(data_predicted_path))

    corner_plot_data = []
    for i in range(N_classes):
        folder_name = f"{data_predicted_path}/{class_labels[i]}/"
        class_data = np.load(folder_name + "QUAK_evals.npy")
        if QUAK_scaling:
            QUAK_scale = np.load(folder_name + f"../../../DATA/EXTRAS/{class_labels[i]}.npy")
            #print("SHAPES", class_data.shape, QUAK_scale.shape)
            class_data = class_data * scale_norm(QUAK_scale[:, np.newaxis])
        #class data is the loss across all autoencoders for a given dataclass
        corner_plot_data.append(np.log(class_data))

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')

    for i, data in enumerate(corner_plot_data):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], label=class_labels[i])

    fig.legend()
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_zscale("log")

    ax.set_xlabel(class_labels[0])
    ax.set_ylabel(class_labels[1])
    ax.set_zlabel(class_labels[2])
    plt.savefig(plot_savedir + "/QUAK_3dscatter.png", dpi=300)

def main(data_predicted_path:str, 
        plot_savedir:str,
        class_labels:list[str],
        make_QUAK:bool,
        do_LS:bool):
    '''
    plotting function for the various encoder evaluation results
    '''
    try:
        os.makedirs(plot_savedir)
    except FileExistsError: 
        None

    QUAK_scaling=False
    #print("plotting .py got for savedir", plot_savedir)
    if do_LS:
        LS_plotting(data_predicted_path, plot_savedir, class_labels)
    if make_QUAK:
        QUAK_plotting(data_predicted_path, plot_savedir, class_labels, QUAK_scaling)
        if len(class_labels)==3:
            QUAK_3Dscatter(data_predicted_path, plot_savedir, class_labels, QUAK_scaling)
        if 0: #didn't seem to be too helpful
            QUAK_STD_plotting(data_predicted_path, plot_savedir, class_labels, QUAK_scaling)
    
