import matplotlib.pyplot as plt
import numpy as np
import os
from labellines import labelLine, labelLines
import scipy.stats as st
CPH = np.load("/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/PLOTS/CPH.npy", allow_pickle=True)
#/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/PLOTS/one_d_hist_0.npy
labels = CPH[0]
CPH = CPH[1:]
N = len(labels)

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

def QUAK_bounds(quak_space_points, time_index=None):
    weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    default_vals = np.array([0.75, 0.65, 0.83, 0.75, 0.24])
    #default_vals = np.array([0, 0, 0, 0, 0])
    #[]
    default_vals = np.array([0.78109024, 0.73910072, 0.88487804, 0.78418769,0.16332473])
    #   default_vals = np.array([0.78109024, 0.73910072, 0.88487804, 0.78418769,0.6])
    #default_vals = np.array([0.6, 0.75, 0.83, 0.75, -5])
    #default_vals = np.array([0.75, 0.65, 0.83, 0.75, 0.6])
    model_savedir = "/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/"
    scatter_x = np.load(f"{model_savedir}/PLOTS/scatter_x_BBH_WEIGHTS.npy")
    scatter_y = np.load(f"{model_savedir}/PLOTS/scatter_y_BBH_WEIGHTS.npy")
    C = 0.27
    fig, axs = plt.subplots(N, N, figsize=(20, 20))
    for i in range(N):
        for j in range(i+1, N):
            axs[i, j].axis('off')

    #for i in range(N):
    #    axs[i, -1].axis("off")
    oneD_hist_kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

    cmaps = ["Blues",
    "Purples",
    "Greens",
    "Reds",
    "Purples"]
    class_data = quak_space_points

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

                dens = True
                if 0:
                    if dens:
                        A, B = class_data[:, i], class_data[:, j]
                        xx, yy, f = density_plot(A, B)
                        #cset = axs[i, j].contour(xx, yy, f, cmap='Blues', label=f"class {j}")
                        #cset = axs[i, j].contour(xx, yy, f, cmap = cmaps[k])
                        cset = axs[i, j].contour(yy, xx, f, cmap = "spring")
                        axs[i, j].clabel(cset, inline=1, fontsize=10)
                    else:
                        for quak_space_point in quak_space_points:
                            axs[i, j].scatter(quak_space_point[j], quak_space_point[i],c="black", s=15)

                #plotting lines:
                cmap = plt.get_cmap("YlGn")

                for delta in np.linspace(-0.0455, 0.06, 4, endpoint=True):
                    C_calc = C + delta
                    far_ind = np.searchsorted(scatter_x, C_calc)
                    #print("far ind", far_ind)
                    if far_ind == len(scatter_x):
                        far_ind = len(scatter_x) - 1 #doesn't really matter at this point
                    far_val = (scatter_y[far_ind])

                    for z in range(5):
                        if z != i and z != j:
                            C_calc -= default_vals[z] * weights[z]

                    #weights[i] * x + weights[j] * y = C_calc
                    xs = np.array([0, 1.2])
                    def calc_y(x):
                        return (C_calc - weights[i]*x)/weights[j]
                    ys = calc_y(xs)
                    print("i, j, xs, ys", i, j, xs, ys)
                    scale_val = 0.6**(abs(delta+0.05)/0.035)
                    month=2.628e6
                    sci_not = '{:0.1e}'.format(far_val)
                    print("far", far_val)
                    far_year = round(month*12*far_val, 1)
                    far_month = round(month*far_val, 1)
                    far_day = round(month/30*far_val, 1)
                    far_hour = round(month/30/24*far_val, 1)
                    if far_year < 5:
                        final = f"{far_year} / year"
                    elif far_month < 15:
                        final = f"{far_month} / month"
                    elif far_day < 15:
                        final = f"{far_day} / day"
                    else:
                        final = f"{far_hour} / hour"
                    axs[i, j].plot(ys, xs, c="black", alpha = scale_val, label = final ) #cmap(scale_val)



                labelLines(axs[i, j].get_lines(), zorder=2.5, xvals = (0.3, 0.3, 0.3, 0.3))





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

        color_list = ["blue", "purple", "green", "red"]
        for i in range(N):
            for j in range(N):
                axs[i, i].hist(np.load(f"/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/PLOTS/one_d_hist_{i}_{j}.npy"), 
                            **oneD_hist_kwargs, color=color_list[j])

    fig.savefig("/home/ryan.raikman/s22/temp6/QUAK_bound_noSN.png", dpi=300)
    plt.close(fig)
if 0:
    print("first")
    #a = np.load("/home/ryan.raikman/s22/anomaly/4_12_timeslides/pearson_vals.npy")
    #print(np.mean(a, axis=0))
    #print("second")
    #b = np.load("/home/ryan.raikman/s22/anomaly/4_12_timeslides/QUAK_vals.npy")
    #print(np.mean(b, axis=0))
    #assert 0 
    b = np.load("/home/ryan.raikman/s22/anomaly/4_12_timeslides/QUAK_vals.npy", mmap_mode="r")
    tot_len = b.shape[0]
    N_splits = 2000
    #print(np.average(b[5:100000, :], axis=0))
    #assert 0
    print("tot len", tot_len)
    print("got here")
    averages = []
    for i in range(N_splits):
        print(f"iteration {i}", end="\r")
        #assert 0
        slx = slice(tot_len//N_splits*i, tot_len//N_splits*(i+1))
        #print("got here", tot_len//N_splits*i, tot_len//N_splits*(i+1))
        #assert 0
        #print("slx", slx)
        averages.append(np.average(b[slx, :], axis=0))
    #print(np.average(b[:, 0]))
    print(np.average(np.array(averages), axis=1))

    assert 0
#QUAK_bounds(np.load("/home/ryan.raikman/s22/temp6/numpy/element_0.npy"))

loaded = []
for file in os.listdir("/home/ryan.raikman/s22/temp6/numpy/"):
    if file[-5] != "n":
        loaded.append(np.load(f"/home/ryan.raikman/s22/temp6/numpy/{file}"))

QUAK_bounds(None)