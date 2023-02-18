import numpy as np
import os
from anomaly.evaluation import QUAK_predict, data_segment, NN_predict, KDE_train, KDE_train_BKG
import matplotlib.pyplot as plt


def make_kernel(N):
    return np.ones(N)/N

N_kernel=50
kernel = make_kernel(N_kernel)


def smooth_samples(data):
    new_len = max(data.shape[1], N_kernel) - min(data.shape[1], N_kernel) + 1
    data_smooth = np.empty((data.shape[0], new_len, data.shape[2]))
    for j in range(len(data)):
        for k in range(data.shape[2]):
            #valid mode takes care of cutting off the edge effects
            data_smooth[j, :, k] = np.convolve(data[j, :, k], kernel, mode='valid')

    return data_smooth

def get_samples(savedir:str, runthrough_file:str, eval_savedir:str):
    '''
    Get QUAK space predictions on timeslide data
    '''
    
    #BKG_KDE_model = KDE_train_BKG(savedir)
    N_split = 500
    if 0:
        
        timeslide_data_all = np.load(runthrough_file)

        print("LOADED SHAPE 34!!", timeslide_data_all.shape)
        len_seg = len(timeslide_data_all)
        #print("all shape", timeslide_data_all.shape)
        indiv_len = int(len_seg//N_split)

    train_KDE_on_timeslides = True
    if train_KDE_on_timeslides:
        #load indicies 1->10 of QUAK data
        loaded_samples = 0
        loaded_data = []
        QUAK_path = "/home/ryan.raikman/s22/anomaly/TS_evals3/"
        for counter in range(10):
            data_load = np.load(f"{QUAK_path}/QUAK_evals/QUAK_evals_{counter}.npy")
            loaded_data.append(np.reshape(data_load, (data_load.shape[0]*data_load.shape[1], data_load.shape[2])  ))
            loaded_samples += data_load.shape[0]*data_load.shape[1]

        full_data = np.empty((loaded_samples, 4))
        
        marker = 0
        for data in loaded_data:
            full_data[marker:marker+len(data), :] = data
            marker += len(data)

        print("loaded data shape,", full_data.shape)
        
        #reduce the number of samples for training kde? seems like the right thing to do
        p = np.random.permutation(len(full_data))
        N_samples = int(1e6)
        full_data = full_data[p][:N_samples]

        KDE_model = KDE_train(savedir, full_data)
        
    else:
        KDE_model = KDE_train(savedir)

    N_kernel=100
    kernel = make_kernel(N_kernel)
    #assert 0
    in_shape_counter = 0
    all_samples = []
    for iteration in range(N_split):
        
        do_QUAK = False
        do_KDE = True
        do_NN = False
        do_compile_samples = False

        if do_QUAK:
            data_segs = data_segment(timeslide_data_all[i*indiv_len:(i+1)*indiv_len, :, :], 100, 5)
            print("data segs shape", data_segs.shape)
            #assert 0
            QUAK_values = QUAK_predict(savedir, data_segs)
            print("QUAK values shape", QUAK_values.shape)
            counter = len(os.listdir(f"{eval_savedir}/QUAK_evals/"))
            #QUAK_values_smooth = np.empty((QUAK_values.shape[0], new_len, 4))
            #for j in range(len(QUAK_values)):
            #    for k in range(4):
            #        #valid mode takes care of cutting off the edge effects
            #        QUAK_values_smooth[j, :, k] = np.convolve(QUAK_values[j, :, k], kernel, mode='valid')

            np.save(f"{eval_savedir}/QUAK_evals/QUAK_evals_{counter}.npy", QUAK_values)

        if do_KDE:
            if iteration < 10:
                print("skipping this iter", counter)
                continue
            if 0:
                KDE_values = KDE_model.eval(QUAK_values)
                counter = len(os.listdir(f"{eval_savedir}/KDE_evals/"))

                KDE_values_smooth = smooth_samples(KDE_values)
            else:
                #doing just KDE, specifically with the BKG kde
                counter = len(os.listdir(f"{eval_savedir}/KDE_evals/"))
                print("loading from 3")
                QUAK_values = np.load(f"/home/ryan.raikman/s22/anomaly/TS_evals3/QUAK_evals/QUAK_evals_{iteration}.npy")
                KDE_values = KDE_model.eval(QUAK_values)
                

                KDE_values_smooth = smooth_samples(KDE_values)

            np.save(f"{eval_savedir}/KDE_evals/KDE_evals_{counter}.npy", KDE_values_smooth)
        if do_NN:
            #try:
            #    QUAK_values = np.load(f"/home/ryan.raikman/s22/anomaly/TS_evals/timesegs_evals_{i}.npy")
            #except FileNotFoundError:
            #    break
            NN_values = NN_predict(savedir, 
                                QUAK_values)

            #smooth out the NN values
            new_len = max(NN_values.shape[1], N_kernel) - min(NN_values.shape[1], N_kernel) + 1
            NN_values_smooth = np.empty((NN_values.shape[0], new_len, 4))
            for j in range(len(NN_values)):
                for k in range(4):
                    #valid mode takes care of cutting off the edge effects
                    NN_values_smooth[j, :, k] = np.convolve(NN_values[j, :, k], kernel, mode='valid')
            
            NN_samples = NN_values_smooth.reshape(NN_values.shape[0], new_len, 4)
            counter = len(os.listdir(f"{eval_savedir}/NN_evals/"))
            np.save(f"{eval_savedir}/NN_evals/NN_evals_{counter}.npy", NN_samples)

        if do_compile_samples:
            try:
                KDE_samples = np.load(f"{eval_savedir}/KDE_evals/KDE_evals_{i}.npy")
                in_shape = KDE_samples.shape[0]
                in_shape_counter += in_shape
                #print("loaded shape", KDE_samples.shape)
                print("total segments loaded so far,", in_shape_counter, end="\r")
                #print("KDE_samples shape indiv: ", KDE_samples.shape)
                #print("indiv NN samples shape", NN_samples.shape)
                all_samples.append(KDE_samples)
            except FileNotFoundError:
                break
    #a, b = all_samples[0].shape[1], all_samples[0].shape[2]
    #stacked = np.zeros(in_shape_counter, a, b)

        stacked = np.vstack(all_samples)
        print("114 STACKED SHAPE", stacked.shape)
        return np.reshape(stacked, (stacked.shape[0]*stacked.shape[1], stacked.shape[2]))

    return None
def metric(data):
    #goes from 4d NN output space to a constant value
    #example would be the noise prob, noise+glitch prob, etc
    return data[:, 1]
    #return None

def metric2(data):
    return data[:, 1] + data[:, 2]

def metric3(data):
    return np.maximum(data[:, 1], data[:, 2])

def make_fake_ROC(process_data, val_min, val_max, savedir, metric_name, total_duration):
    '''
    Makes a simple graph of the discriminating value versus background samples
    passed through
    '''
    N_points=300
    scatter_x = []
    scatter_y = []
    print("process data", process_data.shape)
    for val in np.linspace(val_min, val_max, N_points):
        scatter_y.append( (process_data<val).sum())
        scatter_x.append(val)

    scatter_y = np.array(scatter_y)
    #print("92 scatter_y shape", scatter_y.shape)
    #print(scatter_y)
    scatter_y = scatter_y /len(process_data)

    scatter_y = scatter_y * 6533/8

    plt.figure(figsize=(15, 10))
    plt.plot(scatter_x, scatter_y)
    plt.xlabel(f"Discriminating value: {metric_name}", fontsize=15)
    plt.ylabel("False alarm rate, Hz", fontsize=15)
    plt.title(f"Fake ROC curve just for BKG, testing discriminator: {metric_name}", fontsize=20)
    #plt.yscale("log")
    plt.loglog()
    plt.savefig(f"{savedir}/PLOTS/BKG_FAKE_ROC_{metric_name}.png", dpi=300)
    plt.show()

    scatter_x = np.array(scatter_x)
    np.save(f"{savedir}/PLOTS/scatter_y.npy", scatter_y)
    np.save(f"{savedir}/PLOTS/scatter_x.npy", scatter_x)

def main(savedir:str, runthrough_file:str, eval_savedir:str):
    try:
        os.makedirs(eval_savedir)
    except FileExistsError:
        None

    for ext in ['/QUAK_evals/', '/KDE_evals/', '/NN_evals/']:
        try:
            os.makedirs(eval_savedir + ext)
        except FileExistsError:
            None
    samples = get_samples(savedir, runthrough_file, eval_savedir)
    #assert 0
    print("ALL SAMPLES OUT", samples.shape) 
    point_to_time_ratio = 8/6533
    print("background length we have is", samples.shape[0]*point_to_time_ratio, "seconds")
    total_duration = samples.shape[0]*point_to_time_ratio
    #now start doing statistics on the samples you have

    #samples_process = samples[:, 0]
    samples_process = metric3(samples)
    make_fake_ROC(samples_process, 0, 50, 
                savedir,
                "max BKG KDE",
                total_duration)

def make_sample_plots(savedir:str, eval_savedir:str):
    '''
    debugging function to make plots of QUAK evals, KDE predicts, etc
    '''
    try:
        os.makedirs(savedir + "/PLOTS/TEMP_PLOTS/KDE_debugs/")
    except FileExistsError:
        None
    maxval = 10
    plot_me = []
    #for ind in range(len(os.listdir(f"{eval_savedir}/KDE_evals/"))):
    #sorry for messing this up, frantically doing things for meeting tmrw(today)
    for ind in range(len(os.listdir(f"{eval_savedir}/KDE_evals/"))):
        X = np.load(f"{eval_savedir}/KDE_evals/KDE_evals_{ind}.npy")
        #print("X shape", X.shape)
        for sample in range(len(X)):
            outval = (np.maximum( X[sample, :, 1],  X[sample, :, 2]) )
            #print("202", outval.shape)
            outval = np.min(outval)
            #print("outval", outval)
            if outval < maxval:
                #maxind, maxsample = ind, sample
                plot_me.append([ind, sample])

    
    #ind, sample = maxind, maxsample
    #print("ind, sample", ind, sample)
    for i, (ind, sample) in enumerate(plot_me):
        print("plotting sample, ", i, end='\r')
        KDE_data = np.load(f"{eval_savedir}/KDE_evals/KDE_evals_{ind}.npy")
        #QUAK_data = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{ind}.npy")
        #NN_data = np.load(f"{eval_savedir}/NN_evals/NN_evals_{ind}.npy")
        #print("KDE_DATA", KDE_data)
        #print("KDE, QUAK", KDE_data.shape)#, QUAK_data.shape)
        plt.figure(figsize=(12, 7))
        plt.plot(KDE_data[sample, :, 0], label = 'ALL_BKG')
        #plt.plot(KDE_data[sample, :, 1], label = 'BKG')
        #plt.plot(KDE_data[sample, :, 2], label = 'GLITCH')
        #plt.plot(KDE_data[sample, :, 3], label = 'SG')
        plt.legend()
        plt.xlabel("datapoints")
        plt.ylabel("KDE values")
        plt.ylim(0,)
        plt.grid()
        i = len(os.listdir(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debugs/"))
        plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debugs/PLOT_{i}.png", dpi=300)
        plt.show()

    return None
    #sample = 0

    plt.figure(figsize=(12, 7))
    plt.plot(KDE_data[sample, :, 0], label = 'BBH')
    plt.plot(KDE_data[sample, :, 1], label = 'BKG')
    plt.plot(KDE_data[sample, :, 2], label = 'GLITCH')
    plt.plot(KDE_data[sample, :, 3], label = 'SG')
    plt.legend()
    plt.xlabel("datapoints")
    plt.ylabel("KDE values")
    plt.ylim(0,100)
    plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debug.png", dpi=300)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.plot(QUAK_data[sample, :, 0], label = 'BBH')
    plt.plot(QUAK_data[sample, :, 1], label = 'BKG')
    plt.plot(QUAK_data[sample, :, 2], label = 'GLITCH')
    plt.plot(QUAK_data[sample, :, 3], label = 'SG')
    plt.legend()
    plt.xlabel("datapoints")
    plt.ylabel("QUAK space value")
    plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/QUAK_debug.png", dpi=300)
    plt.show()
  
paths = ["/home/ryan.raikman/s22/anomaly/generated_timeslides/1241093492_1241123810/timeslide_data.npy",
        "/home/ryan.raikman/s22/anomaly/generated_timeslides/1241597285_1241616520/timeslide_data.npy",
        "/home/ryan.raikman/s22/anomaly/generated_timeslides/1242610599_1242624762/timeslide_data.npy",
        "/home/ryan.raikman/s22/anomaly/generated_timeslides/1246079821_1246084359/timeslide_data.npy",
        "/home/ryan.raikman/s22/anomaly/generated_timeslides/1249629764_1249662942/timeslide_data.npy",
        "/home/ryan.raikman/s22/anomaly/generated_timeslides/1252150173_1252152348/timeslide_data.npy",
        ]

if 1:
    main("/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/", paths[5], "/home/ryan.raikman/s22/anomaly/TS_evals6/")

if 0:
    #/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100
    make_sample_plots("/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/", "/home/ryan.raikman/s22/anomaly/TS_evals4/")
    main("/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/", paths[5], "/home/ryan.raikman/s22/anomaly/TS_evals4/")
    #get_samples("/home/ryan.raikman/s22/anomaly/bp_runs/COMPILED_RUN/", paths[-1], "/home/ryan.raikman/s22/anomaly/TS_evals2/")
if 0:
    #/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100
    make_sample_plots("/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/", "/home/ryan.raikman/s22/anomaly/TS_evals3/")
    main("/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100/", paths[5], "/home/ryan.raikman/s22/anomaly/TS_evals3/")
    

if 0:
    #/home/ryan.raikman/s22/anomaly/new_architecture_run/RUN_BIGDATA_2_13_20_100
    make_sample_plots("/home/ryan.raikman/s22/anomaly/bp_runs/COMPILED_RUN/", "/home/ryan.raikman/s22/anomaly/TS_evals2/")
    main("/home/ryan.raikman/s22/anomaly/bp_runs/COMPILED_RUN/", paths[5], "/home/ryan.raikman/s22/anomaly/TS_evals2/")
    #get_samples("/home/ryan.raikman/s22/anomaly/bp_runs/COMPILED_RUN/", paths[-1], "/home/ryan.raikman/s22/anomaly/TS_evals2/")
