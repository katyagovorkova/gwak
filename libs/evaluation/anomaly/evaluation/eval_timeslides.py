import numpy as np
import os
import sys
from anomaly.evaluation import QUAK_predict, data_segment, NN_predict, KDE_train, KDE_train_BKG, pearson_QUAK_stats
import matplotlib.pyplot as plt
import time
from scipy.stats import boxcox
from scipy.stats import pearsonr


def make_kernel(N):
    return np.ones(N)/N

N_kernel=20
kernel = make_kernel(N_kernel)

def smooth_samples(data):
    '''
    Input shape: (N_samples, features, 2)
    '''
    new_len = max(data.shape[1], N_kernel) - min(data.shape[1], N_kernel) + 1
    data_smooth = np.empty((data.shape[0], new_len, data.shape[2]))
    for j in range(len(data)):
        for k in range(data.shape[2]):
            #valid mode takes care of cutting off the edge effects
            data_smooth[j, :, k] = np.convolve(data[j, :, k], kernel, mode='valid')

    return data_smooth

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
    maxshift = int(10e-3*4096)//5 #why?? I don't know!!!
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

def get_samples(savedir:str, runthrough_file:str, eval_savedir:str, savedir_list=None, split_specify=None):
    '''
    Get QUAK space predictions on timeslide data
    '''
    
    #BKG_KDE_model = KDE_train_BKG(savedir)
    N_split = 500
    if 1:
        t0 = time.time()
        timeslide_data_all = np.load(runthrough_file)
        print(f"loading data took, {time.time()-t0:.2f}, seconds")
        print("LOADED SHAPE 34", timeslide_data_all.shape)
        len_seg = len(timeslide_data_all)
        indiv_len = int(len_seg//N_split)

    print("indiv_len", indiv_len)
    
    in_shape_counter = 0
    all_samples = []
    all_cors = []
    iteration_indicies = range(N_split)
    if split_specify is not None:
        iteration_indicies = iteration_indicies[split_specify]
    for iteration in iteration_indicies:
        
        do_QUAK = False
        do_KDE = False
        do_NN = False
        do_compile_samples = not do_QUAK
        load_corr = True
        
        save_dataseg=False
        save_pearson=True
        if do_QUAK:
            data_reduced = timeslide_data_all[iteration*indiv_len:(iteration+1)*indiv_len, :, :]
            print("data reduced shape", data_reduced.shape)
            data_segs = data_segment(data_reduced, 100, 5)
            pearson_evals = []
            
            if save_pearson:
                ts=time.time()
                #for x in range(len(data_segs)):
                #    pearson_evals.append(samples_pearson(data_segs[x]))
                #pearson_evals = np.vstack(pearson_evals)
                pearson_evals, edge_cut = samples_pearson(data_reduced)
                print("pearson_evals shape, edge cut", pearson_evals.shape, edge_cut)
                #assert 0
                np.save(f"{eval_savedir}/strain_data/pearson_{iteration}.npy", pearson_evals)
                print(f'pearson eval took {time.time()-ts:.4f} seconds')
            print("data segs shape", data_segs.shape)
            #assert 0
            
            QUAK_values = QUAK_predict(savedir, data_segs)
            #print("QUAK value shape", QUAK_values.shape)
            #print("after slice QUAK value shape", QUAK_values[:, edge_cut, :].shape)
            #assert 0
            smooth_QUAK = False
            if smooth_QUAK:
                QUAK_values_smooth = smooth_samples(QUAK_values)
                print("QUAK values shape", QUAK_values_smooth.shape)
                #counter = len(os.listdir(f"{eval_savedir}/QUAK_evals/"))

                np.save(f"{eval_savedir}/QUAK_evals/QUAK_evals_{iteration}.npy", QUAK_values_smooth)
            else:
                np.save(f"{eval_savedir}/QUAK_evals/QUAK_evals_{iteration}.npy", QUAK_values[:, edge_cut, :])
        if do_KDE:
            #if iteration < 10:
            #    print("skipping this iter", iteration)
            #    continue
            if do_QUAK:
                t0 = time.time()
                KDE_values = KDE_model.eval(QUAK_values_smooth)
                print(f"KDE evaluation took, {time.time()-t0:.2f}, seconds")
                counter = len(os.listdir(f"{eval_savedir}/KDE_evals/"))

                KDE_values_smooth = smooth_samples(KDE_values)
            else:
                #doing just KDE, specifically with the BKG kde
                counter = len(os.listdir(f"{eval_savedir}/KDE_evals/"))
                #print("loading from 3")
                QUAK_values = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{iteration}.npy")
                KDE_values = KDE_model.eval(QUAK_values)
                KDE_values_smooth = smooth_samples(KDE_values)

            #NOTE: not saving smooth here, going to try to do it with smoothing on QUAK values
            np.save(f"{eval_savedir}/KDE_evals/KDE_evals_{counter}.npy", KDE_values)
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

        if savedir_list is not None:
            maxlens = []
            for savedir_elem in savedir_list:
                max_len = len(os.listdir(f"{savedir_elem}/QUAK_evals"))# + 1
                maxlen_pearson = len(os.listdir(f"{savedir_elem}/QUAK_evals"))# + 1
                maxlens.append(min(max_len, maxlen_pearson))
  
        if do_compile_samples:
            load_KDE_samples = False

            if load_KDE_samples:
                samples = np.load(f"{eval_savedir}/KDE_evals/KDE_evals_{iteration}.npy")
            else:
                if savedir_list == None:
                    samples = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{iteration}.npy")
                    in_shape = samples.shape[0]
                    in_shape_counter += in_shape
                    #print("loaded shape", KDE_samples.shape)
                    print("total segments loaded so far,", in_shape_counter, end="\r")
                    #print("KDE_samples shape indiv: ", KDE_samples.shape)
                    #print("indiv NN samples shape", NN_samples.shape)
                    if load_corr:
                        corr = np.load(f"{eval_savedir}/strain_data/pearson_{iteration}.npy")
                        all_cors.append(corr)
                    all_samples.append(samples)
                    
                else:
                    for index_, savedir_elem in enumerate(savedir_list):
                        
                        if iteration >= maxlens[index_]: continue
                        try:
                            samples = np.load(f"{savedir_elem}/QUAK_evals/QUAK_evals_{iteration}.npy")
                        except:
                            continue
                        #print("259", samples.shape)
                        in_shape = samples.shape[0]
                        in_shape_counter += in_shape
                        print("total segments loaded so far,", in_shape_counter, end="\r")
                        all_samples.append(samples)
                        if load_corr:
                            corr = np.load(f"{savedir_elem}/strain_data/pearson_{iteration}.npy")
                            #print("266,", corr.shape)
                            all_cors.append(corr)

                        #early stop, getting samples for evolutionary search
                        #early_stop=True
                        #if early_stop:
                        #f in_shape_counter > 20000:
                        #    break

    #a, b = all_samples[0].shape[1], all_samples[0].shape[2]
    #stacked = np.zeros(in_shape_counter, a, b)

    SMOOTH_AFTER_LOAD = True
    if do_compile_samples:
        print("all samples len", len(all_samples))
        print("all_cors len", len(all_cors))
        stacked = np.vstack(all_samples)
        some_unsmoothed_QUAK = stacked[:50]
        some_unsmoothed_QUAK = np.reshape(some_unsmoothed_QUAK, (some_unsmoothed_QUAK.shape[0]*some_unsmoothed_QUAK.shape[1], some_unsmoothed_QUAK.shape[2]))
        if SMOOTH_AFTER_LOAD:
            stacked = smooth_samples(stacked)
        print("242", stacked.shape)
        if load_corr:
           # print("all cors", all_cors)
            corr_stacked = np.vstack(all_cors)[:, :, np.newaxis]
            some_unsmoothed_PEARSON = corr_stacked[:50]
            some_unsmoothed_PEARSON = np.reshape(some_unsmoothed_PEARSON, (some_unsmoothed_PEARSON.shape[0]*some_unsmoothed_PEARSON.shape[1], 1))
            if SMOOTH_AFTER_LOAD:
                corr_stacked = smooth_samples(corr_stacked)
            print(corr_stacked.shape)
            #print("220", stacked.shape, corr_stacked.shape)
            return np.reshape(stacked, (stacked.shape[0]*stacked.shape[1], stacked.shape[2])), \
                np.reshape(corr_stacked, (corr_stacked.shape[0]*corr_stacked.shape[1], 1)),\
                some_unsmoothed_QUAK, some_unsmoothed_PEARSON
        print("114 STACKED SHAPE", stacked.shape)
        return np.reshape(stacked, (stacked.shape[0]*stacked.shape[1], stacked.shape[2])), None

    return None

def discriminator(QUAK, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([QUAK, pearson])
    return np.dot(datum, param_vec)

def make_fake_ROC(process_data, val_min, val_max, savedir, metric_name, total_duration, order="lessthan"):
    '''
    Makes a simple graph of the discriminating value versus background samples
    passed through
    '''
    N_points=300
    scatter_x = []
    scatter_y = []
    print("process data", process_data.shape)
    for val in np.linspace(val_min, val_max, N_points):
        if order == "lessthan":
            scatter_y.append( (process_data<val).sum())
        else:
            assert order == "greaterthan"
            scatter_y.append( (process_data>val).sum())
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
    plt.yscale("log")
    #plt.loglog()
    plt.savefig(f"{savedir}/PLOTS/BKG_FAKE_ROC_{metric_name}.png", dpi=300)
    plt.show()

    scatter_x = np.array(scatter_x)
    np.save(f"{savedir}/PLOTS/scatter_y_{metric_name}.npy", scatter_y)
    np.save(f"{savedir}/PLOTS/scatter_x_{metric_name}.npy", scatter_x)

def main(savedir:str, runthrough_file:str, eval_savedir:str, savedir_list=None, split_specify=None):
    try:
         os.makedirs(eval_savedir)
    except FileExistsError:
        None

    for ext in ['/QUAK_evals/', '/KDE_evals/', '/NN_evals/', '/strain_data/']:
        try:
            os.makedirs(eval_savedir + ext)
        except FileExistsError:
            None
    samples, pearson_vals, samples_train, pearson_train = get_samples(savedir,runthrough_file, eval_savedir, savedir_list, split_specify)
    #/home/ryan.raikman/s22/anomaly/ES_savedir
    if 0:
        path = "/home/ryan.raikman/s22/anomaly/ES_savedir_short/"
        try:
            os.makedirs(path)
        except FileExistsError:
            None
        np.save(f"{path}timeslide_QUAK.npy", samples)
        np.save(f"{path}timeslide_pearson.npy", pearson_vals)
        #assert 0

        #try this at first - training on non smoothed data
        #np.save("/home/ryan.raikman/s22/anomaly/ES_savedir2/timeslide_QUAK_TRAIN.npy", samples_train)
        #np.save("/home/ryan.raikman/s22/anomaly/ES_savedir2/timeslide_pearson_TRAIN.npy", pearson_train)

    samples = samples[:min(len(samples), len(pearson_vals))]
    pearson_vals = np.abs(pearson_vals)# + 1e-12
    pearson_train = np.abs(pearson_train)# + 1e-12
    
    #do some saving
    np.save("/home/ryan.raikman/s22/anomaly/4_12_timeslides/QUAK_vals.npy", samples)
    np.save("/home/ryan.raikman/s22/anomaly/4_12_timeslides/pearson_vals.npy", pearson_vals)

    print("392, s", samples.shape, pearson_vals.shape)
    print("393, t", samples_train.shape, pearson_train.shape)
    #assert 0
    print("ALL SAMPLES OUT", samples.shape)
    point_to_time_ratio = 8/6533
    print("background length we have is", samples.shape[0]*point_to_time_ratio, "seconds")
    total_duration = samples.shape[0]*point_to_time_ratio
    #now start doing statistics on the samples you have
    #assert 0
    #samples_process = samples[:, 0]

    #samples_process = katya_metric2(samples)
    #pearson_QUAK_func = pearson_QUAK_stats(samples[:20000], pearson_vals[:20000], f"{savedir}/TRAINED_MODELS/PEARSON_QUAK_STATS/")
    #                         BBH,      background,    glitch,        SG
    '''
    BBH_weights = np.array([ 0.20342629, 0.15863359, -0.24620787,  0.1002136 ])
    BBH_SG_weights = np.array([-0.06755583,  0.17923768, -0.004011,   -0.04297801])
    BBH_SG_weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    BBH_weights = np.array([0.48613072, -0.17497687, -0.01590199,  0.1017961,  -0.14085923])
    BBH_weights = np.array([-0.01666499,  0.29371879, -0.23844285, -0.01687215, -0.05543631])
    #BBH_SG_weights = np.array([-0.01488957,  0.30431612, -0.25321185, -0.01635141, -0.06630924])
    '''
    BBH_SG_weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
    samples_process = discriminator(samples, pearson_vals, BBH_SG_weights)

    make_fake_ROC(samples_process, np.min(samples_process), np.max(samples_process), 
                savedir,
                "BBH_WEIGHTS",
                total_duration,
                order = "lessthan")
    if 0:
        bar=-0.7
        of_interest = []
        of_interest_pearson = []
        for i, elem in enumerate(samples_process):
            if elem < bar:
                of_interest.append(samples[i])
                of_interest_pearson.append(pearson_vals[i])

        print("of interest has", len(of_interest), samples)
        print(np.array(of_interest).shape)
        full_of_interest = np.zeros((len(of_interest), 5))
        full_of_interest[:, :4] = np.array(of_interest)
        full_of_interest[:, 4] = np.array(of_interest_pearson)[:, 0]
        np.save("/home/ryan.raikman/s22/elems_of_interest.npy", full_of_interest)
        np.save("/home/ryan.raikman/s22/pearson_vals.npy", pearson_vals)

def timeslides_to_kde_search():None

def make_sample_QUAK_plots(savedir:str, eval_savedir:str, runthrough_file:str=None):
    QUAK_data = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_0.npy")
    print("247, QUAK shape", QUAK_data.shape)
    for sample in range(5):
        print(f"making sample {sample}", end ="\r")
        if 0:
            plt.figure(figsize=(12, 7))
            plt.plot(QUAK_data[sample, :, 0], label = 'BBH')
            plt.plot(QUAK_data[sample, :, 1], label = 'BKG')
            plt.plot(QUAK_data[sample, :, 2], label = 'GLITCH')
            plt.plot(QUAK_data[sample, :, 3], label = 'SG')
            plt.legend()
            plt.xlabel("datapoints")
            plt.ylabel("QUAK values")
            #plt.ylim(0,100)
            plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/QUAK_debug_{sample}.png", dpi=300)
            plt.show()
        if 1:
            fig, ax = plt.subplots(2, figsize=(12, 14))
            ax[0].plot(QUAK_data[sample, :, 0], label = 'BBH')
            ax[0].plot(QUAK_data[sample, :, 1], label = 'BKG')
            ax[0].plot(QUAK_data[sample, :, 2], label = 'GLITCH')
            ax[0].plot(QUAK_data[sample, :, 3], label = 'SG')
            ax[0].legend()
            ax[0].set_xlabel("datapoints")
            ax[0].set_ylabel("QUAK values")

            katya_measured = katya_metric(QUAK_data[sample, :, :])
            ax[1].plot(np.arange(len(katya_measured)), katya_measured)
            ax[1].set_xlabel("datapoints")
            ax[1].set_ylabel("katya metric")
            
            #plt.ylim(0,100)
            plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/QUAK_debug_{sample}.png", dpi=300)
            plt.show()

def make_sample_plots(savedir:str, eval_savedir:str, runthrough_file:str=None):
    '''
    debugging function to make plots of QUAK evals, KDE predicts, etc
    '''
    saveind = 2
    try:
        os.makedirs(savedir + f"/PLOTS/TEMP_PLOTS/KDE_debugs{saveind}/")
    except FileExistsError:
        None

    try:
        os.makedirs(savedir + "/PLOTS/TEMP_PLOTS/KDE_debugs_timeseries/")
    except FileExistsError:
        None

    maxval = 1
    plot_me = []
    #for ind in range(len(os.listdir(f"{eval_savedir}/KDE_evals/"))):
    #sorry for messing this up, frantically doing things for meeting tmrw(today)
    for ind in range(len(os.listdir(f"{eval_savedir}/KDE_evals/"))):
        X = np.load(f"{eval_savedir}/KDE_evals/KDE_evals_{ind}.npy")
        #print("X shape", X.shape)
        for sample in range(len(X)):
            #outval = (np.maximum( X[sample, :, 1],  X[sample, :, 2]) )
            outval = X[sample, :, 1]
            #print("202", outval.shape)
            outval = np.min(outval)
            #print("outval", outval)
            if outval < maxval:
                #maxind, maxsample = ind, sample
                plot_me.append([ind, sample])
    
    if runthrough_file is not None:
        N_split = 500
        if 1:
            timeslide_data_all = np.load(runthrough_file)

            print("LOADED SHAPE 34!!", timeslide_data_all.shape)
            len_seg = len(timeslide_data_all)
            #print("all shape", timeslide_data_all.shape)
            indiv_len = int(len_seg//N_split)

        print("indiv_len", indiv_len)
        
    #ind, sample = maxind, maxsample
    #print("ind, sample", ind, sample)
    print("plot_me", plot_me)
    for i, (ind, sample) in enumerate(plot_me):
        print("plotting sample, ", i, end='\r')
        KDE_data = np.load(f"{eval_savedir}/KDE_evals/KDE_evals_{ind}.npy")
        #QUAK_data = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{ind}.npy")
        #NN_data = np.load(f"{eval_savedir}/NN_evals/NN_evals_{ind}.npy")
        #print("KDE_DATA", KDE_data)
        #print("KDE, QUAK", KDE_data.shape)#, QUAK_data.shape)
        plt.figure(figsize=(12, 7))
        plt.plot(KDE_data[sample, :, 0], label = 'BBH')
        plt.plot(KDE_data[sample, :, 1], label = 'BKG')
        plt.plot(KDE_data[sample, :, 2], label = 'GLITCH')
        plt.plot(KDE_data[sample, :, 3], label = 'SG')

        plt.legend()
        plt.xlabel("datapoints")
        plt.ylabel("KDE values")
        plt.ylim(0,10)
        plt.grid()
        i = len(os.listdir(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debugs{saveind}/"))
        plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debugs{saveind}/PLOT_{i}.png", dpi=300)
        plt.show()

    if runthrough_file is not None:
        #going to use plot_me values to try and find the timeseries messing things up
        for i, (ind, sample) in enumerate(plot_me):
            data_segs = data_segment(timeslide_data_all[ind*indiv_len:(ind+1)*indiv_len, :, :], 100, 5)
            datum = data_segs[sample]
            print("298 datum shape", datum.shape)

            plt.figure(figsize=(12, 7))
            plt.plot(datum[:, 0], label='H1')
            plt.plot(datum[:, 1], label = "L1")

            plt.legend()

            i = len(os.listdir(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debugs_timeseries/"))
            plt.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/KDE_debugs_timeseries/PLOT_{i}.png", dpi=300)

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

def make_sample_plots_QUAK(savedir:str, eval_savedir:str, runthrough_file:str=None):
    '''
    debugging function to make plots of QUAK evals, KDE predicts, etc
    '''
    saveind = 2
    metric_choice = metric_v5
    try:
        os.makedirs(savedir + f"/PLOTS/TEMP_PLOTS/QUAK_DEBUGS{saveind}/")
    except FileExistsError:
        None
    maxval = 3.2
    plot_me = []
    outval_list = []
    ind_and_samps = []

    for ind in range(len(os.listdir(f"{eval_savedir}/QUAK_evals/"))):
        X = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{ind}.npy")
        #print("X shape", X.shape)
        
        #metric_evals = metric_katyav3(X)
        #print("metric eval shape", metric_evals.shape)
        #assert 0
        for sample in range(len(X)):
     
            #outval = np.min(metric_katyav3(X[sample, :, :]))
            outval = np.min(metric_choice(X[sample, :, :]))
            outval_list.append(outval)
            ind_and_samps.append([ind, sample])
        
    
    outval_list = np.array(outval_list)
    p = np.argsort(outval_list)
    plot_me = np.array(ind_and_samps)[p][:10]

    print("metric values", outval_list[p][:10])
            
    #ind, sample = maxind, maxsample
    #print("ind, sample", ind, sample)
    print("plot_me", plot_me)
    for i, (ind, sample) in enumerate(plot_me):
        print("plotting sample, ", i, end='\r')
        QUAK_data = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{ind}.npy")
        #print("QUAK data shape", QUAK_data.shape)
        metric_evals = metric_choice(QUAK_data[sample, :, :])
        print(metric_evals.shape)
        #QUAK_data = np.load(f"{eval_savedir}/QUAK_evals/QUAK_evals_{ind}.npy")
        #NN_data = np.load(f"{eval_savedir}/NN_evals/NN_evals_{ind}.npy")
        #print("KDE_DATA", KDE_data)
        #print("KDE, QUAK", KDE_data.shape)#, QUAK_data.shape)
        fig, axs =  plt.subplots(2, 1, figsize=(12, 7))
        axs[0].plot(QUAK_data[sample, :, 0], label = 'BBH')
        axs[0].plot(QUAK_data[sample, :, 1], label = 'BKG')
        axs[0].plot(QUAK_data[sample, :, 2], label = 'GLITCH')
        axs[0].plot(QUAK_data[sample, :, 3], label = 'SG')

        axs[0].legend()
        axs[0].set_xlabel("datapoints")
        axs[0].set_ylabel("QUAK values")
        #plt.ylim(0,10)
        #plt.grid()

        axs[1].plot(metric_evals)
        axs[1].set_xlabel("datapoints")
        axs[1].set_ylabel("metric values")
        i = len(os.listdir(f"{savedir}/PLOTS/TEMP_PLOTS/QUAK_DEBUGS{saveind}/"))
        fig.savefig(f"{savedir}/PLOTS/TEMP_PLOTS/QUAK_DEBUGS{saveind}/PLOT_{i}.png", dpi=300)
        fig.show()

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
search_dir = "/home/ryan.raikman/s22/anomaly/generated_timeslides/"
paths = []
for elem in os.listdir(search_dir):
    if len(os.listdir(f"{search_dir}/{elem}/")) == 1:
        #print(elem, os.listdir(f"{search_dir}/{elem}"))
        assert "timeslide_data.npy" in os.listdir(f"{search_dir}/{elem}/")
        paths.append(f"{search_dir}/{elem}/timeslide_data.npy")

paths = sorted(paths)
#make_sample_plots_QUAK("/home/ryan.raikman/s22/anomaly/march23_nets/double_glitch/", "/home/ryan.raikman/s22/anomaly/TS _evals_march_net_0/")
#print(paths)
#assert 0
if 0:
    width = 100
    if len(sys.argv) == 1:
        for i in range(len(paths)):
            for split_start in range(0, 500, width): #n split above is 500
                os.system(f"python3 eval_timeslides.py {i} {split_start}&")

    else:
        #print("697", sys.argv)
        i = int(sys.argv[1])
        split_start = int(sys.argv[2])
        print("running code with in index", i, "over split", f"[{split_start}:{split_start+width}[]")
        main("/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/", paths[i], 
                f"/home/ryan.raikman/s22/TIMESLIDE_PROCESSED/4_27_TS_PROCESS/SET_{i}/",
                split_specify=slice(split_start,split_start+width))


#main("/home/ryan.raikman/s22/anomaly/march23_nets/double_glitch/", paths[0], "/home/ryan.raikman/s22/anomaly/TS_evals_3_21_pearson_0/")
if 1:
    path_of_interest = "/home/ryan.raikman/s22/TIMESLIDE_PROCESSED/4_27_TS_PROCESS/"
    all_saves = []
    for elem in os.listdir(path_of_interest):
        
        if len(os.listdir(f"{path_of_interest}/{elem}/strain_data/")) != 0:
            all_saves.append(f"{path_of_interest}/{elem}/")

    #print("all valid saves", all_saves)
    #assert 0 
    #for i in range(6):
    #    all_saves.append(f"/home/ryan.raikman/s22/TIMESLIDE_PROCESSED/3_27_TS_PROCESS/SET_{i}/")
    #    #all_saves.append(f"/home/ryan.raikman/s22/anomaly/TS_evals_3_19_pearson_{i}/")
    main("/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/", paths[5], all_saves[0], all_saves)

#main("/home/ryan.raikman/s22/anomaly/march23_nets/double_glitch/", paths[1], "/home/ryan.raikman/s22/anomaly/TS_evals_march_net_pearson_0/", ["/home/ryan.raikman/s22/anomaly/TS_evals_march_net_0/", "/home/ryan.raikman/s22/anomaly/TS_evals_march_net_1/"])

'''
DEAD 
CODE


'''
def metric(data):
    #goes from 4d NN output space to a constant value
    #example would be the noise prob, noise+glitch prob, etc
    return data[:, 1]
    #return None

def metric2(data):
    return data[:, 1] + data[:, 2]

def metric3(data):
    return np.maximum(data[:, 1], data[:, 2])

def katya_metric(data):
    #sum of two signal losses minus bkg and glitch losses
    return data[:, 0] - data[:, 1] - data[:, 2] + data[:, 3]

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
    #print("279", quak_part.shape, corrs.shape)
    return np.subtract(quak_part, 3*corrs[:, 0])

def metric_nball(data, center, lmbdas):
    #find the distance from the center based on the boxcox transformation

    dists = np.zeros(len(data))
    for axis in range(4):
        xt = boxcox(data[:, axis], lmbdas[axis])
        dists += (xt-center[axis])**2
    return dists**0.5
