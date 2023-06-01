import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from helper_functions import mae
from pre_processing import process


def nn_predict(savedir, data):
    '''
    Function to evaluate the trained NN on data from the QUAK space
    assuming that data is in the shape (N_samples, sample_axis, 4)
    '''

    nn_quak_model = load_model(f'{savedir}/TRAINED_MODELS/QUAK_NN/quak_nn.h5')
    print('data going into NN QUAK', data.shape)
    data_r = np.reshape(data, (data.shape[0]*data.shape[1], 4))
    print('data after reshape', data_r.shape)
    preds = nn_quak_model.predict(data_r)
    print('after neural network prediction shape', preds.shape)

    return np.reshape(preds, (data.shape[0], data.shape[1], 4))


def data_segment(data, seg_len, overlap):
    '''
    Function to slice up data into overlapping segments
    seg_len: length of resulting segments
    overlap: overlap of the windows in units of indicies

    assuming that data is of shape (N_samples, axis_to_slice_on, features)
    '''
    print('data segment input shape', data.shape)
    N_slices = (data.shape[1]-seg_len)//overlap
    print('N slices,', N_slices)
    print('going to make it to, ', N_slices*overlap+seg_len)
    data = data[:, :N_slices*overlap+seg_len]

    result = np.empty((data.shape[0], N_slices, seg_len, data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(N_slices):
            start = j*overlap
            end = j*overlap + seg_len
            result[i, j, :, :] = data[i, start:end, :]

    return result


def quak_predict(savedir, datae, preprocess=True):
    '''
    Function to get the QUAK losses on an input set of data
    '''
    print("into quak prediction", )
    print("datae in 33, ", datae.shape)
    datae = datae.swapaxes(2, 3)
    a, b = datae.shape[0], datae.shape[1]

    #load QUAK models
    QUAK_models = dict()
    QUAK_model_path = f"{savedir}/TRAINED_MODELS/QUAK/"
    for QUAK_class in os.listdir(QUAK_model_path):
        #print("QUAK class", QUAK_class)
        QUAK_models[QUAK_class] = load_model(f"{QUAK_model_path}/{QUAK_class}/AE.h5")

    all_results = np.empty((datae.shape[0], datae.shape[1], 4))
    #for i in range(len(datae)):
    data = datae.reshape(datae.shape[0]*datae.shape[1], 2, 100)
    #data = np.swapaxes(data, 1, 2)
    if preprocess:
        data = process(data)
    QUAK_evals = dict()
    for QUAK_class in QUAK_models:
        #print("95, data_class, QUAK_class", data_class, QUAK_class)
        #print("data", data)

        #pred = QUAK_models[QUAK_class].predict(data)
        pred = QUAK_models[QUAK_class].__call__(data)
        QUAK_evals[QUAK_class] = mae(data, pred)
        #print("99, mae from autoencoder", QUAK_evals[data_class][QUAK_class])
        #out_len = len(mae(data_dict[data_class], pred))

    if "BBH" in QUAK_evals.keys():
        index_map = {'BBH': 0,
                    "BKG": 1,
                    "GLITCH":2,
                    "SG":3}
        index_map_inv = {0:'BBH',
            1:"BKG",
            2:"GLITCH",
            3:"SG"}

    else:
        index_map = {'bbh': 0,
                    "bkg": 1,
                    "glitches_new":2,
                    "injected":3}
        index_map_inv = {0:'bbh',
                    1:"bkg",
                    2:"glitches_new",
                    3:"injected"}

    QUAK_stack = np.zeros(shape=(len(QUAK_evals[list(index_map.keys())[0]]), 4))

    for val in index_map:
        QUAK_stack[:, index_map[val]] = QUAK_evals[val]

    return QUAK_stack.reshape(a, b, 4)


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

            QUAK_values = quak_predict(savedir, data_segs)
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
            NN_values = nn_predict(savedir,
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


def main(savedir:str, runthrough_file:str, eval_savedir:str, savedir_list=None, split_specify=None):


    def make_kernel(N):
        return np.ones(N)/N

    N_kernel=20
    kernel = make_kernel(N_kernel)


    # for ext in ['/QUAK_evals/', '/KDE_evals/', '/NN_evals/', '/strain_data/']:

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
    path_of_interest = "/home/ryan.raikman/s22/TIMESLIDE_PROCESSED/4_27_TS_PROCESS/"
    all_saves = []
    for elem in os.listdir(path_of_interest):

        if len(os.listdir(f"{path_of_interest}/{elem}/strain_data/")) != 0:
            all_saves.append(f"{path_of_interest}/{elem}/")
    main("/home/ryan.raikman/s22/anomaly/march23_nets/bbh_updated/", "/home/ryan.raikman/s22/anomaly/generated_timeslides/1252150173_1252152348/timeslide_data.npy",
        all_saves[0], all_saves)
