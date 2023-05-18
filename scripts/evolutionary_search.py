import os
import numpy as np
from scipy.stats import norm


def katya_metric(data, weights):
    # sum of two signal losses minus bkg and glitch losses
    return weights[0] * data[:, 0] + weights[1] * data[:, 1] + weights[2] * data[:, 2] + weights[3] * data[:, 3]

def build_model_from_save(savedir):
    model_params = np.load(f'{savedir}/trained/pearson_quak_stats_bbh/v2_model_params.npy')
    weights = np.load(f'{savedir}/trained/pearson_quak_stats_bbh/v2_model_params_QUAK_weights.npy')
    print('model_params', model_params)


    pearson_bc_lmbda, mu_pearson, sigma_pearson, mu_QUAK, sigma_QUAK = model_params

    return make_full_discriminator(pearson_bc_lmbda, mu_pearson, sigma_pearson,
                                     mu_QUAK, sigma_QUAK, weights)


def make_full_discriminator(
    pearson_bc_lmbda,
    mu_pearson,
    sigma_pearson,
    mu_QUAK,
    sigma_QUAK,
    weights):
    '''
    Given these parameters as fit based on the main function,
    return a function that takes in the QUAK and pearson values
    and returns a final discriminating metric
    '''
    if 0:
        xs_pearson_test = np.linspace(0, 1, 100)
        if kde_interp:
            pearson_peak = xs_pearson_test[np.argmax(pearson_kde(xs_pearson_test))]
        else:
            pearson_peak = xs_pearson_test[np.argmax(pearson_kde.__call__(xs_pearson_test))]


    def full_discriminator(quak_vals, pearson_vals):
        # check that data sizes match up
        N = len(quak_vals)
        assert len(pearson_vals) == N
        pearson_vals = np.abs(pearson_vals) + 1e-12 # on eval time, this should be okay, any point that's at -0.1 is going to get moved to the peak anyway

        # shift the pearson_vals below the peak to the peak
        # this equates to unnaturally low correlation, which is low probability
        # but we don't really care about these, so we just treat them as "average" uncorrelated
        if len(pearson_vals.shape)==2:
            assert pearson_vals.shape[1] == 1
            pearson_vals = pearson_vals[:, 0]

        pearson_bc = boxcox(pearson_vals, pearson_bc_lmbda)
        pearson_bc = np.clip(pearson_bc, mu_pearson, None) # on the right side of the distribution!

        # already did clipping by moving everything with pearson below mean to mean (roughly)
        log_pearson_density = -np.log(sigma_pearson * (2*np.pi)**0.5) - 0.5 * ((pearson_bc - mu_pearson)/sigma_pearson)**2

        #eval katya metric and shift
        km_QUAK = katya_metric(quak_vals, weights)

        # don't care about values to the right of the mean, those are just "really not signal"
        # so move them to the mean value
        km_QUAK = np.clip(km_QUAK, None, mu_QUAK)
        log_QUAK_density = -np.log(sigma_QUAK * (2*np.pi)**0.5) - 0.5 * ((km_QUAK - mu_QUAK)/sigma_QUAK)**2

        #now we have the final result:
        const = -np.log(sigma_QUAK * (2*np.pi)**0.5) -np.log(sigma_pearson * (2*np.pi)**0.5)
        #a, b = -pearson_lambda*pearson_vals,-0.5 * ((km_QUAK_bc - mu_QUAK)/sigma_QUAK)**2
        #print("FLOOD", np.mean(a), np.std(a), np.mean(b), np.std(b))
        #return a + 4*b
        return log_pearson_density + log_QUAK_density# or dont - const #'add it back
        #return np.clip(log_pearson_density, -35, None) + np.clip(log_QUAK_density, -35, None)

    return full_discriminator

def pearson_quak_stats(train_QUAK, train_pearson, savedir=None, weights = np.array([1, -0.5, -1.5, 1])):
    '''
    Callibrates the discrimination metric based on the
    training set
    '''
    if savedir is not None:
        try:
            os.makedirs(savedir)
        except FileExistsError:
            None

    assert len(train_pearson) >= 10000
    #shuffle data
    p1 = np.random.permutation(len(train_QUAK))
    p2 = np.random.permutation(len(train_pearson))
    train_QUAK = train_QUAK[p1]
    train_pearson = train_pearson[p2]
    if len(train_pearson.shape)==2:
        assert train_pearson.shape[1] == 1
        train_pearson = train_pearson[:, 0]

    #split each into half, one for training the boxcox and the other for training the parameters
    #not sure if this is necessary, but why not?
    train_pearson_lmbda = train_pearson[:len(train_pearson)//2]
    train_pearson_mu_sig = train_pearson[len(train_pearson)//2:]

    #train_QUAK_lmbda = train_QUAK[:len(train_QUAK)//2]
    train_QUAK_mu_sig = train_QUAK

    #build pearson model
    #no shift necessary!


    pearson_bc, pearson_bc_lmbda = boxcox(train_pearson_lmbda)
    pearson_bc_mu_sig = boxcox(train_pearson_mu_sig, pearson_bc_lmbda)
    mu_pearson, sigma_pearson = norm.fit(pearson_bc_mu_sig)


    #QUAK piece

    #eval katya metric on data
    km_QUAK = katya_metric(train_QUAK, weights)
    #km_shift = -np.min(km_QUAK) + 1e-6 + 1.5 #MAINTAIN
    #km_shift = -np.min(km_QUAK) + 10 #
    #km_QUAK = km_QUAK + km_shift

    #boxcox the data

    #km_QUAK_bc, QUAK_bc_lmbda = boxcox(km_QUAK)

    #fit a gaussian to the boxcoxed data
    #km_QUAK_mu_sig = (katya_metric(train_QUAK_mu_sig, weights))
    mu_QUAK, sigma_QUAK = norm.fit(km_QUAK)

    '''
    now we have collected all the parameters we need:
    PEARSON
    same as below without shift

    QUAK
    km_shift : shift value for post-katya metric values
    QUAK_bc_lmbda : lambda value for boxcox transformation
    mu_QUAK, sigma_QUAK : fitting paramteres for the gaussian distribution
    '''
    model_params = np.array([pearson_bc_lmbda, mu_pearson, sigma_pearson,
                                    mu_QUAK, sigma_QUAK])
    #print("PARAM NAMES", "pearson_bc_lmbda, mu_pearson, sigma_pearson, \
    #                                km_shift, QUAK_bc_lmbda, mu_QUAK, sigma_QUAK")
    #print("MODEL PARAMS", model_params)
    if savedir is not None:
        np.save(f"{savedir}/v2_model_params.npy", model_params)
        np.save(f"{savedir}/v2_model_params_QUAK_weights.npy", weights)

    final_discriminator = make_full_discriminator(pearson_bc_lmbda, mu_pearson, sigma_pearson,
                                     mu_QUAK, sigma_QUAK, weights)
    return final_discriminator

def integrate(x, y):
    tot = 0
    for i in range(len(x)-1):
        tot += y[i]*(x[i+1]-x[i])
    return tot

def score_parameters(param_vec, files):
    ts_quak_train, ts_pearson_train, timeslide_quak, timeslide_pearson, signal_quak_flat, signal_pearson_flat, signal_pearson, signal_quak = files
    discriminator = pearson_quak_stats(ts_quak_train, ts_pearson_train, weights=param_vec)

    scores = discriminator(signal_quak_flat, signal_pearson_flat)
    scores = np.reshape(scores, signal_pearson.shape) #should go back to what this originally was

    # take max along axis
    bestvals = np.amin(scores, axis=1)

    score_timeslides = discriminator(timeslide_quak, timeslide_pearson)
    score_timeslides = np.reshape(score_timeslides, timeslide_pearson.shape)

    N_points=300
    scatter_x = []
    scatter_y = []
    val_min = np.amin(score_timeslides)
    val_max = np.amax(score_timeslides)
    for val in np.linspace(val_min, val_max, N_points):
        scatter_y.append( (score_timeslides<val).sum())
        scatter_x.append(val)

    scatter_y = np.array(scatter_y)
    scatter_y = scatter_y /len(score_timeslides)
    scatter_y = scatter_y * 6533/8 # convert to FPR

    min_vals = bestvals
    TPRs = []
    for cutoff in scatter_x:
        # count the number of samples that pass through(FPR)
        TPRs.append((min_vals<cutoff).sum()/len(min_vals))

    TPRs=np.array(TPRs)
    FPRs = scatter_y

    print("params", param_vec, "score", TPRs[0])
    return TPRs[0]
    if FPRs[-1] == 0: # weird bug, not sure how to fix
        return 1
    return -integrate(np.log10(FPRs)[1:], TPRs[1:])


def cmaes(fn, dim, files, num_iter=10):
    """Optimizes a given function using CMA-ES.

    Args:
    fn: A function that takes as input a vector and outputs a scalar value.
    dim: (int) The dimension of the vector that fn expects as input.
    num_iter: (int) Number of iterations to run CMA-ES.

    Returns:
    mu_vec: An array of size [num_iter, dim] storing the value of mu at each
      iteration.
    best_sample_vec: A list of length [num_iter] storing the function value
      for the best sample from each iteration of CMA-ES.
    mean_sample_vec: A list of length [num_iter] storing the average function
      value across samples from each iteration of CMA-ES.
    """
    decimate = 70
    # Hyperparameters
    sigma = 10/decimate
    population_size = 100
    p_keep = 0.10  # Fraction of population to keep
    noise = 0.25/decimate  # Noise added to covariance to prevent it from going to 0.
    N_ELITE = int(p_keep*population_size)
    # Initialize the mean and covariance
    mu = np.zeros(dim)
    cov = sigma**2 * np.eye(dim)

    mu_vec = []
    best_sample_vec = []
    mean_sample_vec = []
    for t in range(num_iter):
        # WRITE CODE HERE
        children = np.random.multivariate_normal(mu, cov, size=population_size)
        fn_vals = []
        for child in children:
          fn_vals.append(fn(child, files))
        fn_vals = np.array(fn_vals)

        mean_sample_vec.append(np.average(fn_vals))

        p_sort = np.argsort(fn_vals)[::-1]
        elite = children[p_sort][:N_ELITE]

        best_sample_vec.append(fn_vals[p_sort][0])

        mu_vec.append(mu[:])

        mu = np.average(elite, axis=0)
        cov = np.cov(elite.T) + noise * np.eye(dim)

    return mu_vec, best_sample_vec, mean_sample_vec


def main(args):
    '''
    Conduct an evolutionary search over the QUAK "weights"
    '''
    timeslide_quak = np.load(f'{args.timeslide_path}/timeslide_quak.npy')
    timeslide_pearson = np.load(f'{args.timeslide_path}/timeslide_pearson.npy')
    timeslide_pearson = np.abs(timeslide_pearson) + 1e-12

    p = np.random.permutation(len(timeslide_quak))
    timeslide_quak = timeslide_quak[p][:4000000]
    timeslide_pearson = timeslide_pearson[p][:4000000]

    ts_quak_train = np.load(f'{args.timeslide_path}/timeslide_quak_train.npy')
    ts_pearson_train = np.load(f'{args.timeslide_path}/timeslide_pearson_train.npy')
    ts_pearson_train = np.abs(ts_pearson_train) + 1e-12
    print(f'timeslides train loaded in {time.time()-ts:.2f}')

    signal_quak = np.load(f'{args.timeslide_path}/signal_quak.npy')
    signal_pearson = np.load(f'{args.timeslide_path}/signal_pearson.npy')
    print(f'signal data loaded in {time.time()-ts:.2f}')

    signal_quak_flat = np.reshape(signal_quak, (signal_quak.shape[0]*signal_quak.shape[1], signal_quak.shape[2]))
    signal_pearson_flat = np.reshape(signal_pearson, (signal_pearson.shape[0]*signal_pearson.shape[1], 1))
    files = [ts_quak_train, ts_pearson_train, timeslide_quak, timeslide_pearson, signal_quak_flat, signal_pearson_flat, signal_pearson, signal_quak]
    score_parameters(np.array([1, -1, -1, 1]), files)

    print(cmaes(score_parameters, 4, files))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('timeslide_path', help='Contains timeslide_quak.py, timeslide_pearson.npy',
                        type=str)
    # parser.add_argument('signal_path', help='Contains signal_quak.py, signal_pearson.npy',
    #                     type=str, choices=['L1', 'H1'])
    args = parser.parse_args()
# path = "/home/ryan.raikman/s22/anomaly/ES_savedir"
# main(path, path)
    main(args)

