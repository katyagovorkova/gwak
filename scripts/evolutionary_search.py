import argparse
import numpy as np


def integrate(x, y):
    tot = 0
    for i in range(len(x)-1):
        tot += y[i]*(x[i+1]-x[i])
    return tot

def discriminator(quak, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([quak, pearson])
    return np.dot(datum, param_vec)

def make_norm_dist(mu, sig):
    def norm_dist(x):
        return -np.log(sig*(2*np.pi)**0.5) -0.5*((x-mu)/sig)**2
    return norm_dist

def score_parameters(param_vec, files):
    timeslide_quak, timeslide_pearson, signal_quak_flat, signal_pearson_flat, signal_pearson, signal_quak = files

    scores = discriminator(signal_quak_flat, signal_pearson_flat, param_vec)
    scores = np.reshape(scores, signal_pearson.shape) #should go back to what this originally was

    # take max along axis
    bestvals = np.amin(scores, axis=1)

    score_timeslides = discriminator(timeslide_quak, timeslide_pearson, param_vec)
    score_timeslides = np.reshape(score_timeslides, timeslide_pearson.shape)
    N_points = 300
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

    # new score: KL divergence from background and signal distributions
    bin_min = min(np.amin(score_timeslides), np.amin(bestvals))
    bin_max = max(np.amax(score_timeslides), np.amax(bestvals))

    NBINS = 50
    signal_dist, bin_edges = np.histogram(bestvals, bins=NBINS, range=(np.amin(bestvals), np.amax(bestvals)))
    signal_dist = signal_dist / len(bestvals)

    timeslide_dist = make_norm_dist(np.mean(score_timeslides), np.std(score_timeslides))

    if 0:
        KL_div = 0
        for i in range(len(timeslide_dist)):
            Px, Qx = timeslide_dist[i], signal_dist[i]
            KL_div += Px * (np.log(Px)-np.log(Qx))

        KL_div = 0
        for testval in np.linspace(bin_min, bin_max, 100):
            Px, Qx = kernel_timeslide(testval)[0], kernel_signal(testval)[0]
            KL_div += Px * (np.log(Px)-np.log(Qx))

    KL_div = 0
    for i, edge in enumerate(bin_edges[:-1]): #left edge
        Px, logQx = signal_dist[i], timeslide_dist(edge)
        if Px == 0: continue #eh....
        KL_div += Px**2 * (np.log(Px)-logQx)

    score3 = (bestvals<val_min).sum()/len(bestvals)

    print(f'params {param_vec}, score {TPRs[0]:.3f}, KL_div {KL_div* (np.mean(score_timeslides) > np.mean(bestvals)):.3f}')
    return TPRs[0]


def cmaes(fn, dim, files, num_iter=20):
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
    sigma = 10/decimate*1.5
    population_size = 100
    p_keep = 0.10  # Fraction of population to keep
    noise = 0.25/decimate/5  # Noise added to covariance to prevent it from going to 0.
    N_ELITE = int(p_keep*population_size)
    # Initialize the mean and covariance
    mu = np.zeros(dim)
    cov = sigma**2 * np.eye(dim)

    mu_vec = []
    best_sample_vec = []
    mean_sample_vec = []
    for t in range(num_iter):
        print("NEW ITER", t)
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
    Conduct an evolutionary search over the quak "weights"
    '''
    timeslide_quak = np.load(f'{args.timeslide_path}/timeslide_quak.npy')
    timeslide_pearson = np.load(f'{args.timeslide_path}/timeslide_pearson.npy')
    timeslide_pearson = np.abs(timeslide_pearson) # + 1e-12

    print(f'timeslides loaded in {time.time()-ts:.2f}',)


    p = np.random.permutation(len(timeslide_quak))
    cut = 50000000
    timeslide_quak = timeslide_quak[p][:cut]
    timeslide_pearson = timeslide_pearson[p][:cut]
    if 0:
      test_weights = np.array([ 0, 0, 1,  0.0, 0])
      vals = discriminator(timeslide_quak, timeslide_pearson, test_weights)
      sort_perm = np.argsort(vals)

      random_len = 0
      glitch_len = 100000
      random_part_quak = timeslide_quak[:random_len] #15 million of these
      glitchy_part_quak = timeslide_quak[sort_perm][:glitch_len]
      random_part_pearson = timeslide_pearson[:random_len] #15 million of these
      glitchy_part_pearson = timeslide_pearson[sort_perm][:glitch_len]

      timeslide_quak = np.vstack([random_part_quak, glitchy_part_quak])
      timeslide_pearson = np.vstack([random_part_pearson, glitchy_part_pearson])

    signal_quak = np.load(f'{args.signal_path}/signal_quak.npy')#[:, 1207-300:1207+300, :]
    signal_pearson = np.load(f'{args.signal_path}/signal_pearson.npy')#[:, 1207-300:1207+300, :]
    print('Signal shape', signal_quak.shape)

    signal_quak_flat = np.reshape(signal_quak, (signal_quak.shape[0]*signal_quak.shape[1], signal_quak.shape[2]))
    signal_pearson_flat = np.reshape(signal_pearson, (signal_pearson.shape[0]*signal_pearson.shape[1], 1))
    files = [timeslide_quak, timeslide_pearson, signal_quak_flat, signal_pearson_flat, signal_pearson, signal_quak]

    mu_vec, best_sample_vec, mean_sample_vec = cmaes(score_parameters, 5, files)
    np.save(args.save_file, best_sample_vec[-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('timeslide_path', type=str,
        help='Contains timeslide_quak.py, timeslide_pearson.npy')
    parser.add_argument('signal_path', type=str,
        help='Contains signal_quak.py, signal_pearson.npy')
    parser.add_argument('save_file', type=str,
        help='Where to save the best ES parameters')
    args = parser.parse_args()
    main(args)
