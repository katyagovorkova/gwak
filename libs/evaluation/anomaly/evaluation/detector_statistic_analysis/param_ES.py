import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import gaussian_kde

def integrate(x, y):
    tot = 0
    for i in range(len(x)-1):
        tot += y[i]*(x[i+1]-x[i])
    return tot

def discriminator(QUAK, pearson, param_vec):
    pearson = np.reshape(pearson, (len(pearson), 1))
    datum = np.hstack([QUAK, pearson])
    return np.dot(datum, param_vec)

def make_norm_dist(mu, sig):
    def norm_dist(x):
        #return 1/(sig*(2*np.pi)**0.5) * np.exp(-0.5*((x-mu)/sig)**2)
        return -np.log(sig*(2*np.pi)**0.5) -0.5*((x-mu)/sig)**2
    return norm_dist

def score_parameters(param_vec, files):
    timeslide_QUAK, timeslide_pearson, signal_QUAK_flat, signal_pearson_flat, signal_pearson, signal_QUAK = files
    #print("SIGNAL QUAK FLAT", signal_QUAK_flat)
    #np.save("/home/ryan.raikman/s22/temp2/TS_QUAK_data.npy", TS_QUAK_TRAIN)
    #discriminator = pearson_QUAK_stats(TS_QUAK_TRAIN, TS_PEARSON_TRAIN, weights=param_vec)

    scores = discriminator(signal_QUAK_flat, signal_pearson_flat, param_vec)
    scores = np.reshape(scores, signal_pearson.shape) #should go back to what this originally was

    #take max along axis
    #print("12 scores shape", scores.shape)
    bestvals = np.amin(scores, axis=1)
    #print("14, after minning shape", bestvals.shape)

    score_timeslides = discriminator(timeslide_QUAK, timeslide_pearson, param_vec)
    score_timeslides = np.reshape(score_timeslides, timeslide_pearson.shape)
    #print("scores shape, score_timeslides shape", scores.shape, score_timeslides.shape)
    #print("bestvals shape", bestvals.shape)
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
    scatter_y = scatter_y * 6533/8 #convert to FPR

    
    #FPRS = scatter_y[np.searchsorted(scatter_x, bestvals)]
    #print("FPRS, ", FPRS)
    #print("FPR average", np.average(FPRS), "with params", param_vec)
    #return -np.average(FPRS) #this is where I'm a bit stuck, perhaps log and average?
    min_vals = bestvals
    TPRs = []
    for cutoff in scatter_x:
        #count the number of samples that pass through(FPR)
        TPRs.append((min_vals<cutoff).sum()/len(min_vals))

    TPRs=np.array(TPRs)
    FPRs = scatter_y

    #new score: KL divergence from background and signal distributions
    bin_min = min(np.amin(score_timeslides), np.amin(bestvals))
    bin_max = max(np.amax(score_timeslides), np.amax(bestvals))

    #timeslide_dist, _ = np.histogram(score_timeslides, bins=50, range=(bin_min, bin_max), density=True)
    NBINS=50
    signal_dist, bin_edges = np.histogram(bestvals, bins=NBINS, range=(np.amin(bestvals), np.amax(bestvals)))
    signal_dist = signal_dist / len(bestvals)

    timeslide_dist = make_norm_dist(np.mean(score_timeslides), np.std(score_timeslides))

    #use kde instead
    #print(score_timeslides.shape)
    #assert 0
    #np.save("/home/ryan.raikman/s22/temp2/score_timeslides.npy", score_timeslides)
    #np.save("/home/ryan.raikman/s22/temp2/score_signal.npy", bestvals)

    #kernel_timeslide = gaussian_kde(score_timeslides.T)
    #kernel_signal = gaussian_kde(bestvals.T)

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
    #for i, testval in enumerate(np.linspace(np.amin(signal_dist), np.amax(signal_dist), NBINS)):
    #  Px, Qx = signal_dist, kernel_signal(testval)[0]
    #  KL_div += Px * (np.log(Px)-np.log(Qx))
    #print("signal dist", signal_dist)
    for i, edge in enumerate(bin_edges[:-1]): #left edge
      Px, logQx = signal_dist[i], timeslide_dist(edge)
     # print("Px, Qx", Px, Qx)
      if Px == 0: continue #eh....
      KL_div += Px**2 * (np.log(Px)-logQx)


    #have to integrate to get AUC
    #np.save("/home/ryan.raikman/s22/temp2/FPR.npy", FPRs)
    #np.save("/home/ryan.raikman/s22/temp2/TPR.npy", TPRs)
    #print("FPRs", FPRs)
    #print("min, max", val_min, val_max)
    score3 = (bestvals<val_min).sum()/len(bestvals)
    #print("params", param_vec, "score", TPRs[0], "KL", KL_div* (np.mean(score_timeslides) > np.mean(bestvals)))
    print(f"params {param_vec}, score {TPRs[0]:.3f}, KL_div {KL_div* (np.mean(score_timeslides) > np.mean(bestvals)):.3f}")
    #print("FPR", FPRs[0], FPRs[1])
    #return KL_div * (np.mean(score_timeslides) > np.mean(bestvals))
    return TPRs[0]
    #if FPRs[-1] == 0: #weird bug, not sure how to fix
    #    return 1
    #return -integrate(np.log10(FPRs)[1:], TPRs[1:])


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
  #mu = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
  #u = np.array([-0.105, 0.3799, -0.155, -0.109, -0.05])
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
    #print(mu.shape)
    #assert 0

    cov = np.cov(elite.T) + noise * np.eye(dim)
    #print(cov.shape)
    #assert 0


  return mu_vec, best_sample_vec, mean_sample_vec

def main(timeslide_path, signal_path):
    '''
    Conduct an evolutionary search over the QUAK "weights"
    Inputs:
    timeslide_path: contains timeslide_QUAK.py, timeslide_pearson.npy
    signal_path: contains signal_QUAK.py, signal_pearson.npy
    '''
    ts=time.time()
    timeslide_QUAK = np.load(f"{timeslide_path}/timeslide_QUAK.npy")
    timeslide_pearson = np.load(f"{timeslide_path}/timeslide_pearson.npy")
    timeslide_pearson = np.abs(timeslide_pearson)# + 1e-12

    print(f"timeslides loaded in {time.time()-ts:.2f}",)

    
    p = np.random.permutation(len(timeslide_QUAK))
    cut = 50000000
    timeslide_QUAK = timeslide_QUAK[p][:cut]
    timeslide_pearson = timeslide_pearson[p][:cut]
    if 0:
      test_weights = np.array([ 0, 0, 1,  0.0, 0])
      vals = discriminator(timeslide_QUAK, timeslide_pearson, test_weights)
      sort_perm = np.argsort(vals)

      random_len = 0
      glitch_len = 100000
      random_part_QUAK = timeslide_QUAK[:random_len] #15 million of these
      glitchy_part_QUAK = timeslide_QUAK[sort_perm][:glitch_len]
      random_part_pearson = timeslide_pearson[:random_len] #15 million of these
      glitchy_part_pearson = timeslide_pearson[sort_perm][:glitch_len]

      timeslide_QUAK = np.vstack([random_part_QUAK, glitchy_part_QUAK])
      timeslide_pearson = np.vstack([random_part_pearson, glitchy_part_pearson])

    #print("quak, pearson",timeslide_QUAK.shape, timeslide_pearson.shape)
    #assert 0

    #timeslide_QUAK_flat = np.reshape(timeslide_QUAK, (timeslide_QUAK.shape[0]*timeslide_QUAK.shape[1], timeslide_QUAK.shape[2]))
    #timeslide_pearson_flat = np.reshape(timeslide_pearson, (timeslide_pearson.shape[0]*timeslide_pearson.shape[1], 1))
    ts = time.time()
    signal_QUAK = np.load(f"{signal_path}/signal_QUAK.npy")#[:, 1207-300:1207+300, :]
    signal_pearson = np.load(f"{signal_path}/signal_pearson.npy")#[:, 1207-300:1207+300, :]
    print(f"signal data loaded in {time.time()-ts:.2f}")
    print("Signal shape", signal_QUAK.shape)

    signal_QUAK_flat = np.reshape(signal_QUAK, (signal_QUAK.shape[0]*signal_QUAK.shape[1], signal_QUAK.shape[2]))
    signal_pearson_flat = np.reshape(signal_pearson, (signal_pearson.shape[0]*signal_pearson.shape[1], 1))
    files = [timeslide_QUAK, timeslide_pearson, signal_QUAK_flat, signal_pearson_flat, signal_pearson, signal_QUAK]
    score_parameters(np.array([1, -1, -1, 1, 1]), files)
    score_parameters(np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005]), files)

    print(cmaes(score_parameters, 5, files))
timeslide_path = "/home/ryan.raikman/s22/anomaly/ES_savedir_short/"
#path = "/home/ryan.raikman/s22/anomaly/ES_savedir_BBH_SG/"
signal_path = "/home/ryan.raikman/s22/anomaly/ES_savedir_15SNR/"
main(timeslide_path, signal_path)
    
