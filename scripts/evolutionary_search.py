import numpy as np
import matplotlib.pyplot as plt
from anomaly.evaluation import pearson_QUAK_stats
import time

def integrate(x, y):
    tot = 0
    for i in range(len(x)-1):
        tot += y[i]*(x[i+1]-x[i])
    return tot

def score_parameters(param_vec, files):
    TS_QUAK_TRAIN, TS_PEARSON_TRAIN, timeslide_QUAK, timeslide_pearson, signal_QUAK_flat, signal_pearson_flat, signal_pearson, signal_QUAK = files
    #print("SIGNAL QUAK FLAT", signal_QUAK_flat)
    #np.save("/home/ryan.raikman/s22/temp2/TS_QUAK_data.npy", TS_QUAK_TRAIN)
    discriminator = pearson_QUAK_stats(TS_QUAK_TRAIN, TS_PEARSON_TRAIN, weights=param_vec)

    scores = discriminator(signal_QUAK_flat, signal_pearson_flat)
    scores = np.reshape(scores, signal_pearson.shape) #should go back to what this originally was

    #take max along axis
    #print("12 scores shape", scores.shape)
    bestvals = np.amin(scores, axis=1)
    #print("14, after minning shape", bestvals.shape)

    score_timeslides = discriminator(timeslide_QUAK, timeslide_pearson)
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

    #have to integrate to get AUC
    #np.save("/home/ryan.raikman/s22/temp2/FPR.npy", FPRs)
    #np.save("/home/ryan.raikman/s22/temp2/TPR.npy", TPRs)
    #print("FPRs", FPRs)
    print("params", param_vec, "score", TPRs[0])
    return TPRs[0]
    if FPRs[-1] == 0: #weird bug, not sure how to fix
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
    timeslide_pearson = np.abs(timeslide_pearson) + 1e-12

    print(f"timeslides loaded in {time.time()-ts:.2f}",)

    
    p = np.random.permutation(len(timeslide_QUAK))
    timeslide_QUAK = timeslide_QUAK[p][:4000000]
    timeslide_pearson = timeslide_pearson[p][:4000000]

    #TRAIN_LEN = 20000
    #TS_QUAK_TRAIN = timeslide_QUAK[:TRAIN_LEN]
    #TS_PEARSON_TRAIN = timeslide_pearson[:TRAIN_LEN]
    #timeslide_QUAK = timeslide_QUAK[TRAIN_LEN:]
    #timeslide_pearson = timeslide_pearsons[TRAIN_LEN:]
    ts = time.time()
    TS_QUAK_TRAIN = np.load(f"{timeslide_path}/timeslide_QUAK_TRAIN.npy")
    TS_PEARSON_TRAIN = np.load(f"{timeslide_path}/timeslide_pearson_TRAIN.npy")
    TS_PEARSON_TRAIN = np.abs(TS_PEARSON_TRAIN) + 1e-12
    print(f"timeslides train loaded in {time.time()-ts:.2f}")
    

    #timeslide_QUAK_flat = np.reshape(timeslide_QUAK, (timeslide_QUAK.shape[0]*timeslide_QUAK.shape[1], timeslide_QUAK.shape[2]))
    #timeslide_pearson_flat = np.reshape(timeslide_pearson, (timeslide_pearson.shape[0]*timeslide_pearson.shape[1], 1))
    ts = time.time()
    signal_QUAK = np.load(f"{timeslide_path}/signal_QUAK.npy")
    signal_pearson = np.load(f"{timeslide_path}/signal_pearson.npy")
    print(f"signal data loaded in {time.time()-ts:.2f}")

    signal_QUAK_flat = np.reshape(signal_QUAK, (signal_QUAK.shape[0]*signal_QUAK.shape[1], signal_QUAK.shape[2]))
    signal_pearson_flat = np.reshape(signal_pearson, (signal_pearson.shape[0]*signal_pearson.shape[1], 1))
    files = [TS_QUAK_TRAIN, TS_PEARSON_TRAIN, timeslide_QUAK, timeslide_pearson, signal_QUAK_flat, signal_pearson_flat, signal_pearson, signal_QUAK]
    score_parameters(np.array([1, -1, -1, 1]), files)

    print(cmaes(score_parameters, 4, files))
path = "/home/ryan.raikman/s22/anomaly/ES_savedir"
main(path, path)
    
