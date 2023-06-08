import os
import time
import torch
import argparse
import numpy as np

from config import (
    SAMPLE_RATE,
    GPU_NAME,
    INIT_SIGMA,
    POPULATION_SIZE,
    N_ELITE,
    NOISE
)
DEVICE = torch.device(GPU_NAME)

def discriminator(values, param_vec):
  return torch.matmul(values, param_vec)

def score_parameters(param_vec, files):
    timeslide_quak, timeslide_pearson, signal_quak_flat, signal_pearson_flat, signal_pearson, signal_quak = files

    scores = discriminator(signal_quak_flat, signal_pearson_flat, param_vec)
    scores = np.reshape(scores, signal_pearson.shape) #should go back to what this originally was

    # take max along axis
    bestvals = np.amin(scores, axis=1)

    score_timeslides = discriminator(timeslide_quak, timeslide_pearson, param_vec)
    score_timeslides = np.reshape(score_timeslides, timeslide_pearson.shape)

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
    assert N_ELITE <= POPULATION_SIZE
    # Initialize the mean and covariance
    mu = np.zeros(dim)
    cov = INIT_SIGMA**2 * np.eye(dim)

    mu_vec = []
    best_sample_vec = []
    mean_sample_vec = []
    for t in range(num_iter):
        children = np.random.multivariate_normal(mu, cov, size=POPULATION_SIZE)
        fitness_score = []
        for child in children:
          fitness_score.append(fn(child, files))
        fitness_score = np.array(fitness_score)

        mean_sample_vec.append(np.average(fitness_score))

        p_sort = np.argsort(fitness_score)[::-1]
        elite = children[p_sort][:N_ELITE]

        best_sample_vec.append(fitness_score[p_sort][0])
        mu_vec.append(mu[:])
        mu = np.average(elite, axis=0)

        cov = np.cov(elite.T) + NOISE * np.eye(dim)

    return mu_vec, best_sample_vec, mean_sample_vec


def main(args):
    '''
    Conduct an evolutionary search over the quak "weights"
    '''
    signal_evals = np.load(f'{args.signal_path}')

    for file_name in os.listdir(f'{args.timeslide_folder_path}'):
       None

    

    mu_vec, best_sample_vec, mean_sample_vec = cmaes(score_parameters, 5, files)
    np.save(args.save_file, best_sample_vec[-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('timeslide_folder_path', type=str,
        help='Path to folder containing timeslide evals')
    parser.add_argument('signal_path', type=str,
        help='Path of signal_evals.npy')
    parser.add_argument('save_file', type=str,
        help='Where to save the best ES parameters')
    args = parser.parse_args()
    main(args)
