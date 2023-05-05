import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import expon, norm
from scipy.stats import boxcox


N_INTERPOLATE = 10000
def katya_metric(data, weights):
    #sum of two signal losses minus bkg and glitch losses
    #return data[:, 0] - 0.5*data[:, 1] - 1.5*data[:, 2] + data[:, 3]
    return weights[0] * data[:, 0] + weights[1] * data[:, 1] + weights[2] * data[:, 2] + weights[3] * data[:, 3]

def build_model_from_save(savedir):
    model_params = np.load(f"{savedir}/TRAINED_MODELS/PEARSON_QUAK_STATS_BBH/v2_model_params.npy")
    weights = np.load(f"{savedir}/TRAINED_MODELS/PEARSON_QUAK_STATS_BBH/v2_model_params_QUAK_weights.npy")
    print("model_params", model_params)
    mu_QUAK, sigma_QUAK = model_params

    return make_full_discriminator(mu_QUAK, sigma_QUAK, weights)

def make_full_discriminator(mu_QUAK, sigma_QUAK, weights):
    '''
    Given these parameters as fit based on the main function,
    return a function that takes in the QUAK and pearson values 
    and returns a final discriminating metric
    '''
    def full_discriminator(QUAK_vals):
        #check that data sizes match up
        N = len(QUAK_vals)
        #assert len(pearson_vals) == N
        #pearson_vals = np.abs(pearson_vals) + 1e-12 #on eval time, this should be okay, any point that's at -0.1 is going to get moved to the peak anyway
        
        #shift the pearson_vals below the peak to the peak
        #this equates to unnaturally low correlation, which is low probability
        #but we don't really care about these, so we just treat them as "average" uncorrelated
        #pearson_vals = np.clip(pearson_vals, pearson_peak, None)
        #if len(pearson_vals.shape)==2:
        #    assert pearson_vals.shape[1] == 1
        #    pearson_vals = pearson_vals[:, 0]
        
        #pearson_bc = boxcox(pearson_vals, pearson_bc_lmbda)
        #pearson_bc = np.clip(pearson_bc, mu_pearson, None) #on the right side of the distribution!

        #already did clipping by moving everything with pearson below mean to mean (roughly)
        #log_pearson_density = -np.log(sigma_pearson * (2*np.pi)**0.5) - 0.5 * ((pearson_bc - mu_pearson)/sigma_pearson)**2
        
        #for QUAK
        
        #eval katya metric and shift
        km_QUAK = katya_metric(QUAK_vals, weights)
        #km_QUAK = km_QUAK + km_shift
        
        #boxcox transform
        #km_QUAK_bc = boxcox(km_QUAK, QUAK_bc_lmbda)
        
        #don't care about values to the right of the mean, those are just "really not signal"
        #so move them to the mean value
        km_QUAK = np.clip(km_QUAK, None, mu_QUAK)
        log_QUAK_density = -np.log(sigma_QUAK * (2*np.pi)**0.5) - 0.5 * ((km_QUAK - mu_QUAK)/sigma_QUAK)**2
        
        #now we have the final result:
        #log_pearson_density, log_QUAK_density
        #equivalent to multiplying densities, now we just add the log values
        #const = -np.log(sigma_QUAK * (2*np.pi)**0.5) -np.log(sigma_pearson * (2*np.pi)**0.5)
        #a, b = -pearson_lambda*pearson_vals,-0.5 * ((km_QUAK_bc - mu_QUAK)/sigma_QUAK)**2
        #print("FLOOD", np.mean(a), np.std(a), np.mean(b), np.std(b))
        #return a + 4*b
        return log_QUAK_density# or dont - const #'add it back
        #return np.clip(log_pearson_density, -35, None) + np.clip(log_QUAK_density, -35, None)
    
    return full_discriminator
        
def main(train_QUAK, savedir=None, weights = np.array([1, -0.5, -1.5, 1])):
    '''
    Callibrates the discrimination metric based on the 
    training set
    '''
    if savedir is not None:
        try:
            os.makedirs(savedir)
        except FileExistsError:
            None

    #assert len(train_pearson) >= 10000
    #shuffle data
    p1 = np.random.permutation(len(train_QUAK))
    #p2 = np.random.permutation(len(train_pearson))
    train_QUAK = train_QUAK[p1]
    #train_pearson = train_pearson[p2]
    #if len(train_pearson.shape)==2:
    #    assert train_pearson.shape[1] == 1
    #    train_pearson = train_pearson[:, 0]

    #split each into half, one for training the boxcox and the other for training the parameters
    #not sure if this is necessary, but why not?
    #train_pearson_lmbda = train_pearson[:len(train_pearson)//2]
    #train_pearson_mu_sig = train_pearson[len(train_pearson)//2:]
    
    #train_QUAK_lmbda = train_QUAK[:len(train_QUAK)//2]
    train_QUAK_mu_sig = train_QUAK
    
    #build pearson model
    #no shift necessary!
    
        
   # pearson_bc, pearson_bc_lmbda = boxcox(train_pearson_lmbda)
    #pearson_bc_mu_sig = boxcox(train_pearson_mu_sig, pearson_bc_lmbda)
    #mu_pearson, sigma_pearson = norm.fit(pearson_bc_mu_sig)
    
    
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
    model_params = np.array([mu_QUAK, sigma_QUAK])
    #print("PARAM NAMES", "pearson_bc_lmbda, mu_pearson, sigma_pearson, \
    #                                km_shift, QUAK_bc_lmbda, mu_QUAK, sigma_QUAK")
    #print("MODEL PARAMS", model_params)
    if savedir is not None:
        np.save(f"{savedir}/v2_model_params.npy", model_params)
        np.save(f"{savedir}/v2_model_params_QUAK_weights.npy", weights)
    
    final_discriminator = make_full_discriminator(mu_QUAK, sigma_QUAK, weights)

    return final_discriminator