import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import expon, norm
from scipy.stats import boxcox


N_INTERPOLATE = 10000
def katya_metric(data):
    #sum of two signal losses minus bkg and glitch losses
    return data[:, 0] - 0.5*data[:, 1] - 1.5*data[:, 2] + data[:, 3]

def build_model_from_save(savedir):
    model_params = np.load(f"{savedir}/TRAINED_MODELS/PEARSON_QUAK_STATS/model_params.npy")
    KDE_data = np.load(f"{savedir}/TRAINED_MODELS/PEARSON_QUAK_STATS/pearson_KDE_data.npy")

    pearson_kde = gaussian_kde(KDE_data)
    pearson_kde_interp_points = np.linspace(0, 1, N_INTERPOLATE+1) #1000 points, -1 to 1 is domain of pearson output
    pearson_kde_interp = make_interp_function(pearson_kde_interp_points, pearson_kde)

    pearson_max, pearson_lambda, km_shift, QUAK_bc_lmbda, mu_QUAK, sigma_QUAK = model_params
    return make_full_discriminator(pearson_kde_interp, pearson_max, pearson_lambda, 
                                    km_shift, QUAK_bc_lmbda, mu_QUAK, sigma_QUAK, kde_interp=True)

def make_interp_function(interp_points, pearson_kde):
    #vals = pearson_kde.__call__(interp_points)
    vals = pearson_kde.logpdf(interp_points)
    def kde_interp_func(data):
        data_indicies = (data) * N_INTERPOLATE/1
        data_indicies = data_indicies.astype('int')
        return np.take(vals, data_indicies)

    return kde_interp_func


def make_full_discriminator(pearson_kde, pearson_max, 
                                          pearson_lambda, km_shift, 
                                          QUAK_bc_lmbda, mu_QUAK, sigma_QUAK, kde_interp=False):
    '''
    Given these parameters as fit based on the main function,
    return a function that takes in the QUAK and pearson values 
    and returns a final discriminating metric
    '''
    xs_pearson_test = np.linspace(0, 1, 100)
    if kde_interp:
        pearson_peak = xs_pearson_test[np.argmax(pearson_kde(xs_pearson_test))]
    else:
        pearson_peak = xs_pearson_test[np.argmax(pearson_kde.__call__(xs_pearson_test))]
    
    def full_discriminator(QUAK_vals, pearson_vals):
        #check that data sizes match up
        N = len(QUAK_vals)
        assert len(pearson_vals) == N
        
        #shift the pearson_vals below the peak to the peak
        #this equates to unnaturally low correlation, which is low probability
        #but we don't really care about these, so we just treat them as "average" uncorrelated
        pearson_vals = np.clip(pearson_vals, pearson_peak, None)
        
        if kde_interp:
            pearson_evals = pearson_kde(pearson_vals)#[:, 0]
            if len(pearson_evals.shape) == 2:
                pearson_evals = pearson_evals[:, 0]
        else:
            pearson_evals = pearson_kde.__call__(pearson_vals.T)
            pearson_evals = pearson_kde.logpdf(pearson_vals.T)
        #pearson_evals = np.log10(pearson_evals)
        pearson_evals = -(pearson_evals-pearson_max)
        
        #data that goes into negative is low in correlation(fits well with distribution), so we can just move it
        pearson_vals = np.clip(pearson_evals, 0, None)
        
        #log of exponential distribution function
        log_pearson_density = np.log(pearson_lambda) - pearson_lambda*pearson_vals
        
        #for QUAK
        
        #eval katya metric and shift
        km_QUAK = katya_metric(QUAK_vals)
        km_QUAK = km_QUAK + km_shift
        
        #boxcox transform
        km_QUAK_bc = boxcox(km_QUAK, QUAK_bc_lmbda)
        
        #don't care about values to the right of the mean, those are just "really not signal"
        #so move them to the mean value
        km_QUAK_bc = np.clip(km_QUAK_bc, None, mu_QUAK)
        log_QUAK_density = -np.log(sigma_QUAK * (2*np.pi)**0.5) - 0.5 * ((km_QUAK_bc - mu_QUAK)/sigma_QUAK)**2
        
        #now we have the final result:
        #log_pearson_density, log_QUAK_density
        #equivalent to multiplying densities, now we just add the log values
        const = -np.log(sigma_QUAK * (2*np.pi)**0.5) + np.log(pearson_lambda)
        a, b = -pearson_lambda*pearson_vals,-0.5 * ((km_QUAK_bc - mu_QUAK)/sigma_QUAK)**2
        print("FLOOD", np.mean(a), np.std(a), np.mean(b), np.std(b))
        return a + 4*b
        return log_pearson_density + log_QUAK_density - const #'add it back
        return np.clip(log_pearson_density, -35, None) + np.clip(log_QUAK_density, -35, None)
    
    return full_discriminator
        
def main(train_QUAK, train_pearson, savedir):
    '''
    Callibrates the discrimination metric based on the 
    training set
    '''
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
    
    #build pearson kde
    PEARSON_TRAIN_LEN = 500
    np.save(f"{savedir}/pearson_KDE_data.npy", train_pearson[:PEARSON_TRAIN_LEN].T)
    pearson_kde = gaussian_kde(train_pearson[:PEARSON_TRAIN_LEN].T)
    pearson_kde_interp_points = np.linspace(0, 1, N_INTERPOLATE+1) #1000 points, -1 to 1 is domain of pearson output
    pearson_kde_interp = make_interp_function(pearson_kde_interp_points, pearson_kde)

    #evaluate it on the rest of the data
    #pearson_evals = pearson_kde.__call__(train_pearson[PEARSON_TRAIN_LEN:].T)
    pearson_evals = pearson_kde_interp(train_pearson[PEARSON_TRAIN_LEN:])

    #pearson_evals = np.log10(pearson_evals)

    #shift and flip to get the distribution to be like exponential
    pearson_max = np.max(pearson_evals) #MAINTAIN
    pearson_evals = -(pearson_evals-pearson_max)
    
    #fit exponential to pearson
    _, pearson_lambda = expon.fit(pearson_evals) #the first value is the offset but it is zero as set above
    
    
    #QUAK piece
    
    #eval katya metric on data
    km_QUAK = katya_metric(train_QUAK)
    km_shift = -np.min(km_QUAK) + 1e-6 + 1.5 #MAINTAIN
    km_shift = 5 #
    km_QUAK = km_QUAK + km_shift
    
    #boxcox the data
    km_QUAK_bc, QUAK_bc_lmbda = boxcox(km_QUAK)
    
    #fit a gaussian to the boxcoxed data
    mu_QUAK, sigma_QUAK = norm.fit(km_QUAK_bc)
    
    '''
    now we have collected all the parameters we need:
    PEARSON
    pearson_kde : kde model for pearson correlation values
    pearson_max : shift value for pearson
    pearson_lambda: exponential fitting value for pearson distribution
    
    QUAK
    km_shift : shift value for post-katya metric values
    QUAK_bc_lmbda : lambda value for boxcox transformation
    mu_QUAK, sigma_QUAK : fitting paramteres for the gaussian distribution
    '''
    model_params = np.array([pearson_max, pearson_lambda, km_shift, QUAK_bc_lmbda, mu_QUAK, sigma_QUAK])
    np.save(f"{savedir}/model_params.npy", model_params)
    
    final_discriminator = make_full_discriminator(pearson_kde_interp, pearson_max, 
                                                  pearson_lambda, km_shift, 
                                                  QUAK_bc_lmbda, mu_QUAK, sigma_QUAK, kde_interp=True)
    return final_discriminator