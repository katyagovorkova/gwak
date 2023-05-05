import numpy as np
def make_kernel(N):
    return np.ones(N)/N

def smooth_samples(data, kernel, N_kernel):
    new_len = max(data.shape[1], N_kernel) - min(data.shape[1], N_kernel) + 1
    data_smooth = np.empty((data.shape[0], new_len, data.shape[2]))
    for j in range(len(data)):
        for k in range(data.shape[2]):
            #valid mode takes care of cutting off the edge effects
            data_smooth[j, :, k] = np.convolve(data[j, :, k], kernel, mode='valid')

    return data_smooth
    
def main(data, N_kernel=50):
    kernel = make_kernel(N_kernel)
    return smooth_samples(data, kernel, N_kernel)
