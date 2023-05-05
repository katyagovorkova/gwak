
import numpy as np

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