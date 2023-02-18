import numpy as np


def main(data, seg_len, overlap):
    '''
    Function to slice up data into overlapping segments
    seg_len: length of resulting segments
    overlap: overlap of the windows in units of indicies

    assuming that data is of shape (N_samples, axis_to_slice_on, features)
    '''
    print("data segment input shape", data.shape)
    N_slices = (data.shape[1]-seg_len)//overlap
    print("N slices,", N_slices)
    print("going to make it to, ", N_slices*overlap+seg_len)
    data = data[:, :N_slices*overlap+seg_len]
    #assert 0
    
    result = np.empty((data.shape[0], N_slices, seg_len, data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(N_slices):
            start = j*overlap
            end = j*overlap + seg_len
            #print("SHAPES 21", result[i, j, :, :].shape,data[i, start:end, :].shape)
            result[i, j, :, :] = data[i, start:end, :]

    return result

