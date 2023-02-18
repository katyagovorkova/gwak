import numpy as np


def main(data, seg_len, overlap):
    '''
    Function to slice up data into overlapping segments
    seg_len: length of resulting segments
    overlap: overlap of the windows in units of indicies

    assuming that data is of shape (N_samples, axis_to_slice_on, features)
    '''
    N_slices = (data.shape[1]-seg_len)//overlap
    data = data[:, :N_slices*overlap+seg_len]
    
    result_shape = (data.shape[0], N_slices, seg_len, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(N_slices):
            start = j*seg_len
            end = j*(seg_len+1)
            result_shape[i, j, :, :] = data[i, start:end, :]

    return result_shape

