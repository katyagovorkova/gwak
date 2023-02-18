import numpy as np

def process(data:np.ndarray):
    # make sure that all data dimensions are of even size, if needbe pad with zeros
    data_shape = data.shape[1:]

    new_shape = [len(data)]
    for dim in data_shape:
        if dim % 2 == 0:
            new_shape.append(dim)
        else:
            new_shape.append(dim+1)
    
    new_arr = np.zeros(new_shape)

    #stuck here, was going to make it fully general above but will just keep it to 2D now
    new_arr[:, :data_shape[0], :data_shape[1]] = data


    return np.expand_dims(new_arr, axis = 3)
