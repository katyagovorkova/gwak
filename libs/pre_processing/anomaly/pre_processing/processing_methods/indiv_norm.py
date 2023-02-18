import numpy as np

def process(data):
	#individually normalize each segment
    #data -= np.average(data, axis=1)[:, np.newaxis] #doesn't seem right to do
    std_vals = np.std(data, axis=1)
    data /= std_vals[:, np.newaxis]
    #return (data, std_vals)
    return (data[:, :, np.newaxis], std_vals) #for the LSTM stuff, the extra axis is needed