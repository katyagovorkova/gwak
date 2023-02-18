import numpy as np

def process(data):
    print("DATA SHAPE", data.shape)
	#individually normalize each segment
    #data -= np.average(data, axis=1)[:, np.newaxis] #doesn't seem right to do
    
    #find the bigger axis to average over
    time_axis = 2
    feature_axis = 1
    if data.shape[1] > data.shape[2]:
        time_axis = 1
        feature_axis = 2
    print("time axis, feature axis:", time_axis, feature_axis)
    
    std_vals = np.std(data, axis=time_axis)
    print("std vals shape", std_vals.shape)
    if time_axis == 2:
        data /= std_vals[:,:, np.newaxis]
    else:
        assert time_axis == 1
        data /= std_vals[:,np.newaxis, :]
    
    print("now data shape:", data.shape)
    #assert False

    #return (data, std_vals)
    #return (data[:, :, np.newaxis], std_vals) #for the LSTM stuff, the extra axis is needed
    return (data, std_vals) #unless using 2 detector streams!