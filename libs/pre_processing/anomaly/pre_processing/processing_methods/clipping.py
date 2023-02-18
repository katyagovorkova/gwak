import numpy as np

def process(data:np.ndarray):
	#assuming that data is 3 seconds at 2048 Hz
	fs = 4096
	clip_size = 0.5 #0.5 seconds on either end
	clip_size = int(clip_size*fs)

	center = data.shape[1]//2

	return data[:, center-clip_size:center+clip_size, np.newaxis]