import numpy as np
from scipy import signal

def process(data):
	fs=2048
	#do individually, by detector
	_, _, Sxx_det1 = signal.spectrogram(data[:, 0, :], fs)
	_, _, Sxx_det2 = signal.spectrogram(data[:, 1, :], fs)
	Sxx = np.swapaxes(np.stack([Sxx_det1, Sxx_det2]), 0, 1)
	
	#now should have shape (N, 2, Sxx_dim1, Sxx_dim2)
	#do padding to match inputs for CNN3D
	pad_x = abs(Sxx.shape[2]%-4)
	pad_y = abs(Sxx.shape[3]%-4)

	#doing ones so that log10 brings it to zero
	Sxx_new = np.ones((
		Sxx.shape[0],
		Sxx.shape[1],
		Sxx.shape[2] + pad_x,
		Sxx.shape[3] + pad_y
	))
	Sxx_new[:, :, :Sxx.shape[2], :Sxx.shape[3]] = Sxx
	Sxx = Sxx_new

	#going to change to match the CNN3D
	Sxx = np.swapaxes(Sxx, 2, 3)
	#(N, 2, Sxx_dim2, Sxx_dim1)
	Sxx = np.swapaxes(Sxx, 1, 3)
	#(N, Sxx_dim1, Sxx_dim2, 2)
	return np.log10(Sxx)