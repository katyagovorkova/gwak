import numpy as np

def process(data:np.ndarray):
	return 40 * np.tanh(data/40)