import numpy as np

def process(data):
	#add across detectors
	return data[:, 0, :] + data[:, 1, :]