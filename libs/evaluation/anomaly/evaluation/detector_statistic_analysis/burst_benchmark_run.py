import numpy as np 
import os
import matplotlib.pyplot as plt
path = "/home/eric.moreno/QUAK/BurstBenchmark_Preprocessed/Output/"
weights = np.array([ 0.47291488, -0.19085281, -0.14985232,  0.52350923, -0.18645005])
try:
    os.makedirs("/home/ryan.raikman/s22/temp7/")
except FileExistsError:
    None
evals = dict()
N_file = len(os.listdir(path))
files = []
for i, file in enumerate(os.listdir(path)):
    print(f"iteration: {i} out of {N_file}", end = "\r")
    data = np.load(f"{path}/{file}")[50:-50] #clip edge effects
    data = data[np.isfinite(data[:, 0])] #remove nans
    start = int(file.split("-")[1])
    
    N=20
    new_len = max(data.shape[0], N) - min(data.shape[0], N) + 1
    data2 = np.zeros((new_len, 5))
    kernel = np.ones(N)/N
    for i in range(5):
        try:
            data2[:, i] = np.convolve(data[:, i], kernel, mode="valid")
        except ValueError:
            None
    evals[start] = np.dot(data2, weights)
    np.save(f"/home/ryan.raikman/s22/temp7/{start}_v2.npy", evals[start])
    files.append(np.dot(data2, weights))

print(np.concatenate(files).shape)
plt.figure(figsize=(8, 5))
plt.hist(np.concatenate(files), bins=100)
plt.yscale("log")
plt.savefig(f"/home/ryan.raikman/s22/temp7/.hist.png", dpi=300)



    