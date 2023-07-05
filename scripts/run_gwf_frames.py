from gwpy.timeseries import TimeSeries
import numpy as np
import os
from astropy import units as u
from evaluate_data import full_evaluation 

def whiten_bandpass_resample(file, sample_rate, bandpass_low, bandpass_high, savedir):
    fnL1 = "/scratch/florent.robinet/BurstBenchmark/L-L1_%s"%file
    fnH1 = "/scratch/florent.robinet/BurstBenchmark/H-H1_%s"%file
    print(fnL1, fnH1)
    
    # Load LIGO data
    strainL1 = TimeSeries.read(fnL1, "L1:STRAIN_BURST_0")
    strainH1 = TimeSeries.read(fnH1, "H1:STRAIN_BURST_0")
    t0 = int(strainL1.t0 / u.s)
    print(t0)

    # Whiten, bandpass, and resample
    strainL1 = strainL1.whiten()
    strainL1 = strainL1.bandpass(bandpass_low, bandpass_high) 
    strainL1 = strainL1.resample(sample_rate)
    strainH1 = strainH1.whiten()
    strainH1 = strainH1.bandpass(bandpass_low, bandpass_high) 
    strainH1 = strainH1.resample(sample_rate)
    
    # split strainL1 and strainH1 into 3 equal-sized parts each for GPU data loading
    total_elements = strainH1.shape[0]
    split_size = total_elements // 3

    strain_H1_1 = np.array(strainH1[:split_size])
    strain_H1_2 = np.array(strainH1[split_size:2*split_size])
    strain_H1_3 = np.array(strainH1[2*split_size:])
    strain_L1_1 = np.array(strainL1[:split_size])
    strain_L1_2 = np.array(strainL1[split_size:2*split_size])
    strain_L1_3 = np.array(strainL1[2*split_size:])

    strain_H1_3 = strain_H1_3[:split_size]
    strain_L1_3 = strain_L1_3[:split_size]

    if 0==1 and 1==0:
        #save files 
        np.save(savedir + "/H1_BurstBenchmark_%s.npy"%int(t0), strain_H1_1)
        np.save(savedir + "/H1_BurstBenchmark_%s.npy"%int(t0+(split_size/4096)), strain_H1_2)
        np.save(savedir + "/H1_BurstBenchmark_%s.npy"%int(t0+(2*split_size/4096)), strain_H1_3)
        np.save(savedir + "/L1_BurstBenchmark_%s.npy"%int(t0), strain_L1_1)
        np.save(savedir + "/L1_BurstBenchmark_%s.npy"%int(t0+(split_size/4096)), strain_L1_2)
        np.save(savedir + "/L1_BurstBenchmark_%s.npy"%int(t0+(2*split_size/4096)), strain_L1_3)

    strain_H1 = np.stack([strain_H1_1, strain_H1_2, strain_H1_3], axis=0)
    strain_L1 = np.stack([strain_L1_1, strain_L1_2, strain_L1_3], axis=0)

    strain = np.stack([strain_H1, strain_L1], axis=1)
    model_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/trained/models/"
    model_paths = []
    for elem in os.listdir(model_path):
        model_paths.append(f"{model_path}/{elem}")  
    evals = []
    for i in range(3):
        evals.append(full_evaluation(strain[i:i+1], model_paths).detach().cpu().numpy())
    evals = np.vstack(np.stack(evals, axis=1)[0])
    print(evals.shape)

    params = np.load("/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/trained/final_metric_params.npy")
    means, stds = np.load("/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/trained/norm_factor_params.npy")

    final = np.dot((evals-means)/stds, params)
    
    np.save("./evals.npy", evals)
    np.save("./final.npy", final)    
    np.save("./strains.npy", np.stack([np.array(strainH1), np.array(strainL1)]))
    assert 0

def main():
    lst = os.listdir('/scratch/florent.robinet/BurstBenchmark/')
    lst = [i[5:] for i in lst if "V1" not in i]
    lst_noduplicates = [*set(lst)]

    sample_rate = 4096
    bandpass_low = 30
    bandpass_high = 1500
    savedir = '/scratch/eric.moreno/BurstBenchmark/Preprocessed'
    for file in lst_noduplicates:
        whiten_bandpass_resample(file, sample_rate, bandpass_low, bandpass_high, savedir)
    
if __name__ == '__main__':
    main()