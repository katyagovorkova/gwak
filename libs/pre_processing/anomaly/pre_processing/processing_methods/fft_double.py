import numpy as np

def process(data):
    #add across detectors
    total = np.fft.fft(data)

    #ignore the complex stuff for now, try with just real part
    #return np.real(total)
    #need to come up with a better way of handiling complex numbers, 
    #there is support in keras (or through some github) to do ML with complex numbners
    #fill = np.zeros( (total.shape[0], total.shape[1]*2))
    #fill[:, ::2] = np.real(total)
    #fill[:, 1::2] = np.imag(total)
    fill = np.real(total)
    #print("average, std", np.average(fill), np.std(fill))

    std_value = 5
    normalized = 40 * np.tanh(fill/40) / std_value
    print("average, std", np.average(normalized), np.std(normalized))
    return normalized