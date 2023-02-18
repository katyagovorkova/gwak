import numpy as np

def process(data:np.ndarray):
    data = np.log10(data)
    print("AVERAGE VALUE", np.average(data))
    data -= -23
    print(np.std(data))
    data /= 0.5
    return data