import numpy as np
import matplotlib.pyplot as plt
import os
from quak_predict import quak_eval
import torch
model_path = '/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/trained/models/'
model_paths = [model_path  + elem for elem in os.listdir(model_path)]
GPU_NAME = 'cuda:0'
DEVICE = torch.device(GPU_NAME)
tags = ['supernova', 'wnbhf']
tags = ['wnblf']
savepath = '/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/rec_and_quak/anomalies/'
CLASS_ORDER = ['background', 'bbh', 'glitch', 'sg']
try:
    os.makedirs(savepath)
except FileExistsError:
    None
for tag in tags:
    data = np.load(f'/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/data/{tag}_varying_snr.npy')
    snrs = np.load(f'/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/data/{tag}_varying_snr_SNR.npy')

    print(data.shape)
    if tag == 'wnbhf' or tag == 'wnblf':
        windows = np.arange(9300, 9600, 50)
    elif tag == 'supernova':
        windows = np.arange(10800, 11100, 50)
    for window in windows:
        eval_data = data[:, :10, window:window+200]
       # print(eval_data.shape)
        #print(np.std(eval_data, axis=-1).shape)
        eval_data = eval_data / np.std(eval_data, axis=-1)[:, :, None]
        eval_data = np.swapaxes(eval_data, 0, 1)
        #eval_data = np.swapaxes(eval_data, 1, 2)
        #print(eval_data.shape)
       # assert 0

        qeval = quak_eval(torch.from_numpy(eval_data).float().to(DEVICE), model_paths, reduce_loss=False)
        rec = qeval['recreated']['bbh']
        orig = qeval['original']['bbh']

        fig, axs = plt.subplots(5, 2, figsize=(15, 5*6))
        for i in range(5):
            for j in range(2):
                for class_name in CLASS_ORDER:
                    axs[i, j].plot(qeval['recreated'][class_name][i, j, :], label = f'rec, {class_name}')
                axs[i, j].plot(qeval['original']['bbh'][i, j, :], label = 'orig', c='black')

                axs[i, j].legend()

                axs[i, j].set_title(f'{j}, {snrs[i]}')

        plt.savefig(f'{savepath}/{tag}_recreation_{window}.png', dpi=300)



