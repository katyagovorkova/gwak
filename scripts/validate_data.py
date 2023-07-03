import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import VERSION


def main(args):

    fig, ax = plt.subplots(len(args.datasets), figsize=(15, len(args.datasets)*5))

    with open(f'data/{VERSION}/info.txt', 'w') as f:

        for i, dataset_file in enumerate(args.datasets):

            data = np.load(dataset_file)
            dataname = os.path.basename(dataset).strip('.npy')

            if 'data' in data.keys():
                dataset = data['data']

                print(f'{dataset_file}')
                print(f'Dataset {dataname} shape is {dataset.shape}')
                f.write(f'{dataset_file} \n')
                f.write(f'Dataset {dataname} shape is {dataset.shape} \n')

                ax[i].plot(dataset[0].flatten())
                ax[i].set_title(f'{dataset_file}')
            else:

                for data_t in ['clean', 'noisy']:
                    dataset = data[data_t]
                    print(f'{dataset_file}')
                    f.write(f'{dataset_file} {data_t}\n')
                    f.write(f'Dataset {dataname} shape is {dataset.shape} \n')

                    ax[i].plot(dataset.flatten())
                    ax[i].set_title(f'{dataset}')

    fig.savefig(f'data/{VERSION}/info.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('datasets', nargs='+', type=str,
        help='Path to the Omicron output')
    args = parser.parse_args()
    main(args)
