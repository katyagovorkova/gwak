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

        for i, dataset in enumerate(args.datasets):

            data = np.load(dataset)
            dataname = os.path.basename(dataset).strip('.npy')

            print(f'{dataset}')
            print(f'Dataset {dataname} shape is {data.shape}')
            f.write(f'{dataset} \n')
            f.write(f'Dataset {dataname} shape is {data.shape} \n')

            ax[i].plot(data[0].flatten())
            ax[i].set_title(f'{dataset}')

    fig.savefig(f'data/{VERSION}/info.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('datasets', nargs='+', type=str,
        help='Path to the Omicron output')
    args = parser.parse_args()
    main(args)
