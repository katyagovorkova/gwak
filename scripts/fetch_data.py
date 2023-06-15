import os
import numpy as np
import argparse

from gwpy.timeseries import TimeSeries

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import CHANNEL


def main(args):

    segments = np.load(args.intersections)
    for segment in segments:

        data = TimeSeries.get(f'{args.site}:{CHANNEL}', segment[0], segment[1])
        data.write(f'{args.folder_path}/{segment[0]}_{segment[1]}/data.h5')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('folder_path', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('intersections', help='Path to the intersections file',
                        type=str)

    parser.add_argument('--site', help='Where to save the file with injections',
                        type=str, choices=['L1', 'H1'])
    args = parser.parse_args()
    main(args)
