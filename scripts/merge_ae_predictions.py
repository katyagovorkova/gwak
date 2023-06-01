import numpy as np
import argparse


def main(args):

    datasets = dict(
        bbh = np.load(args.bbh),
        sg = np.load(args.sg),
        glitch = np.load(args.glitch),
        background = np.load(args.background)
        )

    np.savez(args.output_file, **datasets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('bbh', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('sg', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('glitch', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('background', help='Path to the Omicron output',
                        type=str)
    parser.add_argument('output_file', help='Path to the Omicron output',
                        type=str)
    args = parser.parse_args()
    main(args)
