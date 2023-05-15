import os
import numpy as np


def process(data):
    print("DATA SHAPE", data.shape)
    # individually normalize each segment
    # find the bigger axis to average over
    time_axis = 2
    feature_axis = 1
    if data.shape[1] > data.shape[2]:
        time_axis = 1
        feature_axis = 2
    print("time axis, feature axis:", time_axis, feature_axis)

    std_vals = np.std(data, axis=time_axis)
    print("std vals shape", std_vals.shape)
    if time_axis == 2:
        data /= std_vals[:,:, np.newaxis]
    else:
        assert time_axis == 1
        data /= std_vals[:,np.newaxis, :]

    print("now data shape:", data.shape)

    #return (data, std_vals)
    #return (data[:, :, np.newaxis], std_vals) #for the LSTM stuff, the extra axis is needed
    return (data, std_vals) # unless using 2 detector streams!


def main(args):

    # extract all of the classes from the folder
    file_paths = []
    filenames = []
    print(os.getcwd())
    for filename in os.listdir(folder_path):
        file_paths.append(f"{folder_path}/{filename}")
        filenames.append(filename)

    data_paths, names = file_paths, filenames

    loaded_data = []
    for path in data_paths:
        loaded_data.append(np.load(path))

    # pre-process the data
    for i, data in enumerate(loaded_data):
        pdata, stds = process(data)
        indiv_name = names[i][:-4] # cut off .npy
        print('before save', pdata.shape)
        np.save(f'{savedir}/{indiv_name}.npy', pdata)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('train_dir', help='Required output directory for train dataset',
        type=str)
    parser.add_argument('test_dir', help='Required output directory for test dataset',
        type=str)

    # Additional arguments
    parser.add_argument('--test-split', help='Part of the dataset that is going to be used for training',
        type=float, default=0.9)
    parser.add_argument('--data-path', help='Where is the data to do train/test split on',
        type=str)
    args = parser.parse_args()
    main(args)