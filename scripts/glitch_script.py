import os
import numpy as np
make_files = False
compile_files = True
if make_files:
    for file in os.listdir("./output/omicron/"):
        if file == ".snakemake_timestamp":
            continue
        full_path = "./output/omicron/" + file
        if len(os.listdir(full_path)) == 3:
            start, stop = [int(elem) for elem in file.split("_")]
            seglen = stop - start

            for j in range(seglen//3600):
                split_start, split_stop = j*3600, (j+1)*3600
                print(f"Working on file, {file}, from {split_start}, to {split_stop}")
                os.system(f"python3 scripts/generate.py {full_path}/ {full_path}/glitch.npy --stype glitch --start {split_start} --stop {split_stop}")

if compile_files:
    datums = []
    for file in os.listdir("./output/omicron/"):
        if file == ".snakemake_timestamp":
            continue
        full_path = "./output/omicron/" + file

        for data_file in os.listdir(full_path):
            if data_file[0] == "g":
                datum = np.load(f"{full_path}/{data_file}")
                print(datum.shape[0])
                datums.append(datum)

    final = np.concatenate(datums, axis=0)
    print(final.shape)
    np.save("./output/data/glitch_segs.npy", final)