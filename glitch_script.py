import os
import numpy as np
make_files = False
compile_files = True
if make_files:
    N_files = len(os.listdir("./output/omicron/"))
    p = np.random.permutation(N_files)
    files_to_do = np.array(os.listdir("./output/omicron/"))[p]
    for file in files_to_do:
        if file == ".snakemake_timestamp":
            continue
        full_path = "./output/omicron/" + file
        condition = True
        for file_ in ["data_H1.h5", "data_L1.h5", "omicron"]:
            if file_ not in os.listdir(full_path):
                condition=False
        if "glitch_0_3600_4096_200.npy" in os.listdir(full_path):
            print("done", file)
            condition=False
        try:
            if len(os.listdir(f"{full_path}/omicron/training/H1/triggers/")) == 0:
                print("empty, " f"{full_path}/omicron/training/H1/triggers/")
                condition=False
            if len(os.listdir(f"{full_path}/omicron/training/L1/triggers/")) == 0:
                print("empty, " f"{full_path}/omicron/training/L1/triggers/")
                condition=False
        except FileNotFoundError:
            condition=False
        if condition:
            start, stop = [int(elem) for elem in file.split("_")]
            seglen = stop - start
            print("SEGLEN", seglen)
            for j in range(seglen//3600):
                split_start, split_stop = j*3600, (j+1)*3600
                print(f"Working on file, {file}, from {split_start}, to {split_stop}")
                os.system(f"python3 scripts/generate.py {full_path}/ {full_path}/glitch.npy --stype glitch --start {split_start} --stop {split_stop}")
                #assert 0
            
if compile_files:
    datums = []
    for file in os.listdir("./output/omicron/"):
        if file == ".snakemake_timestamp":
            continue
        full_path = "./output/omicron/" + file

        for data_file in os.listdir(full_path):
            if data_file[0] == "g":
                if data_file.split("_")[-1] == "200.npy" and data_file.split("_")[-2] == "4096":
                    datum = np.load(f"{full_path}/{data_file}")
                    print(datum.shape[0])
                    datums.append(datum)

    final = np.concatenate(datums, axis=0)
    print(final.shape)
    np.save("./output/glitch_segs_4096_200.npy", final)