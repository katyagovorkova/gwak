import os
import numpy as np
import time
from anomaly.datagen.injection_2d import main_all3, main_big_bkg_segs
from anomaly.datagen.timeslides import main_timeslides
#def main_X(savedir, N = 20, folder_path = None, prior_file = None): (format for arguments)

#first, get the folders that are available for getting data from
available_folders_full = []
available_folders = []
source_folder_path = "/home/ryan.raikman/s22/anomaly/data2/glitches/"
already_finished_path = "/home/ryan.raikman/s22/anomaly/generated_timeslides/"

save_path = already_finished_path
try:
    os.makedirs(save_path)
except FileExistsError:
    None

for folder in os.listdir(source_folder_path):
    condition = False
    contains = os.listdir(source_folder_path + folder)

    #check if it has both data files   #1238450550 -> 1253946511
    if 'detec_data_H1.h5' in contains and 'H1' in contains:
        if 'detec_data_L1.h5' in contains and 'L1' in contains:
            condition = True
    #check that it has not already been done
    if folder in os.listdir(already_finished_path):
        condition = False
        condition=True

    if condition:
        available_folders_full.append(source_folder_path + folder)
        available_folders.append(folder)

#randomized algo makes it easier to run a bunch at a time
p = np.random.permutation(len(available_folders))
available_folders_full = list(np.array(available_folders_full)[p])
available_folders = list(np.array(available_folders)[p])

do_timeslides = Fakse
if do_timeslides:
    #if 1:
    for i, folder in enumerate(available_folders_full):
        #i=0; folder = available_folders_full[0]
        savedir = save_path + available_folders[i]
        start, end = [int(elem) for elem in available_folders[i].split("_")]
        seg_len = end-start
        N = seg_len//5 #just going to try with more data files...
        print("starting folder", folder.split("/")[-1]) #just the last part of the path is the folder, don't need the rest
        

        main_timeslides(savedir, folder)
        #assert 0
        #main_big_bkg_segs(savedir, folder)

 #   assert 0
if not do_timeslides:
    for i, folder in enumerate(available_folders_full):
        savedir = save_path + available_folders[i]
        start, end = [int(elem) for elem in available_folders[i].split("_")]
        seg_len = end-start
        N = seg_len//5 #just going to try with more data files...
        print("starting folder", folder.split("/")[-1]) #just the last part of the path is the folder, don't need the rest

        main_all3(savedir, N, folder, None)
            
    print("done with all")
