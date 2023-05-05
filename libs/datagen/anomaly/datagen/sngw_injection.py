import os
import numpy as np
import time
import os
from anomaly.datagen.injection_2d import main_SNGW

available_folders_full = []
available_folders = []
source_folder_path = "/home/ryan.raikman/s22/anomaly/old/data2/glitches/"
#already_finished_path = "/home/ryan.raikman/s22/anomaly/injection_save_2detec/"
for folder in os.listdir(source_folder_path):
    condition = False
    contains = os.listdir(source_folder_path + folder)
    if 'detec_data_H1.h5' in contains and 'H1' in contains:
        if 'detec_data_L1.h5' in contains and 'L1' in contains:
            #print(f"passed condition for folder: {folder}")
            condition = True
   # if folder in os.listdir(already_finished_path):
    #    #print("found one that was already done,", folder)
    #    condition = False
    #if condition:
    #    print("This one made it through the check", folder)
    if condition:
        available_folders_full.append(source_folder_path + folder)
        available_folders.append(folder)

p = np.random.permutation(len(available_folders))
available_folders_full = list(np.array(available_folders_full)[p])
available_folders = list(np.array(available_folders)[p])

#print("full", available_folders_full)
best = 1000000
bestval = None
for elem in available_folders_full:
    a, b = [int(hi) for hi in elem.split("/")[-1].split("_")]
    elemval = b-a
    if elemval < best:
        best = elemval
        bestval = elem

print("bestval", bestval)
save_path = "/home/ryan.raikman/s22/generated_SNs_noise_5/"
polarization_path = "/home/ryan.raikman/s22/SN_polarizations/"
if 0:
    for folder in os.listdir(polarization_path):
        print("working on folder,", folder)
        try:
            os.makedirs(save_path + folder)
        except FileExistsError:
            None
        main_SNGW(savedir=save_path + folder, folder_path=bestval, polarization_files=polarization_path + folder)
folder = "Powell_2020/"
try:
    os.makedirs(save_path + folder)
except FileExistsError:
    None
main_SNGW(savedir=save_path + folder, folder_path=bestval, polarization_files=polarization_path + folder)