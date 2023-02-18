import os
import numpy as np
import argparse
import shutil
#import anomaly
#print("imported full anomaly library")
#import anomaly.training
#print("imported anomaly training sublibary")
def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        None
import os
'''
take in .ini file, compare values with default value
dictionary, and save that to a .json file
'''

def float_check(value):
    #could make this more concise with "1" line
    if "." in value and len(value.split(".")) == 2:
        if (value.split(".")[0]).isdigit():
            return True
    return False

def ini_reader_main(ini_path):
    #None indicates required value
    default_values = {
        # [paths]
        "data_path" : None,
        "save_path" : None,
        "runthrough_path" : None,
        
        # [data-hyperparameters]
        "data_preprocessing_method" : "full",
        "test_split" : 0.9,

        # [training]
        "batch_size" : 50,
        "epochs" : 5,
        "network_type" : "dense",
        "bottleneck" : 5,

        # [plotting]
        "make_QUAK_plot" : True,
        "make_LS_plot" : True,

        # [steps]
        "pre_processing_step" : True,
        "training_step" : True,
        "train_LS" : True,
        "eval_data_prediction_step" : True,
        "train_data_predict" : False,
        "eval_plotting_step" : True,
        "roc_plotting_step" : True,
        "ae_prediction_step" : True,
        "data_runthrough_step" : True,
        "kde_runthrough_step" : False,
        "kde_plotting_step" : False,
        "nn_quak_runthrough_step" : False,

        # [extra]
        "data_shape_logging" : False

    }

    with open(ini_path, "r") as f:
        for line in f.readlines():
            if line[0] in ["[", " ", "#", "\n", "\\"]:
                #skipping the line
                continue

            var, value = line.split("=")
            value.replace("\n", "")

            #support for a = 5 and a=5
            var = var.strip()
            value = value.strip()

            if var not in default_values:
                print("INVALID VAR", var)
                assert False #invalid variable

            #naive method for getting to type
            #can improve later

            #bool check
            if value.lower() in ["true", "false"]:
                if value.lower() == "true":
                    default_values[var] = True
                else:
                    default_values[var] = False

            #integer check
            elif value.isdigit():
                default_values[var] = int(value)

            #float check
            elif float_check(value): #this is a terrible way to do this
                default_values[var] = float(value)


            else: #, just a string type, no args so far have tuple, list, etc
                default_values[var] = value

    #final check to make sure necessary values are filled in
    for key in default_values:
        if default_values[key] == None:
            print(key, default_values[key])
            print(default_values)
            assert False #required argument not satified

    return default_values

parser = argparse.ArgumentParser()
parser.add_argument('ini_path', type=str)
args = parser.parse_args()


values = ini_reader_main(args.ini_path)
V = values
savedir = V['save_path']

mkdir(savedir)

#save a copy of the .ini file
if not os.path.exists(f"{savedir}/args.ini"):
    shutil.copyfile(args.ini_path, f"{savedir}/args.ini")

if V['data_shape_logging']:
    mkdir(f"{savedir}/LOG/")
    shape_log_file = f"{savedir}/LOG/shape_log.log"
    with open(shape_log_file, "w") as f:
        f.write("log tracking the progressive data shapes \n")

if V['pre_processing_step']:
    from anomaly.pre_processing import main as pre_processing_main

    mkdir(savedir+"/DATA/TRAIN/")
    mkdir(savedir+"/DATA/TEST/")
    for file in sorted(os.listdir(V['data_path'])):
        
        #split into training and testing data, process
        data = np.load(V['data_path']+f"/{file}")
        #data = data[:2804]
        print("right after loading", data.shape)
        p = np.random.permutation(len(data))
        data = data[p]

        #going to add data size reduction just for testing
        data_reduction = False
        if data_reduction:
            data = data[:1000]

        print("before pre-processing", data.shape)
        split_index = int(V['test_split'] * len(data))


        if V['data_shape_logging']:
            with open(shape_log_file, "a") as f:
                f.write(f"before preprocessing START file: {file} \n")
                f.write(f"train: {data[:split_index].shape} \n")
                f.write(f"test: {data[split_index:].shape} \n")
                f.write(f"before preprocessing END file: {file} \n")

        np.save(f"{savedir}/DATA/TRAIN/{file}", data[:split_index])
        np.save(f"{savedir}/DATA/TEST/{file}", data[split_index:])

    pre_processing_main(f"{savedir}/DATA/TRAIN/",
                        V['data_preprocessing_method'],
                        f"{savedir}/DATA/TRAIN_PROCESS/")

    pre_processing_main(f"{savedir}/DATA/TEST/",
                        V['data_preprocessing_method'],
                        f"{savedir}/DATA/TEST_PROCESS/")

    if V['data_shape_logging']:
        with open(shape_log_file, "a") as f:
            f.write("post preprocessing, just TRAIN, START \n")
            for file in sorted(os.listdir(f"{savedir}/DATA/TRAIN_PROCESS/")):
                x = np.load(f"{savedir}/DATA/TRAIN_PROCESS/{file}")
                f.write(f"file: {file}, shape: {x.shape} \n")
            f.write("post processing, just TRAIN, END \n")

if os.path.exists(f"{savedir}/DATA/TRAIN_PROCESS/"):
    class_labels = []
    for file in sorted(os.listdir(f"{savedir}/DATA/TRAIN_PROCESS/")):
        class_labels.append(file[:-4]) #cut off .npy

print("Training step choice:", V['training_step'])
if V['training_step']:
    from anomaly.training import train_LS_main, train_QUAK_main

    #load training data
    datae = []
    
    for file in sorted(os.listdir(f"{savedir}/DATA/TRAIN_PROCESS/")):
        data = np.load(f"{savedir}/DATA/TRAIN_PROCESS/" + file)
        print(f"loaded data from file: {file}, shape is: {data.shape}")
        datae.append(data)
        print("after process, indiv shape", data.shape)
    trained_model_path = f"{savedir}/TRAINED_MODELS/"

    #options for model are [LSTM, CNN, CNN3D, dense]
    if V['train_LS']:
        train_LS_main(datae, 
                    V['network_type'], 
                    f"{trained_model_path}/LS/",
                    V['batch_size'],
                    V['epochs'],
                    V['bottleneck'],
                    f"{savedir}/DATA/TEST_PROCESS/")


        #don't want it to run through the rest of the analysis, since something will break
        assert False
    else:
        train_QUAK_main(datae, 
                    V['network_type'], 
                    f"{trained_model_path}/QUAK/",
                    V['batch_size'],
                    V['epochs'],
                    V['bottleneck'],
                    class_labels)

if V['eval_data_prediction_step']:
    from anomaly.evaluation import predict_main

    datae = []
    #class_labels = []
    for file in sorted(os.listdir(f"{savedir}/DATA/TEST_PROCESS/")):
        datae.append(np.load(f"{savedir}/DATA/TEST_PROCESS/" + file))
        #class_labels.append(file[:-4])
    
    predict_main(datae, 
                f"{savedir}/TRAINED_MODELS/",
                f"{savedir}/DATA_PREDICTION/TEST/",
                class_labels,
                V['train_LS'])

    if V['train_data_predict']: #turning this off since it takes a while
        #going to do the training data as well, potentially for training KDE
        datae = []
        #class_labels = []
        for file in sorted(os.listdir(f"{savedir}/DATA/TRAIN_PROCESS/")):
            datae.append(np.load(f"{savedir}/DATA/TRAIN_PROCESS/" + file))
            #class_labels.append(file[:-4])
        
        predict_main(datae, 
                    f"{savedir}/TRAINED_MODELS/",
                    f"{savedir}/DATA_PREDICTION/TRAIN/",
                    class_labels,
                    V['train_LS'])

if V['eval_plotting_step']:
    from anomaly.evaluation import plotting_main

    #class_labels = os.listdir(f"{savedir}/TRAINED_MODELS/QUAK/")
    plotting_main(f"{savedir}/DATA_PREDICTION/TEST/",
                  f"{savedir}/PLOTS/",
                  class_labels,
                  True,
                  V['train_LS'])   
    if 0: 
        plotting_main(f"{savedir}/DATA_PREDICTION/TRAIN/",
                    f"{savedir}/PLOTS/TRAIN/",
                    class_labels,
                    True,
                    V['train_LS']) 

if V['roc_plotting_step']:
    from anomaly.evaluation import kde_main

    #class_labels = os.listdir(f"{savedir}/TRAINED_MODELS/QUAK/")
    kde_main(f"{savedir}/DATA_PREDICTION/TRAIN/",
            f"{savedir}/DATA_PREDICTION/TEST/",
            f"{savedir}/PLOTS/",
            class_labels,
            V['train_LS'])

if V['ae_prediction_step']: 
    from anomaly.evaluation import autoencoder_prediction_main
    autoencoder_prediction_main(savedir, V['train_LS'])

if V['nn_quak_runthrough_step']:
    from anomaly.evaluation import nn_quak_runthrough_main
    nn_quak_runthrough_main(savedir)

kde_models = None
if V['kde_runthrough_step']:
    from anomaly.evaluation import kde_runthrough_main
    #do it for all data classes
    kde_models = kde_runthrough_main(savedir)

if V['kde_plotting_step']:
    from anomaly.evaluation import kde_plotting_main
    kde_plotting_main(f"{savedir}/DATA_PREDICTION/TEST/",
                  f"{savedir}/PLOTS/",
                  class_labels)   

if V['data_runthrough_step']:
    from anomaly.evaluation import runthrough_main
    runthrough_main(V['runthrough_path'], savedir, 5, kde_models, NN_quak=True)




