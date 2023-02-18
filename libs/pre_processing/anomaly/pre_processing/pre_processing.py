import os
import numpy as np

def load_data(data_paths):
	loaded_data = []
	for path in data_paths:
		loaded_data.append(np.load(path))
	return loaded_data

def fetch_data_path(folder_path):
	#Move to the datagen directory
	from anomaly.datagen import main as datagen_main
	data_paths, names = datagen_main(folder_path)
	
	#Reset directory
	return data_paths, names

def process_import(choice):
	#naive way for now, will improve later
	if choice == "clipping":
		from anomaly.pre_processing.processing_methods.clipping import process as process
	elif choice == "spectrogram":
		from anomaly.pre_processing.processing_methods.spectrogram import process as process
	elif choice == "full": #does nothing
		from anomaly.pre_processing.processing_methods.full import process as process
	elif choice == "add":
		from anomaly.pre_processing.processing_methods.add import process as process
	elif choice == "even":
		from anomaly.pre_processing.processing_methods.even_dim import process as process
	elif choice == "expand_dim":
		from anomaly.pre_processing.processing_methods.expand_dim import process as process
	elif choice == "log_and_norm":
		from anomaly.pre_processing.processing_methods.log_and_norm import process as process
	elif choice == "squeeze":
		from anomaly.pre_processing.processing_methods.squeeze import process as process
	elif choice == "tanh":
		from anomaly.pre_processing.processing_methods.tanh_norm import process as process
	elif choice == "fft":
		from anomaly.pre_processing.processing_methods.fft_double import process as process
	elif choice == "indiv_norm":
		from anomaly.pre_processing.processing_methods.indiv_norm import process as process
	elif choice == "indiv_norm_2d":
		from anomaly.pre_processing.processing_methods.indiv_norm_2d import process as process
	else:
		assert False # invalid data processing choice

	return process

def data_shaping(data):
	'''
	Special case for 1-d inputs going into the dense
	neural network, 
	where (M, 1) must be reformatted into (M, )
	'''
	if len(data.shape) == 3 and data.shape[2] == 1:	
		return np.squeeze(data, axis=2)
	return data

def main(datapath:str, process_choice:str, savedir:str):
	data_paths, names = fetch_data_path(datapath)
	loaded_data = load_data(data_paths)

	#fetch the desired data processing method
	process = process_import(process_choice)

	#pre-process the data
	processed_data = []
	extra = []
	for data in loaded_data:
		if process_choice != "indiv_norm" and process_choice != "indiv_norm_2d":
			processed_data.append(process(data))
		else:
			pdata, stds = process(data)
			processed_data.append(pdata)
			extra.append(stds)

	#make folder if it does not exist
	try:
		os.mkdir(savedir)
	except FileExistsError:
		None

	#save
	for i, data in enumerate(processed_data):
		indiv_name = names[i][:-4] #cut off .npy
		#print(data)
		#print("BEFORE SAVE", len(data))
		print("BEFORE SAVE", data.shape)
		np.save(f"{savedir}/{indiv_name}.npy", data)

	if process_choice == "indiv_norm":
		#make folder if it does not exist
		try:
			os.mkdir(savedir + "../EXTRAS/")
		except FileExistsError:
			None

		for i, data in enumerate(extra):
			indiv_name = names[i][:-4] #cut off .npy
			np.save(f"{savedir}/../EXTRAS/{indiv_name}.npy", data)