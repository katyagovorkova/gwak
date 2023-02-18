import os
'''
Tell the user where the glitch and background data can be found.
Once data generation code exists, it will be under the data
library. However, as of now, just redirect to where I have saved the data
'''

def main(folder_path):	

	#extract all of the classes from the folder
	file_paths = []
	filenames = []
	print(os.getcwd())
	for filename in os.listdir(folder_path):
		file_paths.append(f"{folder_path}/{filename}")
		filenames.append(filename)

	return file_paths, filenames
