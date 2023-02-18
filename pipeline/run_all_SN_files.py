import os
#os.system(str) to run a command line file

#os.system("cd /home/ryan.raikman/s22/forks/gw-anomaly/pipeline")
#os.system("conda activate anomaly-pipeline")

def cmd(ini_file):
    return f"python3 pipeline_main.py ~/s22/anomaly/SN_RUNS_1_22_regular/ini_files/{ini_file}"

ini_files = os.listdir("/home/ryan.raikman/s22/anomaly/SN_RUNS_1_22_regular/ini_files")

for ini_file in ini_files:
    line = cmd(ini_file)
    os.system(line)