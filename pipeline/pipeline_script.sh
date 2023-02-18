#!/bin/bash -l

source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda info --envs

conda activate anomaly-pipeline

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python3 pipeline_main.py ~/s22/anomaly/new_architecture_run/new_arch.ini