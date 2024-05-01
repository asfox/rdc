#!/bin/bash
#SBATCH -p production
#SBATCH --array=1 # Replace with number of chunk_n folders
#SBATCH --job-name=fmri_rdc_firstLevel
#SBATCH --output=fmri_rdc_firstLevel_output.log
#SBATCH --error=fmri_rdc_firstLevel_error.log
#SBATCH --time=15:00:00  # Set the maximum run time
#SBATCH --ntasks=1      # Number of CPU tasks needed (1 in this example)
#SBATCH --cpus-per-task=4  # Number of CPU cores per task (4 in this example)
#SBATCH --mem=64000       # Memory per node

#export environment info
env
BASE="/share/foxlab-backedup/RDC/python/rdc/"

# Change to the directory where your Python script is located
cd $BASE
source $BASE/.venv/bin/activate 
which python

# Run your Python script
python $BASE/rdc/run_single_subj.py

