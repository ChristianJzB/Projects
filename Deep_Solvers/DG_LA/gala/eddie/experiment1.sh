#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#  job name
#$ -N Experiment1              

#  use the directory: 
#$ -o /exports/eddie/scratch/s2113174      

#  runtime limit of 5 minutes: -l h_rt          
#$ -l h_rt=01:00:00 


# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -pe gpu-a100 1

# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=4G


# Initialise the environment modules
. /etc/profile.d/modules.sh

module load cuda
module load anaconda
conda activate gala

# Run the program
python ./Heat_setup.py
