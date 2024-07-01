#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#  job name
#$ -N operators              

#  runtime limit of 5 minutes: -l h_rt          
#$ -l h_rt=05:00:00 

# Set working directory to the directory where the job is submitted from:
#$ -cwd

# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -pe gpu-a100 1

# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=32G


# Initialise the environment modules
. /etc/profile.d/modules.sh

module load cuda/12.1
module load anaconda
conda activate experiments

# Run the program
python FOnet.py
