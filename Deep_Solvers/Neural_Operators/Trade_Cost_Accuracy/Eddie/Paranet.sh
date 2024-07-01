#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#  job name
#$ -N ParaNet              

#  runtime limit of 5 minutes: -l h_rt          
#$ -l h_rt=24:00:00 

# Set working directory to the directory where the job is submitted from:
#$ -cwd

# Request one GPU in the gpu queue:

# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=64G


# Initialise the environment modules
. /etc/profile.d/modules.sh

module load anaconda
conda activate experiments

# Run the program
python ParaNet.py
