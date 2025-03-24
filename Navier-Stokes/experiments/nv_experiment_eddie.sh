#!/bin/sh

# Define different values of N for experiments
N_VALUES=(10 25 50 150 250 500)  # Adjust as needed

# Define default values for parameters (change as needed)
VERBOSE=""
TRAIN="--train"  # Set empty "" if you want default (False)
DEEPGALA=""      # Empty means default (False)
NOISE_LEVEL="--noise_level 1e-3"
NN_MCMC="--nn_mcmc"  # Example: enabled
DGALA_MCMC=""
DA_MCMC_NN=""
DA_MCMC_DGALA=""

for N in "${N_VALUES[@]}"; do
    echo "Submitting job for N=$N"

    qsub -N "nv_N${N}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_vmem=40G
#$ -l h_rt=18:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Navier-Stokes/experiments/nv_experiment.py --N $N $TRAIN $DEEPGALA $NOISE_LEVEL $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF

done
