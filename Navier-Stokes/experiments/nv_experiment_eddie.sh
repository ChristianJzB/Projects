#!/bin/sh

# Define different values of N for experiments
N_VALUES=(2 10 50 250 500)  # Adjust as needed

# Define default values for parameters (change as needed)
VERBOSE=""
TRAIN="--train"  # Set empty "" if you want default (False)
HIDDEN_LAYERS="--hidden_layers 3"
NUM_NEURONS= "--num_neurons 300"
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
#$ -l h_rt=24:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Navier-Stokes/experiments/nv_experiment.py --N $N $TRAIN $HIDDEN_LAYERS $NUM_NEURONS $DEEPGALA $NOISE_LEVEL $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF

done
