#!/bin/sh

# Define different values of N for experiments
N_VALUES=(2 10 50 250 500)  # Adjust as needed
N_LAYERS=(1 2 3)  # Define hidden layers

# Define default values for parameters
VERBOSE=""
TRAIN=""  # Set empty "" if you want default (False)
DEEPGALA=""  # Empty means default (False)
NOISE_LEVEL="--noise_level 1e-2"
NN_MCMC="--nn_mcmc"  # Example: enabled
DGALA_MCMC=""
DA_MCMC_NN="--da_mcmc_nn"
DA_MCMC_DGALA=""

for N in "${N_VALUES[@]}"; do
    for L in "${N_LAYERS[@]}"; do
        echo "Submitting job for N=$N with $L hidden layers"

        qsub -N "nv_N${N}_L${L}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_vmem=40G
#$ -l h_rt=15:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Navier-Stokes/experiments/nv_experiment.py --N $N --hidden_layers $L --num_neurons 300 $TRAIN $DEEPGALA $NOISE_LEVEL $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF

    done
done

