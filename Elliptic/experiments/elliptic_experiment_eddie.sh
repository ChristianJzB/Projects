#!/bin/sh

# Define different values of N for experiments
N_VALUES=(250 500 750 1000 1500 2500)  # Adjust as needed

# Define default values for parameters (change as needed)
VERBOSE=""
TRAIN="--train"  # Set empty "" if you want default (False)
DEEPGALA="--deepgala"      # Empty means default (False)
NOISE_LEVEL="--noise_level 1e-4"
FEM_MCMC="--fem_mcmc"
NN_MCMC="--nn_mcmc"  # Example: enabled
DGALA_MCMC="--dgala_mcmc"
DA_MCMC_NN="--da_mcmc_nn"
DA_MCMC_DGALA="--da_mcmc_dgala"

# Flag to check if FEM_MCMC has been added
FEM_MCMC_ADDED=false

for N in "${N_VALUES[@]}"; do
    echo "Submitting job for N=$N"

    # Add FEM_MCMC only for the first iteration
    if [ "$FEM_MCMC_ADDED" = false ]; then
        FEM_MCMC_FLAG="--fem_mcmc"
        FEM_MCMC_ADDED=true
    else
        FEM_MCMC_FLAG=""
    fi

    qsub -N "elliptic_N${N}" <<EOF
#!/bin/bash
#$ -cwd
#$ -q gpu
#$ -l gpu=1 
#$ -l h_vmem=40G
#$ -l h_rt=5:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Elliptic/experiments/elliptic_experiment.py --N $N $TRAIN $DEEPGALA $NOISE_LEVEL $FEM_MCMC_FLAG $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF

done
