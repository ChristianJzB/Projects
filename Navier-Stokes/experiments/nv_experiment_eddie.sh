#!/bin/sh

# Define different values of N for experiments
N_VALUES=(100 150 200 250)  # Adjust as needed

# Define default values for parameters (change as needed)
VERBOSE=""
TRAIN="--train"  # Set empty "" if you want default (False)
DEEPGALA="deepgala"      # Empty means default (False)
NOISE_LEVEL="1e-4"
FEM_MCMC="fem_mcmc"
NN_MCMC="--NN_MCMC"  # Example: enabled
DGALA_MCMC="dgala_mcmc"
DA_MCMC_NN="da_mcmc_nn"
DA_MCMC_DGALA="da_mcmc_dgala"

for N in "${N_VALUES[@]}"; do
    echo "Submitting job for N=$N"

    qsub -N "elliptic_N${N}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_vmem=40G

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python elliptic_experiment.py --N $N $TRAIN $DEEPGALA --noise_level $NOISE_LEVEL $FEM_MCMC $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF

done
