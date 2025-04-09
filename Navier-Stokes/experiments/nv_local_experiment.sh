#!/bin/sh

# Define different values of N for experiments
N_VALUES=(2 10 50 250)  # Adjust as needed
N_LAYERS=(1 2 3)  # Define hidden layers

# Define default values for parameters
VERBOSE="--verbose"
KL="--kl 10"
TRAIN=""  # Set empty "" if you want default (False)
DEEPGALA=""  # Empty means default (False)
NOISE_LEVEL="--noise_level 1e-3"
PROPOSAL="--proposal pCN"
NN_MCMC="--nn_mcmc"  # Example: enabled
DGALA_MCMC=""
DA_MCMC_NN=""
DA_MCMC_DGALA=""

for N in "${N_VALUES[@]}"; do
    for L in "${N_LAYERS[@]}"; do
        echo "Working job for N=$N with $L hidden layers"
python Navier-Stokes/experiments/nv_experiment.py $VERBOSE --N $N --hidden_layers $L --num_neurons 300 $KL $TRAIN $DEEPGALA $NOISE_LEVEL $PROPOSAL $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
    done
done
