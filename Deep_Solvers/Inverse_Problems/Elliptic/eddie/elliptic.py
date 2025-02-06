import sys
import os
# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import wandb
from ml_collections import ConfigDict
import torch

from elliptic_files.train_elliptic import train_elliptic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_deepgalerkin_config():
    config = ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "Elliptic-training"
    wandb.name = "MDNN"
    wandb.tag = None

    # General settings
    config.nn_model = "MDNN"  # Options: "NN", "WRF", "MDNN"
    config.lambdas = {"elliptic":1, "ubcl":1, "ubcr":1}

    # Model-specific settings
    config.model = ConfigDict()
    config.model.input_dim = 3
    config.model.hidden_dim = 20
    config.model.num_layers = 2
    config.model.out_dim = 1
    config.model.activation = "tanh"

    # Weight-Random-Factorization
    #config.reparam = ConfigDict({"type":"weight_fact","mean":1.0,"stddev":0.1})

     # Periodic embeddings
    #config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    #config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":256,"exclude_last_n":100})

    # Config KL parameters
    config.nparameters = 2

    # Training settings
    config.seed = 42
    config.learning_rate = 0.001
    config.decay_rate = 0.95
    # config.alpha = 0.9  # For updating loss weights
    config.epochs = 5000
    config.start_scheduler = 0.5
    # config.weights_update = 250
    config.scheduler_step = 50

    config.samples = 5000
    config.batch_size = 150
    return config


# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = get_deepgalerkin_config()

N = [150,250,500,750,1000,1500,2500]

# Loop through each sample size
for samples in N:
    print(f"Running training with {samples} samples...")
    config = get_deepgalerkin_config()
    config.wandb.name = f"MDNN_s{samples}"
    config.samples = samples  # Update the samples in the config
    pinn_nvs = train_elliptic(config, device=device)
    print(f"Completed training with {samples} samples.")