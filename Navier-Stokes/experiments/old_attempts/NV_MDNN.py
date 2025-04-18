import wandb
from ml_collections import ConfigDict
import torch

from nv_files.train_nvs import train_pinn_nvs

def get_deepgalerkin_config():
    config = ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "NVs-training-rnd"
    wandb.name = "MDNN_rnd"
    wandb.tag = None

    # General settings
    config.nn_model = "MDNN"  # Options: "NN", "WRF", "MDNN"
    config.lambdas = {"nvs":1, "cond":1, "u0":1, "v0":1, "w0":1}
    config.use_softadapt = False

    # Model-specific settings
    config.model = ConfigDict()
    config.model.input_dim = 3 
    config.model.hidden_dim = 256
    config.model.num_layers = 4
    config.model.out_dim = 2
    config.model.activation = "tanh"

    # Periodic embeddings
    config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":256,"exclude_last_n":0})

    # Navier Stokes Config
    config.nu = 1e-2
    config.time_domain = 5

    # Training settings
    config.seed = 108
    config.learning_rate = 0.001
    config.decay_rate = 0.9
    config.alpha = 0.9  # For updating loss weights
    config.iterations = 15000
    config.chunks = 16
    config.points_per_chunk = 250
 
    config.scheduler_step = 2000
    config.start_scheduler = 0.1
    config.weights_update = 100
    
    return config


# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = get_deepgalerkin_config()

pinn_nvs = train_pinn_nvs(config,device=device)