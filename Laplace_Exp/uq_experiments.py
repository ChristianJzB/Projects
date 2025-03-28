import sys
import os
import wandb
import torch
import numpy as np

from ml_collections import ConfigDict

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from Base.lla import dgala
from Base.utilities import clear_hooks
from files.training import train_dga
from files.train_elliptic import train_elliptic,generate_data,generate_data_elliptic


def get_burgers_config():
    config = ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "PDEs-training"
    wandb.name = "Burgers_MDNN"
    wandb.tag = None

    # General settings
    config.dga = "burgers"
    config.nn_model = "NN"  # Options: "NN", "WRF", "MDNN"
    config.lambdas = {"burgers":1,"uic":1, "ubcl":1, "ubcr":1}

    # Model-specific settings
    config.model = ConfigDict()
    config.model.input_dim = 3
    config.model.hidden_dim = 40
    config.model.num_layers = 3
    config.model.out_dim = 1
    config.model.activation = "tanh"

    # Training settings
    config.seed = 42
    config.learning_rate = 0.01
    config.decay_rate = 0.95
    config.weight_decay = 1e-3
    # config.alpha = 0.9  # For updating loss weights
    config.epochs = 1200
    config.start_scheduler = 1.0
    # config.weights_update = 250
    config.scheduler_step = 50

    config.samples = 150
    
    return config

def run_burgers_experiment(config_experiment,device):
    model_specific = f"_{config_experiment.nn_model}_s{config_experiment.samples}"
    model_path = f"./models/burgers_" + model_specific + ".pth"
    model_dgala = f"./models/burgers_ll_" + model_specific + ".pth"

    # Step 1: Training Phase (if needed)
    if config_experiment.train:
        print("Running Burgers model with " + model_specific)
        config = get_burgers_config()
        config.wandb.name = "Burgers"+ model_specific
        config.samples = config_experiment.samples  # Update the samples in the config
        pinn_nvs = train_dga(config, device=device)

    # Step 2: Fit deepGALA
    if config_experiment.deepgala:
        print("Starting DeepGaLA fitting for NN" + model_specific)
        nn_surrogate_model = torch.load(model_path)
        nn_surrogate_model.eval()

        data_int,ini_c, left_bc, right_bc = generate_data(size=config_experiment.samples, nparam = 1, seed = 10, burgers=True)

        data_trainig = {"data_fit": {"pde":data_int, "ic_loss":ini_c,"bc_loss":(left_bc, right_bc)}, 
                        "class_method": {"pde": ["burgers_pde"], "ic_loss":["u"], "bc_loss":["u","u"]},
                        "outputs": {"pde": ["burgers"], "ic_loss": ["uic"],"bc_loss":["ubcl","ubcr"]}}

        burgers_model = torch.load(model_path)
        burgers_model.eval()
        llp = dgala(burgers_model, prior_precision=config_experiment.weight_decay)

        llp.fit(data_trainig)
        # Before saving:
        clear_hooks(llp)
        torch.save(llp, model_dgala)


def get_heat_config():
    config = ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "PDEs-training"
    wandb.name = "Heat_MDNN"
    wandb.tag = None

    # General settings
    config.dga = "heat"
    config.nn_model = "NN"  # Options: "NN", "WRF", "MDNN"
    config.lambdas = {"heat":1,"uic":1, "ubcl":1, "ubcr":1}

    # Model-specific settings
    config.model = ConfigDict()
    config.model.input_dim = 3
    config.model.hidden_dim = 20
    config.model.num_layers = 2
    config.model.out_dim = 1
    config.model.activation = "tanh"

    # Training settings
    config.seed = 42
    config.learning_rate = 0.01
    config.decay_rate = 0.95
    config.weight_decay = 1e-3
    # config.alpha = 0.9  # For updating loss weights
    config.epochs = 5000
    config.start_scheduler = 1
    # config.weights_update = 250
    config.scheduler_step = 50

    config.samples = 150
    
    return config


def run_heat_experiment(config_experiment,device):
    model_specific = f"_{config_experiment.nn_model}_s{config_experiment.samples}"
    model_path = f"./models/heat_" + model_specific + ".pth"
    model_dgala = f"./models/heat_ll_" + model_specific + ".pth"

    # Step 1: Training Phase (if needed)
    if config_experiment.train:
        print("Running Heat model with " + model_specific)
        config = get_heat_config()
        config.wandb.name = "Heat"+ model_specific
        config.samples = config_experiment.samples  # Update the samples in the config
        pinn_nvs = train_dga(config, device=device)

    # Step 2: Fit deepGALA
    if config_experiment.deepgala:
        print("Starting DeepGaLA fitting for NN" + model_specific)
        nn_surrogate_model = torch.load(model_path)
        nn_surrogate_model.eval()

        data_int,ini_c, left_bc, right_bc = generate_data(size=config_experiment.samples, nparam = 1, seed = 10, burgers=False)

        data_trainig = {"data_fit": {"pde":data_int, "ic_loss":ini_c,"bc_loss":(left_bc, right_bc)}, 
                        "class_method": {"pde": ["heat_pde"], "ic_loss":["u"], "bc_loss":["u","u"]},
                        "outputs": {"pde": ["heat"], "ic_loss": ["uic"],"bc_loss":["ubcl","ubcr"]}}

        heat_model = torch.load(model_path)
        heat_model.eval()

        llp = dgala(heat_model, prior_precision=config_experiment.weight_decay)

        llp.fit(data_trainig)
        # Before saving:
        clear_hooks(llp)
        torch.save(llp, model_dgala)


def get_elliptic_config():
    config = ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "PDEs-training"
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

    # Config KL parameters
    config.nparameters = 2

    # Training settings
    config.seed = 42
    config.learning_rate = 0.001
    config.decay_rate = 0.95
    config.weight_decay = 1e-3
    # config.alpha = 0.9  # For updating loss weights
    config.epochs = 5000
    config.start_scheduler = 0.5
    # config.weights_update = 250
    config.scheduler_step = 50

    config.samples = 5000
    config.batch_size = 150

    
    return config


def run_elliptic_experiment(config_experiment,device):
    model_specific = f"_{config_experiment.nn_model}_s{config_experiment.samples}"
    model_path = f"./models/elliptic_" + model_specific + ".pth"
    model_dgala = f"./models/elliptic_ll_" + model_specific + ".pth"

    # Step 1: Training Phase (if needed)
    if config_experiment.train:
        print("Running Heat model with " + model_specific)
        config = get_elliptic_config()
        config.wandb.name = "Elliptic" + model_specific
        config.samples = config_experiment.samples  # Update the samples in the config
        pinn_nvs = train_elliptic(config, device=device)

    # Step 2: Fit deepGALA
    if config_experiment.deepgala:
        print("Starting DeepGaLA fitting for NN" + model_specific)
        nn_surrogate_model = torch.load(model_path)
        nn_surrogate_model.eval()

        data_int,left_bc,right_bc = generate_data_elliptic(size=config_experiment.samples) 

        data_trainig = {"data_fit": {"pde":data_int, "left_bc":left_bc,"right_bc":right_bc}, 
                    "class_method": {"pde": ["elliptic_pde"], "left_bc":["u"],"right_bc":["u"]},
                    "outputs": {"pde": ["elliptic"], "left_bc": ["ubcl"],"right_bc":["ubcr"]}}

        heat_model = torch.load(model_path)
        heat_model.eval()

        llp = dgala(heat_model, prior_precision=config_experiment.weight_decay)

        llp.fit(data_trainig)
        # Before saving:
        clear_hooks(llp)
        torch.save(llp, model_dgala)

def experiment():
    config = ConfigDict()
    
    # Train config
    config.train = True
    config.deepgala = True
    config.samples = 250
    config.nn_model = "NN"

    return config

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Burgers & Heat Experiments
    N = [500,1500]
    nn_models = ["NN","MDNN"]

    config_experiment = experiment()

    for model in nn_models:
        config_experiment.nn_model = model
        for nsamples in N:
            config_experiment.samples = nsamples
            run_burgers_experiment(config_experiment,device)
            run_heat_experiment(config_experiment,device)


    N = [150,2500]

    config_experiment = experiment()
    for model in nn_models:
        config_experiment.nn_model = model
        for nsamples in N:
            config_experiment.samples = nsamples
            run_elliptic_experiment(config_experiment,device)

 