import sys
import os
import torch
import wandb
import numpy as np
from ml_collections import ConfigDict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "Elliptic"))  # Explicitly add Elliptic folder


from Base.lla import dgala
from Base.utilities import clear_hooks
from elliptic_files.elliptic_mcmc import EllipticMCMC, EllipticMCMCDA
from elliptic_files.train_elliptic import train_elliptic
from elliptic_files.utilities import generate_noisy_obs,deepgala_data_fit
from elliptic_files.FEM_Solver import FEMSolver

def elliptic_experiment():
    config = ConfigDict()
    config.train = False
    config.nn_model = 5000
    config.KL_expansion = 2

    # DeepGala
    config.deepgala = True

    # Inverse problem parameters
    config.noise_level = 1e-4
    config.num_observations = 6
    config.theta_thruth = np.array([0.098, 0.430])
    config.fem_solver = 50

    # MCMC configuration
    config.NN_samples = False
    config.proposal = "random_walk"
    config.proposal_variance = 1e-3
    config.samples = 10
    config.FEM_samples = False
    config.FEM_h = 50

    # Delayed Acceptance
    config.DA_samples_nn = True
    config.DA_samples_dgala = True
    config.iter_mcmc = 1000000
    config.iter_da = 20000

    return config

# Helper function to set up configuration
def get_deepgalerkin_config():
    config = ConfigDict()

    # Weights & Biases
    config.wandb = ConfigDict()
    config.wandb.project = "Elliptic-training"
    config.wandb.name = "MDNN"
    config.wandb.tag = None

    # Model settings
    config.nn_model = "MDNN"
    config.lambdas = {"elliptic": 1, "ubcl": 1, "ubcr": 1}
    config.model = ConfigDict()
    config.model.input_dim = 3
    config.model.hidden_dim = 20
    config.model.num_layers = 2
    config.model.out_dim = 1
    config.model.activation = "tanh"
    config.nparameters = 2  # KL parameters

    # Weight-Random-Factorization
    #config.reparam = ConfigDict({"type":"weight_fact","mean":1.0,"stddev":0.1})

    # Periodic embeddings
    #config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    #config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":256,"exclude_last_n":100})

    # Training settings
    config.seed = 42
    config.learning_rate = 0.001
    config.decay_rate = 0.95
    config.epochs = 5000
    config.start_scheduler = 0.5
    config.scheduler_step = 50
    config.samples = 5000
    config.batch_size = 150
    # config.alpha = 0.9  # For updating loss weights
    # config.weights_update = 250
    return config

# Helper function to set up MCMC chain
def run_mcmc_chain(surrogate_model, obs_points, sol_test, config_experiment, device):
    mcmc = EllipticMCMC(
        surrogate=surrogate_model,
        observation_locations=obs_points,
        observations_values=sol_test,
        observation_noise=np.sqrt(config_experiment.noise_level),
        nparameters=config_experiment.KL_expansion,
        nsamples=config_experiment.samples,
        proposal_type=config_experiment.proposal,
        step_size=config_experiment.proposal_variance,
        device=device
    )
    return mcmc.run_chain()

# Main experiment runner
def run_experiment(config_experiment,device):
    # Step 1: Training Phase (if needed)
    if config_experiment.train:
        print(f"Running training with {config_experiment.nn_model} samples...")
        config = get_deepgalerkin_config()
        config.wandb.name = f"MDNN_s{config_experiment.nn_model}"
        config.samples = config_experiment.nn_model
        config.nparameters = config_experiment.KL_expansion
        pinn_nvs = train_elliptic(config, device=device)
        print(f"Completed training with {config_experiment.nn_model} samples.")

    # Step 2: Fit deepGALA
    if config_experiment.deepgala:
        data_fit = deepgala_data_fit(config_experiment.nn_model,config_experiment.KL_expansion,device)
        llp = dgala(nn_surrogate_model)
        llp.fit(data_fit)
        llp.optimize_marginal_likelihood()
        clear_hooks(llp)
        torch.save(llp, f"./Elliptic/models/elliptic_dgala_{config_experiment.nn_model}.pth")

    # Step 3: Generate noisy observations for Inverse Problem
    obs_points, sol_test = generate_noisy_obs(obs=config_experiment.num_observations,
                                              std=np.sqrt(config_experiment.noise_level),
                                              theta_t=config_experiment.theta_thruth,
                                              vert=config_experiment.fem_solver)

    # Step 4: MCMC FEM Samples (if enabled)
    fem_path = f'./Elliptic/results/FEM_var{config_experiment.noise_level}.npy'
    if config_experiment.FEM_samples:
        print("Starting MCMC with FEM")
        fem_solver = FEMSolver(np.zeros(2), vert=config_experiment.FEM_h)
        fem_samples = run_mcmc_chain(fem_solver, obs_points, sol_test, config_experiment, device)
        np.save(fem_path, fem_samples[0])
    
    # Step 5: Neural Network Surrogate for MCMC
    if config_experiment.NN_samples:
        print(f"Starting MCMC with NN_s{config_experiment.nn_model}")
        nn_surrogate_model = torch.load(f"./Elliptic/models/MDNN_s{config_experiment.nn_model}.pth")
        nn_surrogate_model.eval()

        nn_samples = run_mcmc_chain(nn_surrogate_model, obs_points, sol_test, config_experiment, device)
        np.save(f'./Elliptic/results/NN_ss{config_experiment.nn_model}_var{config_experiment.noise_level}.npy', nn_samples[0])

    mcmc_da_res = {}
    # Step 6: Delayed Acceptance for NN
    if config_experiment.DA_samples_nn:
        print(f"Starting MCMC-DA with NN_s{config_experiment.nn_model} and FEM")
        nn_surrogate_model = torch.load(f"./Elliptic/models/MDNN_s{config_experiment.nn_model}.pth")
        nn_surrogate_model.eval()

        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h)

        elliptic_mcmcda =  EllipticMCMCDA(nn_surrogate_model,fem_solver, 
                        observation_locations= obs_points, observations_values = sol_test, 
                        observation_noise=np.sqrt(config_experiment.noise_level), 
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        step_size=config_experiment.proposal_variance, device=device )
        
        inner_mh,inner_accepted = elliptic_mcmcda.run_chain(verbose=True)

        mcmc_da_res = {f"NN_{config_experiment.nn_model}": [inner_mh, inner_accepted,inner_accepted/inner_mh]}

    
    # Step 7: Delayed Acceptance for Dgala
    if config_experiment.DA_samples_nn:
        print(f"Starting MCMC-DA with NN_s{config_experiment.nn_model} and FEM")

        llp = torch.load(f"./Elliptic/models/elliptic_dgala_{config_experiment.nn_model}.pth")
        llp.model.set_last_layer("output_layer")  # Re-register hooks

        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h)

        elliptic_mcmcda =  EllipticMCMCDA(llp,fem_solver, 
                        observation_locations= obs_points, observations_values = sol_test, 
                        observation_noise=np.sqrt(config_experiment.noise_level), 
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        step_size=config_experiment.proposal_variance, device=device)
        
        inner_mh,inner_accepted = elliptic_mcmcda.run_chain(verbose=True)

        mcmc_da_res = {f"dgala_{config_experiment.nn_model}": [inner_mh, inner_accepted,inner_accepted/inner_mh]}
    return mcmc_da_res

# Main loop for different sample sizes
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resda = []
    N = [150, 250, 500, 750, 1000, 1500, 2500]
    for nn in N:
        config_experiment = elliptic_experiment()
        fem_path = f'./Elliptic/results/FEM_var{config_experiment.noise_level}.npy'
        config_experiment.nn_model = nn
        config_experiment.FEM_samples = not os.path.exists(fem_path)  # Correct file existence check
        da = run_experiment(config_experiment,device)
        resda.append(da)

if __name__ == "__main__":
    main()