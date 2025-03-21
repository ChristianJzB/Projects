import sys
import os
import torch
import wandb
import argparse
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
    config.verbose = False
    config.train = False
    config.nn_model = 5000
    config.KL_expansion = 2

    # DeepGala
    config.deepgala = False

    # Inverse problem parameters
    config.noise_level = 1e-4
    config.num_observations = 6
    config.theta_thruth = np.array([0.098, 0.430])
    config.fem_solver = 50

    # MCMC configuration
    config.fem_mcmc = False
    config.nn_mcmc = False
    config.dgala_mcmc = False

    config.proposal = "random_walk"
    config.proposal_variance = 1e-3
    config.samples = 1000000
    config.FEM_h = 50

    # Delayed Acceptance
    config.da_mcmc_nn = False
    config.da_mcmc_dgala = False
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
def run_mcmc_chain(surrogate_model, obs_points, sol_test, config_experiment,device):
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
    return mcmc.run_chain(verbose=config_experiment.verbose)

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
        print(f"Starting DeepGaLA fitting for NN_s{config_experiment.nn_model}")
        nn_surrogate_model = torch.load(f"./Elliptic/models/MDNN_s{config_experiment.nn_model}.pth")
        nn_surrogate_model.eval()

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
    
    # Step 4: Neural Network Surrogate for MCMC
    if config_experiment.nn_mcmc:
        print(f"Starting MCMC with NN_s{config_experiment.nn_model}")
        nn_surrogate_model = torch.load(f"./Elliptic/models/MDNN_s{config_experiment.nn_model}.pth")
        nn_surrogate_model.eval()

        nn_samples = run_mcmc_chain(nn_surrogate_model, obs_points, sol_test, config_experiment,device)
        np.save(f'./Elliptic/results/NN_ss{config_experiment.nn_model}_var{config_experiment.noise_level}.npy', nn_samples[0])
    
    # Step 5: DeepGaLA Surrogate for MCMC
    if config_experiment.dgala_mcmc:
        print(f"Starting MCMC with DeepGaLA_s{config_experiment.nn_model}")
        llp = torch.load(f"./Elliptic/models/elliptic_dgala_{config_experiment.nn_model}.pth")
        llp.model.set_last_layer("output_layer")  # Re-register hooks
        nn_samples = run_mcmc_chain(llp, obs_points, sol_test, config_experiment, device)
        np.save(f'./Elliptic/results/dgala_ss{config_experiment.nn_model}_var{config_experiment.noise_level}.npy', nn_samples[0])
    
    # Step 6: MCMC FEM Samples (if enabled)
    fem_path = f'./Elliptic/results/FEM_var{config_experiment.noise_level}.npy'
    if config_experiment.fem_mcmc:
        print("Starting MCMC with FEM")
        fem_solver = FEMSolver(np.zeros(2), vert=config_experiment.FEM_h)
        fem_samples = run_mcmc_chain(fem_solver, obs_points, sol_test, config_experiment, device)
        np.save(fem_path, fem_samples[0])

    # Step 7: Delayed Acceptance for NN
    if config_experiment.da_mcmc_nn:
        print(f"Starting MCMC-DA with NN_s{config_experiment.nn_model} and FEM")
        nn_surrogate_model = torch.load(f"./Elliptic/models/MDNN_s{config_experiment.nn_model}.pth")
        nn_surrogate_model.eval()
        mcmc_da_res_nn = np.empty((0, 3))

        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h)

        elliptic_mcmcda =  EllipticMCMCDA(nn_surrogate_model,fem_solver, 
                        observation_locations= obs_points, observations_values = sol_test, 
                        observation_noise=np.sqrt(config_experiment.noise_level), 
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        step_size=config_experiment.proposal_variance, device=device )
        
        acceptance_res = elliptic_mcmcda.run_chain(verbose=config_experiment.verbose)
        np.save(f'./Elliptic/results/mcmc_da_nn_{config_experiment.nn_model}_{config_experiment.noise_level}.npy', acceptance_res)

    # Step 8: Delayed Acceptance for Dgala
    if config_experiment.da_mcmc_dgala:
        print(f"Starting MCMC-DA with DGALA_s{config_experiment.nn_model} and FEM")
        mcmc_da_res_dgala = np.empty((0, 3))  # 0 rows, 3 columns (for inner_mh, inner_accepted, acceptance_ratio)

        llp = torch.load(f"./Elliptic/models/elliptic_dgala_{config_experiment.nn_model}.pth")
        llp.model.set_last_layer("output_layer")  # Re-register hooks

        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h)

        elliptic_mcmcda =  EllipticMCMCDA(llp,fem_solver, 
                        observation_locations= obs_points, observations_values = sol_test, 
                        observation_noise=np.sqrt(config_experiment.noise_level), 
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        step_size=config_experiment.proposal_variance, device=device)
        
        acceptance_res = elliptic_mcmcda.run_chain(verbose=config_experiment.verbose)
        np.save(f'./Elliptic/results/mcmc_da_dgala_{config_experiment.nn_model}_{config_experiment.noise_level}.npy', acceptance_res)

# Main loop for different sample sizes
def main(verbose,N,train,deepgala, noise_level,fem_mcmc,nn_mcmc,dgala_mcmc,da_mcmc_nn,da_mcmc_dgala, device):
    config_experiment = elliptic_experiment()
    config_experiment.verbose = verbose
    config_experiment.nn_model = N 
    config_experiment.train = train
    config_experiment.deepgala = deepgala
    config_experiment.noise_level = noise_level
    config_experiment.fem_mcmc = fem_mcmc
    config_experiment.nn_mcmc = nn_mcmc
    config_experiment.dgala_mcmc = dgala_mcmc
    config_experiment.da_mcmc_nn = da_mcmc_nn
    config_experiment.da_mcmc_dgala = da_mcmc_dgala
    run_experiment(config_experiment,device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Elliptic Experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--N", type=int, required=True, help="Number of training samples")
    parser.add_argument("--train", action="store_true", help="Train NN")
    parser.add_argument("--deepgala", action="store_true", help="Fit DeepGala")
    parser.add_argument("--noise_level", type=float,default=1e-4,help="Noise level for IP")
    parser.add_argument("--fem_mcmc", action="store_true", help="Run MCMC for FEM")
    parser.add_argument("--nn_mcmc", action="store_true", help="Run MCMC for NN")
    parser.add_argument("--dgala_mcmc", action="store_true", help="Run MCMC for dgala")
    parser.add_argument("--da_mcmc_nn", action="store_true", help="Run DA-MCMC for NN")
    parser.add_argument("--da_mcmc_dgala", action="store_true", help="Run DA-MCMC for DeepGala")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(os.getcwd())

    # Pass all arguments
    main(args.verbose, args.N, args.train, args.deepgala, args.noise_level, args.fem_mcmc, 
         args.nn_mcmc, args.dgala_mcmc, args.da_mcmc_nn, args.da_mcmc_dgala, device)