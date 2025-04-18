import sys
import os
import torch
import wandb
import argparse
import numpy as np
from ml_collections import ConfigDict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "Navier-Stokes"))  # Explicitly add Elliptic folder


from Base.lla import dgala
from Base.utilities import clear_hooks
from nv_files.nv_mcmc import NVMCMC, NVMCMCDA
from nv_files.train_nvs import train_vorticity_dg
from nv_files.utilities import generate_noisy_obs,deepgala_data_fit

def nv_experiment():
    config = ConfigDict()
    config.verbose = False

    # Train config
    config.train = False

    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "Experiment_NV"
    wandb.name = "Vorticity"
    wandb.tag = None

    # General settings
    config.nn_model = "MDNN"  # Options: "NN", "WRF", "MDNN"
    config.lambdas = {"nvs":1, "cond":1, "w0":1, "phi":1}

    # Model-specific settings
    config.model = ConfigDict()
    config.model.input_dim = 3 + 2
    config.model.hidden_dim = 200
    config.model.num_layers = 4
    config.model.out_dim = 2
    config.model.activation = "tanh"

    # Weight-Random-Factorization
    #config.reparam = ConfigDict({"type":"weight_fact","mean":1.0,"stddev":0.1})

     # Periodic embeddings
    config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":200,"exclude_last_n":2})

    # Navier Stokes Config
    config.nu = 1e-2
    config.time_domain = 2

    # Training settings
    config.seed = 108
    config.learning_rate = 0.001
    config.decay_rate = 0.9
    config.alpha = 0.9  # For updating loss weights
    config.iterations = 5000
    config.start_scheduler = 0.1
    config.weights_update = 250
    config.scheduler_step = 1000 #2000

    config.chunks = 16
    config.points_per_chunk = 50
    #config.batch_ic = 16*

    # For deep Galerkin- initial conditions
    config.d = 5
    config.tau = np.sqrt(2)
    config.KL_expansion =  1
    config.dim_initial_condition = 128
    config.samples_size_initial = 1000

    # DeepGala
    config.deepgala = False

    # Inverse problem parameters
    config.noise_level = 1e-3
    config.num_observations = 6

    # MCMC configuration
    config.nn_mcmc = False
    config.dgala_mcmc = False

    config.proposal = "random_walk"
    config.uniform_limit = 2
    config.proposal_variance = 5e-1
    config.samples = 1_000_000
    
    # Num Solver Config
    config.fs_n = 128
    config.fs_T = 2
    config.fs_steps = 5e-4

    # Delayed Acceptance
    config.da_mcmc_nn = False
    config.da_mcmc_dgala = False
    config.iter_mcmc = 1_000
    config.iter_da = 5_000

    return config

# # Helper function to set up configuration
# def get_vorticity_train_config():
#     config = ConfigDict()

#     # Weights & Biases
#     config.wandb = wandb = ConfigDict()
#     wandb.project = "Experiment_NV"
#     wandb.name = "Vorticity"
#     wandb.tag = None

#     # General settings
#     config.nn_model = "MDNN"  # Options: "NN", "WRF", "MDNN"
#     config.lambdas = {"nvs":1, "cond":1, "w0":1, "phi":1}

#     # Model-specific settings
#     config.model = ConfigDict()
#     config.model.input_dim = 3 + 2
#     config.model.hidden_dim = 200
#     config.model.num_layers = 4
#     config.model.out_dim = 2
#     config.model.activation = "tanh"

#     # Weight-Random-Factorization
#     #config.reparam = ConfigDict({"type":"weight_fact","mean":1.0,"stddev":0.1})

#      # Periodic embeddings
#     config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

#     # Fourier embeddings
#     config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":200,"exclude_last_n":2})

#     # Navier Stokes Config
#     config.nu = 1e-2
#     config.time_domain = 2

#     # Training settings
#     config.seed = 108
#     config.learning_rate = 0.001
#     config.decay_rate = 0.9
#     config.alpha = 0.9  # For updating loss weights
#     config.iterations = 5000
#     config.start_scheduler = 0.1
#     config.weights_update = 250
#     config.scheduler_step = 1000 #2000

#     config.chunks = 16
#     config.points_per_chunk = 50
#     #config.batch_ic = 16*

#     # For deep Galerkin- initial conditions
#     config.d = 5
#     config.tau = np.sqrt(2)
#     config.NKL =  1
#     config.dim_initial_condition = 128
#     config.samples_size_initial = 1000
    
#     return config

# Helper function to set up MCMC chain
def run_mcmc_chain(surrogate_model, obs_points, sol_test, config_experiment,device):
    mcmc = NVMCMC(
        surrogate=surrogate_model,
        observation_locations=obs_points,
        observations_values=sol_test,
        observation_noise=np.sqrt(config_experiment.noise_level),
        nparameters=2*config_experiment.KL_expansion,
        nsamples=config_experiment.samples,
        proposal_type=config_experiment.proposal,
        step_size=config_experiment.proposal_variance,
        uniform_limit = config_experiment.uniform_limit,
        device=device
    )
    return mcmc.run_chain(verbose=config_experiment.verbose)

def run_da_mcmc_chain(nn_surrogate_model, obs_points, sol_test,obs_indices, config_experiment,device):
    da_mcmc = NVMCMCDA(nn_surrogate_model,observation_locations= obs_points, observations_values = sol_test, 
                        nparameters=2*config_experiment.KL_expansion,observation_noise=np.sqrt(config_experiment.noise_level),
                        fs_n = config_experiment.fs_n, fs_T=config_experiment.fs_T,fs_steps =config_experiment.fs_steps,
                        fs_indices_sol=obs_indices,iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        proposal_type=config_experiment.proposal,step_size=config_experiment.proposal_variance,
                        uniform_limit = config_experiment.uniform_limit, device=device )

    return da_mcmc.run_chain(verbose=config_experiment.verbose)

# Main experiment runner
def run_experiment(config_experiment,device):

    # Model Paths
    model_specific = f"_hl{config_experiment.num_layers}_n{config_experiment.hidden_dim}_s{config_experiment.points_per_chunk}_kl{config_experiment.KL_expansion}"
    nn_path_model = "./Navier-Stokes/models/vorticity" + model_specific + ".pth"
    dgala_path_model = "./Navier-Stokes/models/nv_dgala" + model_specific + ".pth"

    # Step 1: Training Phase (if needed)
    if config_experiment.train:
        print(f"Running training with the specifics" + model_specific)
        config_experiment.wandb.name = "vorticity" + model_specific
        config_experiment.batch_ic = 16*config_experiment.points_per_chunk
        
        config_experiment.model.input_dim = 3 + 2*config_experiment.KL_expansion
        config_experiment.model.fourier_emb["exclude_last_n"] = 2*config_experiment.KL_expansion
    
        pinn_nvs = train_vorticity_dg(config_experiment, device=device)
        print(f"Completed training with {config_experiment.points_per_chunk} samples.")

    # Step 2: Fit deepGALA
    if config_experiment.deepgala:
        print("Starting DeepGaLA fitting for NN" + model_specific)
        config_experiment.batch_ic = 16*config_experiment.points_per_chunk

        nn_surrogate_model = torch.load(nn_path_model)
        nn_surrogate_model.eval()
        nn_surrogate_model.to(device)
        nn_surrogate_model.M = nn_surrogate_model.M.to(device)

        data_fit = deepgala_data_fit(config_experiment, device)

        llp = dgala(nn_surrogate_model)
        llp.fit(data_fit)
        llp.optimize_marginal_likelihoodb(error_tolerance=1e-4, max_iter=5000)
        clear_hooks(llp)
        torch.save(llp, dgala_path_model)

    # Step 3: Generate noisy observations for Inverse Problem
    obs_points, sol_test, obs_indices,_ = generate_noisy_obs(obs=config_experiment.num_observations,
                                              noise_level=config_experiment.noise_level,
                                              NKL = config_experiment.KL_expansion)
    print(config_experiment.noise_level)

    # Step 4: Neural Network Surrogate for MCMC
    if config_experiment.nn_mcmc:
        print(f"Starting {config_experiment.proposal}-MCMC with NN" + model_specific)
        nn_surrogate_model = torch.load(nn_path_model)
        nn_surrogate_model.eval()
        nn_surrogate_model.to(device)
        nn_samples = run_mcmc_chain(nn_surrogate_model, obs_points, sol_test, config_experiment,device)
        np.save("./Navier-Stokes/results/nn" + model_specific + f"_var{config_experiment.noise_level}.npy", nn_samples[0])
    
    # Step 5: DeepGaLA Surrogate for MCMC
    if config_experiment.dgala_mcmc:
        print(f"Starting {config_experiment.proposal}-MCMC with DeepGaLA" + model_specific)
        llp = torch.load(dgala_path_model)
        llp.model.set_last_layer("output_layer")  # Re-register hooks
        nn_samples = run_mcmc_chain(llp, obs_points, sol_test, config_experiment, device)
        np.save("./Navier-Stokes/results/dgala" + model_specific + f"_var{config_experiment.noise_level}.npy", nn_samples[0])

    # Step 7: Delayed Acceptance for NN
    if config_experiment.da_mcmc_nn:
        print(f"Starting {config_experiment.proposal}-MCMC-DA with NN" + model_specific + "and PSM")
        nn_surrogate_model = torch.load(nn_path_model)
        nn_surrogate_model.eval()
        nn_surrogate_model.to(device)

        acceptance_res,proposal_thetas,lh_val_nn,lh_val_solver = run_da_mcmc_chain(nn_surrogate_model, obs_points, sol_test,
                                                                                   obs_indices, config_experiment, device)
        
        np.save("./Navier-Stokes/results/mcmc_da_nn" + model_specific + f"_var{config_experiment.noise_level}.npy", acceptance_res)
        np.save("./Navier-Stokes/results/mcmc_da_nn_proposal_thetas" + model_specific + f"_var{config_experiment.noise_level}.npy", proposal_thetas)
        np.save("./Navier-Stokes/results/mcmc_da_nn_lh_nn" + model_specific + f"_var{config_experiment.noise_level}.npy", lh_val_nn)
        np.save("./Navier-Stokes/results/mcmc_da_nn_lh_solver" + model_specific + f"_var{config_experiment.noise_level}.npy", lh_val_solver)

    # Step 8: Delayed Acceptance for Dgala
    if config_experiment.da_mcmc_dgala:
        print(f"Starting {config_experiment.proposal}-MCMC-DA with DGALA" + model_specific + "and PSM")

        llp = torch.load(dgala_path_model)
        llp.model.set_last_layer("output_layer")  # Re-register hooks

        acceptance_res,proposal_thetas,lh_val_nn,lh_val_solver = run_da_mcmc_chain(llp, obs_points, sol_test,
                                                                                   obs_indices, config_experiment, device)

        np.save("./Navier-Stokes/results/mcmc_da_dgala" + model_specific + f"_var{config_experiment.noise_level}.npy", acceptance_res)


# Main loop for different sample sizes
def main(verbose,N,hidden_layers,num_neurons,kl,train,deepgala,noise_level,proposal,nn_mcmc,dgala_mcmc,da_mcmc_nn,da_mcmc_dgala, device):
    config_experiment = nv_experiment()
    config_experiment.verbose = verbose
    config_experiment.points_per_chunk = N
    config_experiment.num_layers = hidden_layers 
    config_experiment.hidden_dim = num_neurons
    config_experiment.KL_expansion = kl
    config_experiment.train = train
    config_experiment.deepgala = deepgala
    config_experiment.noise_level = noise_level
    config_experiment.proposal = proposal
    config_experiment.nn_mcmc = nn_mcmc
    config_experiment.dgala_mcmc = dgala_mcmc
    config_experiment.da_mcmc_nn = da_mcmc_nn
    config_experiment.da_mcmc_dgala = da_mcmc_dgala
    run_experiment(config_experiment,device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Elliptic Experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--N", type=int, required=True, help="Number of training samples per chunk")
    parser.add_argument("--hidden_layers", type=int,default=3, help="Number of hidden layers of the NN")
    parser.add_argument("--num_neurons", type=int,default=300, help="Number of neurons per layer")
    parser.add_argument("--kl", type=int,default=1, help="KL expansion for RF")
    parser.add_argument("--train", action="store_true", help="Train NN")
    parser.add_argument("--deepgala", action="store_true", help="Fit DeepGala")
    parser.add_argument("--noise_level", type=float,default=1e-3,help="Noise level for IP")
    parser.add_argument("--proposal", type=str,default="random_walk",help="MCMC Proposal")
    parser.add_argument("--nn_mcmc", action="store_true", help="Run MCMC for NN")
    parser.add_argument("--dgala_mcmc", action="store_true", help="Run MCMC for dgala")
    parser.add_argument("--da_mcmc_nn", action="store_true", help="Run DA-MCMC for NN")
    parser.add_argument("--da_mcmc_dgala", action="store_true", help="Run DA-MCMC for DeepGala")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Pass all arguments
    main(args.verbose, args.N, args.hidden_layers, args.num_neurons,args.kl,args.train, args.deepgala, args.noise_level, args.proposal,
         args.nn_mcmc, args.dgala_mcmc, args.da_mcmc_nn, args.da_mcmc_dgala, device)