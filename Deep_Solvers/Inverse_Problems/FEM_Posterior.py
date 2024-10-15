import pyro
import torch
import time
import numpy as np
from pyro.infer import MCMC, mcmc
import pyro.distributions as dist

from utilities import *
from FEM_Solver import  RootFinder,FEMSolver
from MetropolisHastings import MetropolisHastingsSampler, MoreauYosidaPrior



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print(device)

def tang_eq(x):
    return np.tan(x) - 8*(x/(x**2 - 16))

intervals = [(1,2),(2,3),(3.8,4.1),(4.1,4.6),(4.7,4.8),(7.2,7.3),(7.3,8),(10,10.2),(10.2,11.5),(13,13.2),(14,14.5)]
n = 2

# Define the model using a coarse FEM solver
def model(surrogate_fem_solver,synthetic_data, points,noise_std, device='cpu'):

    lam = 0.001  # Set your lambda value for the Moreau-Yosida prior
    moreau_prior = MoreauYosidaPrior(lam,device=device)

    # Prior for the parameters theta (assuming 2 parameters)
    theta = pyro.sample("theta", moreau_prior.expand([2]))

    # Use a coarse mesh FEM solver as the surrogate model
    surrogate_fem_solver.theta = theta.detach().cpu().numpy()  # Update the parameter vector

    surrogate_fem_solver.uh = None

    surrogate_fem_solver.solve()

    # Get model predictions at the given points
    model_predictions = torch.Tensor(surrogate_fem_solver.eval_at_points(points)).to(device)

    # Likelihood: Gaussian likelihood with a fixed standard deviation
    y = pyro.sample("obs", dist.Normal(model_predictions, noise_std).to_event(1), obs=torch.Tensor(synthetic_data).to(device))

    return y



root_finder = RootFinder(n, tang_eq, intervals)
roots = root_finder.find_roots()


theta_th=np.array([0.098, 0.430])
var = [1e-5,1e-3,1e-1,1,10,100]
obs = [10,25,50,100]

# Timing results storage
timing_results = {
    'CHR_RWMH': [],
    'Pyro_RWMH': []
}


for ob in obs:
    for vr in var:
        st = np.sqrt(vr)
        obs_points, obs_sol = generate_noisy_obs(ob, theta_t=theta_th, mean=0, std=st,vert=200)
        
        sampler = MetropolisHastingsSampler(obs_points, obs_sol, sig = st,numerical = True, 
                                            roots=roots, vert=50, device=device)
        
        start_time_mh = time.time()  # Start timing
        samp_num,dt_tracker_num = sampler.run_sampler(n_chains=1005000)
        end_time_mh = time.time()  # End timing

        elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
        timing_results['CHR_RWMH'].append(elapsed_time_mh)  # Store timing

        np.save(f'./models/FEM_h50_var{vr}_obs{ob}.npy', samp_num)


        # Define the MCMC kernel (using Random Walk Metropolis)
        kernel = mcmc.RandomWalkKernel(model, target_accept_prob=0.234)

        # Create an MCMC object with the kernel
        mcmc_sampler = MCMC(kernel, num_samples=1000000, warmup_steps=5000)

        surrogate_fem_solver = FEMSolver(theta=np.zeros(theta_th.shape[0]), k_roots=roots, vert=50)

        # Run the MCMC inference
        start_time_rw = time.time()  # Start timing
        mcmc_sampler.run(surrogate_fem_solver,obs_sol, obs_points,st)
        end_time_rw = time.time()  # End timing

        elapsed_time_rw = end_time_rw - start_time_rw  # Calculate elapsed time
        timing_results['Pyro_RWMH'].append(elapsed_time_rw)  # Store timing

        # Get the results (posterior samples)
        samples = mcmc_sampler.get_samples()
        samples = samples["theta"].numpy()

        np.save(f'./models/FEM_pyro_h50_var{vr}_obs{ob}.npy', samples)

np.save('./models/timing_results.npy', timing_results)
