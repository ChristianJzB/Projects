import torch
import numpy as np

from nv_files.Pseudo_Spectral_Solver import VorticitySolver2D
from nv_files.Field_Generator import omega0_samples_torch

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class MetropolisHastingsSampler:
    """
    A class to perform Metropolis-Hastings sampling for parameter inference.
    It can use a neural network surrogate model or a numerical solver for likelihood evaluation.
    """

    def __init__(self, x, y, surrogate=None, nparam=16, sig=1.0, dt_init=0.5, reg=1e-3, device='cpu'):
        """
        Initialize the Metropolis-Hastings sampler.
        """
        self.device = device

        self.surrogate = surrogate
        self.surrogate.model.to(self.device)

        self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.sig = torch.tensor(sig, dtype=torch.float32, device=self.device)
        self.dt = torch.tensor(dt_init, dtype=torch.float32, device=self.device)
        self.reg = torch.tensor(reg,dtype=torch.float32, device=self.device)  # Regularization parameter

        self.nparam = nparam

        self.scaling_term = 1 / (2 + torch.sqrt(2 * torch.pi * self.reg)).to(self.device)  # Calculate once


    def log_prior_alpha(self, theta):
        """
        Log prior with Moreau-Yosida regularization for the uniform distribution between -1 and 1.
        """
        # Regularization term for all theta values
        regularization_term = -(torch.clamp(torch.abs(theta) - 1, min=0) ** 2) / (2 * self.reg)
        return torch.sum(regularization_term + torch.log(self.scaling_term))

    def proposals(self, alpha):
        """
        Generates proposals for the next step in the Metropolis-Hastings algorithm using a normal distribution.
        """
        # Generate normal proposals
        proposal = alpha + torch.normal(mean=torch.zeros_like(alpha), std=self.dt).to(self.device)
        
        return proposal

    def log_likelihood(self,pr):
        """
        Evaluates the log-likelihood given the surrogate model or numerical solver.
        """
        data_test = torch.cat([self.x, pr.repeat(self.x.size(0), 1)], dim=1)
        surg_mu = self.surrogate.w(data_test).detach().reshape(-1, 1)
        return -0.5 * torch.sum(((self.y - surg_mu) ** 2) / (self.sig ** 2))

        
    def log_posterior(self, pr):
        """
        Evaluates the log-posterior using the surrogate model.
        """
        return self.log_likelihood(pr) + self.log_prior_alpha(pr)

    def run_sampler(self, n_chains, verbose=True):
        """
        Run the Metropolis-Hastings sampling process sequentially.
        """
        # Initialize the parameters randomly within the prior range
        alpha = torch.empty(self.nparam, device=self.device).uniform_(-1, 1)
        alpha_samp = torch.zeros((n_chains, self.nparam), device=self.device)
        dt_tracker = torch.zeros(n_chains, device=self.device)
        acceptance_rate = 0

        for i in range(n_chains):
            # Propose new alpha values
            alpha_proposal = self.proposals(alpha)
            
            # Compute the current log-posterior
            log_posterior_current = self.log_posterior(alpha)
            log_posterior_proposal = self.log_posterior(alpha_proposal)

            # Compute the acceptance ratio
            a = torch.clamp(torch.exp(log_posterior_proposal - log_posterior_current), max=1.0)
            
            # Accept or reject the proposal
            if torch.rand(1, device=self.device) < a:
                alpha = alpha_proposal
                acceptance_rate += 1

            # Store the current sample and step size
            alpha_samp[i] = alpha
            dt_tracker[i] = self.dt

            # Adaptive step size adjustment 
            self.dt += self.dt * (a - 0.234) / (i + 1)

            del log_posterior_current, log_posterior_proposal, alpha_proposal
            if self.device != "cpu":
                torch.cuda.empty_cache()

            # Print progress every 10% of the steps
            if verbose and i % (n_chains // 10) == 0:
                print(f"Iteration {i}, Acceptance Rate: {acceptance_rate / (i + 1):.3f}, Step Size: {self.dt:.4f}")

        if verbose:
            print(f"Final Acceptance Rate: {acceptance_rate / n_chains:.3f}")

        return alpha_samp.detach().cpu().numpy(), dt_tracker.detach().cpu().numpy()
    

### Observation 
dim_obs = 128
NKL = 8

X = torch.linspace(0, 1, dim_obs)*2*torch.pi  # Spatial grid in X direction
Y = torch.linspace(0, 1, dim_obs)*2*torch.pi  # Spatial grid in Y direction
X, Y = torch.meshgrid(X, Y, indexing='ij' )  # Create meshgrid for X, Y

torch.manual_seed(0)  # Replace 42 with your desired seed value

# Generate uniformly distributed values for `theta` in the range [-1, 1]
theta = torch.rand(NKL, 2, 1) * 2 - 1  # Uniform(-1, 1)

w0 = omega0_samples_torch(X, Y, theta,d = 5,tau=np.sqrt(2))

def force_function(X, Y):
    return  (np.sin(X + Y) + np.cos(X + Y))

solver = VorticitySolver2D(N=dim_obs, L=1, T=2.0, nu=1e-2, dt=5e-4,num_sol=100, method='CN',force = force_function)

obs_res = solver.run_simulation( np.array(w0[:,:,0]))

rng = np.random.default_rng(0)
noise = rng.normal(0, np.sqrt(1e-3), obs_res[-1].shape)

noisy_obs = obs_res[-1] + noise

st = torch.sqrt(torch.tensor(1e-3,dtype=torch.float32)).to(device)


### Surrogate
nn_surrogate_model = torch.load(f"./models/vorticity_MDNN_dg.pth")
nn_surrogate_model.eval()

obs_input = torch.cat((X.reshape(-1,1), Y.reshape(-1,1), 2*torch.ones_like(X.reshape(-1,1))), dim=1)
noisy_obs = noisy_obs.reshape(-1,1)

sampler = MetropolisHastingsSampler(surrogate=nn_surrogate_model,x = obs_input,y = noisy_obs,sig = st,device=device)
mcmc_samples,_ = sampler.run_sampler(1000000)

np.save('./results/samples.npy', mcmc_samples)
