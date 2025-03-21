
import torch

from Base.mcmc import MetropolisHastings,MCMCDA
from Base.lla import dgala

from elliptic_files.FEM_Solver import FEMSolver
from elliptic_files.elliptic import Elliptic


class EllipticMCMC(MetropolisHastings):
    def __init__(self, surrogate, observation_locations, observations_values, nparameters=2, 
                 observation_noise=0.5, nsamples=1000000, burnin=None, proposal_type="random_walk", 
                 step_size=0.1, device="cpu"):
        
        super(EllipticMCMC, self).__init__(observation_locations, observations_values, nparameters, 
                 observation_noise, nsamples, burnin, proposal_type, step_size, device)
        
        self.surrogate = surrogate

        # Dictionary to map surrogate classes to their likelihood functions
        likelihood_methods = {FEMSolver: self.fem_log_likelihood,
                                   Elliptic: self.nn_log_likelihood,
                                   dgala: self.dgala_log_likelihood}

        # Precompute the likelihood function at initialization
        surrogate_type = type(surrogate)
        if surrogate_type in likelihood_methods:
            self.log_likelihood_func = likelihood_methods[surrogate_type]
        else:
            raise ValueError(f"Surrogate of type {surrogate_type.__name__} is not supported.")

    def log_prior(self, theta):
        if not ((theta >= -1) & (theta <= 1)).all():
            return -torch.inf
        else:
            return 0

    def fem_log_likelihood(self, theta ):
        """
        Evaluates the log-likelihood given a FEM.
        """
        self.surrogate.theta = theta.cpu().numpy()  # Convert to numpy for FEM solver
        self.surrogate.solve()
        surg = self.surrogate.eval_at_points(self.observation_locations.cpu().numpy()).reshape(-1, 1)
        surg = torch.tensor(surg, device=self.device)
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def nn_log_likelihood(self, theta):
        """
        Evaluates the log-likelihood given a NN.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg = self.surrogate.u(data.float()).detach()
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def dgala_log_likelihood(self, theta):
        """
        Evaluates the log-likelihood given a dgala.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg_mu, surg_sigma = self.surrogate(data)

        surg_mu = surg_mu.view(-1, 1)
        surg_sigma = surg_sigma[:, :, 0].view(-1, 1)

        sigma = self.observation_noise ** 2 + surg_sigma
        dy = surg_mu.shape[0]

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(sigma)))

        return -0.5 * torch.sum(((self.observations_values - surg_mu.reshape(-1, 1)) ** 2) / sigma)- cte
    
    def log_likelihood(self, theta):
        """Directly call the precomputed likelihood function."""
        return self.log_likelihood_func(theta)
        

class EllipticMCMCDA(MCMCDA):
    def __init__(self,coarse_surrogate,finer_surrogate, observation_locations, observations_values, nparameters=2, 
                 observation_noise=0.5, iter_mcmc=1000000, iter_da = 20000,                 
                 proposal_type="random_walk", step_size=0.1, device="cpu" ):
        
        super(EllipticMCMCDA, self).__init__(observation_locations, observations_values, nparameters, 
                 observation_noise, iter_mcmc, iter_da,proposal_type, step_size, device)
        
        self.coarse_surrogate = coarse_surrogate
        self.finer_surrogate = finer_surrogate

        # Dictionary to map surrogate classes to likelihood functions
        likelihood_methods = {
            FEMSolver: self.fem_log_likelihood,
            Elliptic: self.nn_log_likelihood,
            dgala: self.dgala_log_likelihood
        }

        # Precompute likelihood function for both surrogates
        self.log_likelihood_outer_func = self.get_likelihood_function(coarse_surrogate, likelihood_methods)
        self.log_likelihood_inner_func = self.get_likelihood_function(finer_surrogate, likelihood_methods)

    def log_prior(self, theta):
        if not ((theta >= -1) & (theta <= 1)).all():
            return -torch.inf
        else:
            return 0

    def fem_log_likelihood(self,surrogate, theta):
        """
        Evaluates the log-likelihood given a FEM.
        """
        surrogate.theta = theta.cpu().numpy()  # Convert to numpy for FEM solver
        surrogate.solve()
        surg = surrogate.eval_at_points(self.observation_locations.cpu().numpy()).reshape(-1, 1)
        surg = torch.tensor(surg, device=self.device)
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def nn_log_likelihood(self,surrogate,theta):
        """
        Evaluates the log-likelihood given a NN.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg = surrogate.u(data.float()).detach()
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def dgala_log_likelihood(self, surrogate,theta):
        """
        Evaluates the log-likelihood given a dgala.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg_mu, surg_sigma = surrogate(data)

        surg_mu = surg_mu.view(-1, 1)
        surg_sigma = surg_sigma[:, :, 0].view(-1, 1)

        sigma = self.observation_noise ** 2 + surg_sigma
        dy = surg_mu.shape[0]

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(sigma)))
        return -0.5 * torch.sum(((self.observations_values - surg_mu.reshape(-1, 1)) ** 2) / sigma)- cte
    
    def get_likelihood_function(self, surrogate, likelihood_methods):
        """Precompute and return the appropriate likelihood function for a given surrogate."""
        for surrogate_type, likelihood_func in likelihood_methods.items():
            if isinstance(surrogate, surrogate_type):
                return lambda theta: likelihood_func(surrogate, theta)
        raise ValueError(f"Surrogate of type {type(surrogate).__name__} is not supported.")

    def log_likelihood_outer(self, theta):
        return self.log_likelihood_outer_func(theta)

    def log_likelihood_inner(self, theta):
        return self.log_likelihood_inner_func(theta)
    