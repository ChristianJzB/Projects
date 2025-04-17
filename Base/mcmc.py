import torch
import torch.distributions as dist
import torch.multiprocessing as mp

from pyro.distributions import Distribution

import numpy as np
from tqdm import tqdm  # For a progress bar



class MetropolisHastings(torch.nn.Module):
    """Implements Metropolis Hastings with multiple chains and configurable prior, likelihood, and proposal."""
    
    def __init__(self, observation_locations, observations_values, nparameters=2, 
                 observation_noise=0.5, nsamples=1000000, burnin=None,  
                 proposal_type="random_walk", step_size=0.1, 
                 uniform_limit=1,my_reg = 1e-3,device="cpu"):
        super(MetropolisHastings, self).__init__()

        # Likelihood data
        self.device = device
        self.observation_locations = torch.tensor(observation_locations, dtype=torch.float64, device=self.device)
        self.observations_values = torch.tensor(observations_values, dtype=torch.float64, device=self.device)
        self.observation_noise = observation_noise
        self.nparameters = nparameters
        
        # MCMC settings
        self.nsamples = nsamples
        self.burnin = int(self.nsamples * 0.1) if burnin is None else burnin
        self.proposal_type = proposal_type
        self.uniform_limit = torch.tensor(uniform_limit, dtype=torch.float64, device=self.device)
        #self.ideal_acceptace_rate = 0.234 if self.proposal_type == "random_walk" else 0.576
        self.ideal_acceptace_rate = torch.tensor( 0.234, dtype=torch.float64, device=self.device)
        self.dt = torch.tensor(step_size, dtype=torch.float64, device=self.device)  # Step size for proposals

        if proposal_type == "langevin":
            self.my_reg = torch.tensor(my_reg, device=device)
            self.my_scaling_term = 1 / (2 + torch.sqrt(2 * torch.pi * self.my_reg)).to(self.device)  # Calculate once

        # Dictionary to map proposal types to their prior functions
        prior_methods = {
            "random_walk": self.log_uniform,
            "langevin": self.log_MY_envelope,
            # "pCN" intentionally omitted
        }

        if proposal_type == "pCN":
            self.log_prior_func = None  # Or set to a dummy function that returns 0
        elif proposal_type in prior_methods:
            self.log_prior_func = prior_methods[proposal_type]
        else:
            raise ValueError(f"Prior for proposal type '{proposal_type}' is not supported.")
        self.to(device)

    def log_MY_envelope(self, theta):
        """
        Log prior with Moreau-Yosida regularization for the uniform distribution between -1 and 1.
        """
        # Regularization term for all theta values
        regularization_term = -(torch.clamp(torch.abs(theta) - 1, min=0) ** 2) / (2 * self.my_reg)
        return regularization_term + torch.log(self.my_scaling_term)

    def log_uniform(self, theta):
        if not torch.logical_and(theta >= -self.uniform_limit, theta <= self.uniform_limit).all():
            return torch.tensor(-float("inf"))  # Ensure it remains a tensor
        return torch.tensor(0.0)  # Keep consistency

    def log_prior(self, theta):
        """Directly call the precomputed likelihood function."""
        return self.log_prior_func(theta)

    def log_likelihood(self, theta):
        """Define the likelihood function. Must be overridden."""
        raise NotImplementedError("log_likelihood must be implemented in a subclass.")
    
    def MYULA(self, theta):
        """Define the likelihood function. Must be overridden."""
        raise NotImplementedError("log_likelihood must be implemented in a subclass.")

    def proposal(self, theta, dt):
        """Proposal with independent step sizes for each chain."""
        if self.proposal_type == "random_walk":
            return theta + torch.normal(mean=torch.zeros_like(theta), std=dt)  # Scale noise by dt (per chain)
        
        elif self.proposal_type == "pCN":
            return theta*torch.sqrt(1-dt**2) + torch.normal(mean=torch.zeros_like(theta), std=dt)  
        
        elif self.proposal_type == "langevin":
            return theta + dt * self.MYULA(theta) + torch.sqrt(2*dt) * torch.randn_like(theta)
    
    def posterior_distribution(self,theta):
        if self.proposal_type == "pCN":
            return self.log_likelihood(theta)
        else:
            return self.log_prior(theta) + self.log_likelihood(theta)


    def run_chain(self, verbose=True):
        """Run Metropolis-Hastings """
        theta = torch.empty((self.nparameters), device=self.device).uniform_(-1, 1)
        samples = torch.zeros((self.nsamples + self.burnin, self.nparameters), device=self.device)
        accepted_proposals = 0

        # Initialize separate dt for each chai
        dt =  self.dt
        
        if verbose:
            pbar = tqdm(range(self.nsamples + self.burnin), desc="Running MCMC", unit="step")
        else:
            pbar = range(self.nsamples + self.burnin)

        for i in pbar:
            theta_proposal = self.proposal(theta, dt)  # Pass dt to proposal

            log_posterior = self.posterior_distribution(theta) 
            log_posterior_proposal = self.posterior_distribution(theta_proposal)

            # Compute acceptance probabilities (vectorized)
            a = torch.exp(log_posterior_proposal - log_posterior).clamp(max=1.0)

            if torch.rand(1, device=self.device) < a:
                theta = theta_proposal
                accepted_proposals += 1

            # Only store samples after burn-in
            samples[i, :] = theta

            # Adaptive step size adjustment (each chain updates its own dt)
            dt += dt * (a - self.ideal_acceptace_rate) / (i + 1)

            if self.proposal_type == "pCN":
                dt = dt.clamp(max=1.0)

            if verbose and (i % (self.nsamples // 10) == 0)and (i!=0):
                pbar.set_postfix(acceptance_rate=f"{accepted_proposals / (i+1):.4f}", proposal_variance=f"{dt:.4f}")

        return samples[self.burnin:,:].detach().cpu().numpy(),accepted_proposals/self.nsamples


    def _run_chain(self, seed, result_queue):
        """Runs a single chain with a given random seed (for parallel execution)."""
        torch.manual_seed(seed)
        samples, accepted_proposals = self.run_chain(verbose=False)
        result_queue.put((samples, accepted_proposals))

    def run_chains(self,nchains = 2):
        """Run multiple chains in parallel or sequentially."""
        seeds = [torch.randint(0, 100000, (1,)).item() for _ in range(nchains)]
        processes = []
        result_queue = mp.Queue()

        for i in range(nchains):
            p = mp.Process(target=self._run_chain, args=(seeds[i], result_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        return results



class MCMCDA(torch.nn.Module):
    """
    Delayed Acceptance for 
    """
    def __init__(self, observation_locations, observations_values, nparameters=2, 
                 observation_noise=0.5, iter_mcmc=1000000, iter_da = 20000,                 
                 proposal_type="random_walk",uniform_limit=1,my_reg = 1e-3,
                 step_size=0.1, device="cpu"):
        super(MCMCDA, self).__init__()

        # Likelihood data
        self.device = device
        self.observation_locations = torch.tensor(observation_locations, dtype=torch.float64, device=self.device)
        self.observations_values = torch.tensor(observations_values, dtype=torch.float64, device=self.device)
        self.observation_noise = observation_noise
        self.nparameters = nparameters

        self.outer_likelihood_value = None
        self.inner_likelihood_value = None
        
        # MCMC settings
        self.iter_da = iter_da
        self.iter_mcmc = iter_mcmc
        self.proposal_type = proposal_type
        self.uniform_limit = torch.tensor(uniform_limit, dtype=torch.float64, device=self.device)

        self.ideal_acceptace_rate = torch.tensor( 0.234, dtype=torch.float64, device=self.device)
        self.dt = torch.tensor(step_size, dtype=torch.float64, device=self.device)  # Step size for proposals

        if proposal_type == "langevin":
            self.my_reg = torch.tensor(my_reg, device=device)
            self.my_scaling_term = 1 / (2 + torch.sqrt(2 * torch.pi * self.my_reg)).to(self.device)  # Calculate once
        elif proposal_type == "pCN":
            self.pcn_mu = torch.zeros(self.nparameters,device=self.device)
            self.pcn_cov= torch.eye(self.nparameters,device=self.device)
            self.normal_dist = dist.MultivariateNormal(self.pcn_mu, self.pcn_cov)

        # Dictionary to map surrogate classes to their likelihood functions
        prior_methods = {"random_walk": self.log_uniform,
                        "pCN": self.log_normal,
                        "langevin": self.log_MY_envelope}

        if proposal_type in prior_methods:
            self.log_prior_func = prior_methods[proposal_type]
        else:
            raise ValueError(f"Prior of type {proposal_type.__name__} is not supported.")
        
        self.to(device)


    def log_MY_envelope(self, theta):
        """
        Log prior with Moreau-Yosida regularization for the uniform distribution between -1 and 1.
        """
        # Regularization term for all theta values
        regularization_term = -(torch.clamp(torch.abs(theta) - 1, min=0) ** 2) / (2 * self.my_reg)
        return regularization_term + torch.log(self.my_scaling_term)

    def log_uniform(self, theta):
        if not torch.logical_and(theta >= -self.uniform_limit, theta <= self.uniform_limit).all():
            return torch.tensor(-float("inf"))  # Ensure it remains a tensor
        return torch.tensor(0.0)  # Keep consistency
        
    def log_normal(self, theta):
        return self.normal_dist.log_prob(theta)

    def log_prior(self, theta):
        """Directly call the precomputed likelihood function."""
        return self.log_prior_func(theta)

    def log_likelihood_outer(self, theta):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("log_likelihood must be implemented in a subclass.")
    
    def log_likelihood_inner(self, theta):
        """Define the likelihood function for the finner model. Must be overridden."""
        raise NotImplementedError("log_likelihood must be implemented in a subclass.")

    def proposal(self, theta, dt):
        """Proposal with independent step sizes for each chain."""
        if self.proposal_type == "random_walk":
            return theta + torch.normal(mean=torch.zeros_like(theta), std=dt).to(self.device)  # Scale noise by dt (per chain)
        elif self.proposal_type == "pCN":
            return theta*torch.sqrt(1-dt**2) + torch.normal(mean=torch.zeros_like(theta), std=dt)  
        
        # elif self.proposal_type == "langevin":
        #     if not theta.requires_grad:
        #         theta.requires_grad_()  # Ensure theta is differentiable
        #     gradient = torch.autograd.grad(self.log_likelihood(theta), theta, retain_graph=True)[0]
        #     return theta + 0.5 * dt * gradient + dt * torch.randn_like(theta)

    def posterior_distribution_outer(self,theta):

        if self.proposal_type == "pCN":
            return self.log_likelihood_outer(theta)
        else:
            return self.log_prior(theta) + self.log_likelihood_outer(theta)
        
    def posterior_distribution_inner(self,theta):
        return self.log_prior(theta) + self.log_likelihood_inner(theta)


    def run_chain(self, samples = False, verbose=True):
        """Run Metropolis-Hastings """
        theta = torch.empty((self.nparameters), device=self.device).uniform_(-1, 1)
        acceptance_list = torch.zeros((self.iter_da), device=self.device)
        samples_outer = torch.zeros((self.iter_mcmc, self.nparameters), device=self.device)
        theta_inner =  torch.zeros((self.iter_da,self.nparameters), device=self.device)
        likelihoods_val_nn = torch.zeros((self.iter_da,self.observation_locations.shape[0]), device=self.device)
        likelihoods_val_solver = torch.zeros((self.iter_da,self.observation_locations.shape[0]), device=self.device)

        samples_inner = []
        # Initialize separate dt for each chai
        dt =  self.dt

        outer_mh = 0
        if verbose:
            pbar = tqdm(range(self.iter_mcmc), desc="Running MCMC", unit="step")
        else:
            pbar = range(self.iter_mcmc)

        for i in pbar:
            # Propose new theta values
            theta_proposal = self.proposal(theta,dt)
            
            # Compute the current log-posterior
            log_posterior_outer = self.posterior_distribution_outer(theta)
            log_posterior_proposal_outer = self.posterior_distribution_outer(theta_proposal)

            # Compute the acceptance ratio
            a = torch.exp(log_posterior_proposal_outer - log_posterior_outer).clamp(max=1.0)

            if torch.rand(1, device=self.device) < a:
                theta = theta_proposal.clone()
                outer_mh += 1

            if samples:
                samples_outer[i,:] = theta

            # Adaptive step size adjustment (each chain updates its own dt)
            dt += dt * (a - self.ideal_acceptace_rate) / (i + 1)
            if self.proposal_type == "pCN":
                dt = dt.clamp(max=1.0)

            if verbose and i % (self.iter_mcmc // 10) == 0 and (i!=0):
                pbar.set_postfix(acceptance_rate=f"{outer_mh / (i+1):.4f}", proposal_variance=f"{dt:.4f}")
            

        print("Starting Delayed Acceptance....")

        if verbose:
            pbar = tqdm(range(self.iter_da), desc="Running Delayed Acceptance", unit="step")
    
        inner_accepted,inner_mh = 0,0
        while inner_mh < self.iter_da:
            # Propose new theta values
            theta_proposal = self.proposal(theta,dt)
            
            # Compute the current log-posterior
            log_posterior_outer = self.posterior_distribution_outer(theta) 
            log_posterior_proposal_outer = self.posterior_distribution_outer(theta_proposal)

            # Compute the acceptance ratio
            a = torch.clamp(torch.exp(log_posterior_proposal_outer - log_posterior_outer), max=1.0)
            a_rec = torch.clamp(torch.exp(log_posterior_outer - log_posterior_proposal_outer), max=1.0)
            # Accept or reject the proposal
            if torch.rand(1, device=self.device) < a:
                inner_mh += 1
                log_posterior_inner = self.posterior_distribution_inner(theta) 
                log_posterior_proposal_inner = self.posterior_distribution_inner(theta_proposal)

                theta_inner[inner_mh-1,:] = theta_proposal.clone()
                likelihoods_val_nn[inner_mh-1,:] = self.outer_likelihood_value.T.clone()
                likelihoods_val_solver[inner_mh-1,:]  = self.inner_likelihood_value.T.clone()

                # Compute the acceptance ratio
                a = torch.clamp(torch.exp(log_posterior_proposal_inner - (log_posterior_inner))*(a_rec/a), max=1.0)

                if verbose:
                    pbar.update(1)

                if torch.rand(1, device=self.device) < a:
                    theta = theta_proposal.clone()
                    inner_accepted += 1
                    acceptance_list[inner_mh-1] +=1

            if samples:
                # Store the current sample and step size
                samples_inner.append(theta)

                # Update progress bar and print progress every 10% (or any interval)
            if verbose and (inner_mh % (self.iter_da // 10) == 0) and (inner_mh != 0):
                acceptance_rate = inner_accepted / inner_mh
                pbar.set_postfix(acceptance_rate=f"{acceptance_rate:.4f}")
            
        # End progress bar
        if verbose:
            pbar.close()

        if verbose:
            print(f"Times inner step {inner_mh:.4f}, Acceptance Rate: {inner_accepted / inner_mh:.4f}")
        if samples:
            return torch.tensor(samples_inner),samples_outer
        else:
            return acceptance_list.cpu(),theta_inner.cpu(),likelihoods_val_nn.cpu(),likelihoods_val_solver.cpu()
    

            

class MoreauYosidaPrior(Distribution):
    def __init__(self, lam, batch_shape=torch.Size([]), device='cpu'):
        super().__init__()
        self.device = device
        self.batch_shape = torch.Size(batch_shape)  # Ensure batch_shape is a torch.Size object
        self.lam = torch.tensor(lam, dtype=torch.float32, device=self.device)


    @property
    def event_shape(self):
        """The event shape for this distribution is 1-dimensional."""
        return torch.Size([1])

    @property
    def support(self):
        """The support of the distribution is the whole real line."""
        return dist.constraints.real

    def log_prob(self, x):
        """Calculate the log probability of x under the Moreau-Yosida prior."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Calculate the regularization term
        regularization_term = -(torch.clamp(torch.abs(x) - 1, min=0) ** 2) / (2 * self.lam)
        log_prob = regularization_term - torch.log(2 + torch.sqrt(2 * torch.pi * self.lam))

        return log_prob

    def sample(self, sample_shape=torch.Size()):
        """Sample from the Moreau-Yosida prior using importance sampling."""
        sample_shape = torch.Size(sample_shape)  # Ensure sample_shape is a torch.Size object
        total_shape = self.batch_shape + sample_shape  # Concatenate batch_shape with sample_shape

        # Proposal distribution with heavier tails (e.g., Normal distribution)
        proposal_dist = dist.Normal(loc=0.0, scale=2.0)  # Wider scale for more tail coverage
        proposals = proposal_dist.sample(total_shape).to(self.device)

        # Calculate log probabilities for the proposals under the Moreau-Yosida prior
        log_prob_samples = self.log_prob(proposals)

        # Calculate log probability of the proposal distribution
        log_prob_proposal = proposal_dist.log_prob(proposals)

        # Calculate weights for importance sampling
        weights = torch.exp(log_prob_samples - log_prob_proposal)

        # Clamp weights to avoid negative entries
        weights = torch.clamp(weights, min=0)

        # Normalize weights
        weights_sum = torch.sum(weights)
        if weights_sum > 0:  # Prevent division by zero
            weights /= weights_sum
        else:
            # If all weights are zero, return uniformly sampled values
            return proposals

        # Resample according to weights (multinomial resampling)
        idx = torch.multinomial(weights, total_shape[0], replacement=True)
        resampled_proposals = proposals[idx]

        return resampled_proposals

    def expand(self, batch_shape, _instance=None):
        """Expand the distribution to a new batch shape."""
        return MoreauYosidaPrior(self.lam, batch_shape, self.device)