import torch
import torch.distributions as dist
from pyro.distributions import Distribution

import numpy as np
from elliptic_files.FEM_Solver import FEMSolver


class MetropolisHastingsSampler:
    """
    A class to perform Metropolis-Hastings sampling for parameter inference.
    It can use a neural network surrogate model or a numerical solver for likelihood evaluation.
    """

    def __init__(self, x, y, surrogate=None, nparam=2, sig=1.0, dt_init=0.5, 
                 mean=True, numerical=False,vert=30,lam = 1 /4, M = 2, reg=1e-3, device='cpu'):
        """
        Initialize the Metropolis-Hastings sampler.
        
        Args:
            surrogate: Surrogate neural network model.
            x: Input data (independent variables).
            y: Observed data (dependent variables).
            sig: Standard deviation of the noise in the observations (default 1.0).
            dt_init: Initial step size for the proposal distribution (default 0.5).
            mean: Whether to use mean predictions for the likelihood (default True).
            numerical: Whether to use a numerical solver instead of the surrogate model (default False).
            roots: Roots for FEM solver, used if numerical=True (optional).
            vert: Vertices for FEM solver, used if numerical=True (optional).
            lam: Regularization parameter for the Moreau-Yosida regularization.
            device: Device to run the model (default 'cpu').
        """
        self.device = device

        self.surrogate = surrogate
        self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.sig = torch.tensor(sig, dtype=torch.float32, device=self.device)
        self.dt = torch.tensor(dt_init, dtype=torch.float32, device=self.device)
        self.reg = torch.tensor(reg,dtype=torch.float32, device=self.device)  # Regularization parameter
        self.lam = lam
        self.M = M

        self.nparam = nparam
        self.mean = mean
        self.numerical = numerical
        self.vert = vert

        # Initialize the FEMSolver once, if numerical solver is used
        if self.numerical:
            self.solver = FEMSolver(np.zeros(self.nparam), self.lam,self.M, vert=self.vert)

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
        
        Args:
            alpha: Current parameter values.
        
        Returns:
            New proposed parameter values.
        """
        # Generate normal proposals
        proposal = alpha + torch.normal(mean=torch.zeros_like(alpha), std=self.dt).to(self.device)
        
        return proposal

    def log_likelihood(self, pr):
        """
        Evaluates the log-likelihood given the surrogate model or numerical solver.
        
        Args:
            pr: Current parameters.
        
        Returns:
            Log-likelihood value.
        """
        if self.numerical:
            self.solver.theta = pr.cpu().numpy()  # Convert to numpy for FEM solver
            self.solver.solve()
            surg = self.solver.eval_at_points(self.x.cpu().numpy()).reshape(-1, 1)

            surg = torch.tensor(surg, device=self.device)
            return -0.5 * torch.sum(((self.y - surg) ** 2) / (self.sig ** 2))

        elif self.mean:
            data = torch.cat([self.x, pr.repeat(self.x.size(0), 1)], dim=1)
            surg = self.surrogate(data).detach().reshape(-1, 1)
            return -0.5 * torch.sum(((self.y - surg) ** 2) / (self.sig ** 2))

        else:
            data = torch.cat([self.x, pr.repeat(self.x.size(0), 1)], dim=1)
            surg_mu, surg_sigma = self.surrogate(data)

            surg_mu = surg_mu.view(-1, 1)
            surg_sigma = surg_sigma[:, :, 0].view(-1, 1)

            sigma = self.sig ** 2 + surg_sigma
            dy = surg_mu.shape[0]

            cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(sigma)))

            return -0.5 * torch.sum(((self.y - surg_mu.reshape(-1, 1)) ** 2) / sigma)- cte

        
    def log_posterior(self, pr):
        """
        Evaluates the log-posterior using the surrogate model.
        
        Args:
            pr: Current parameters.
        
        Returns:
            Log-posterior value.
        """
        return self.log_likelihood(pr) + self.log_prior_alpha(pr)

    def run_sampler(self, n_chains, verbose=True):
        """
        Run the Metropolis-Hastings sampling process sequentially.
        
        Args:
            n_chains: Number of steps in the chain.
            verbose: Whether to print progress (default True).
        
        Returns:
            alpha_samp: Sampled parameter values.
            dt_tracker: Step size progression over the chain.
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
