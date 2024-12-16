import torch
import numpy as np
from scipy.stats import uniform
from Inverse_Problems.Elliptic.elliptic_files.FEM_Solver import FEMSolver

class MetropolisHastingsSampler:
    """
    A class to perform Metropolis-Hastings sampling for parameter inference.
    It can use a neural network surrogate model or a numerical solver for likelihood evaluation.
    """

    def __init__(self, x, y, surrogate = None,nparam = 2,sig=1.0, dt_init=0.5, mean=True, numerical=False, roots=None, vert = 30):
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
        """
        self.surrogate = surrogate
        self.x = x
        self.y = y
        self.sig = sig
        self.dt = dt_init
        self.nparam = nparam
        self.mean = mean
        self.numerical = numerical
        self.roots = roots
        self.vert = vert
        
        # Error handling for missing roots when numerical solver is required
        if numerical and roots is None:
            raise ValueError("The 'roots' argument cannot be None when 'numerical' is set to True. Please provide valid roots.")

        # Initialize the FEMSolver once, if numerical solver is used
        if self.numerical:
            self.solver = FEMSolver(np.zeros(self.nparam), self.roots, vert=self.vert)

    def log_prior_alpha(self,theta):
        if np.all((theta >= -1) & (theta <= 1)):
            return 0  # log(1) = 0 for uniform prior
        else:
            return -np.inf  # Outside the prior bounds


    # def log_prior_alpha(self, pr):
    #     """
    #     Evaluates the log prior probability density for the parameters `pr`.
    #     Assuming a uniform prior distribution over the range [-1, 1].
    #     """
    #     if np.any(pr < -1) or np.any(pr > 1):
    #         return -np.inf  # Immediately reject if parameters are outside bounds
    #     return np.sum(uniform(loc=-1, scale=2).logpdf(pr))

    def proposals(self, alpha):
        """
        Generates proposals for the next step in the Metropolis-Hastings algorithm.
        
        Args:
            alpha: Current parameter values.
        
        Returns:
            New proposed parameter values based on a normal distribution.
        """
        return np.random.normal(alpha, self.dt, size=alpha.shape)

    def log_likelihood(self, pr):
        """
        Evaluates the log-likelihood given the surrogate model or numerical solver.
        
        Args:
            pr: Current parameters.
        
        Returns:
            Log-likelihood value.
        """
        data = torch.tensor(np.hstack((self.x, np.ones((self.x.shape[0],pr.shape[0])) * pr))).float()

        if self.numerical:

            self.solver.theta = pr  # Update the parameter vector
            self.solver.uh = None
            #surg = self.solver.solve().x.array.reshape(-1, 1)
            
            self.solver.solve()

            surg = self.solver.eval_at_points(self.x).reshape(-1, 1)   

            return -0.5*np.sum(((self.y - surg) ** 2) / (self.sig ** 2))

        elif self.mean:
            surg = self.surrogate(data).detach().cpu().numpy().reshape(-1, 1)
            return -0.5*np.sum(((self.y - surg) ** 2) / (self.sig ** 2))

        else:
            surg_mu, surg_sigma = self.surrogate(data)
            surg_mu = surg_mu.cpu().numpy().reshape(-1, 1)
            surg_sigma = surg_sigma[:, :, 0].reshape(-1, 1).cpu().numpy()

            sigma = self.sig ** 2 + surg_sigma   
            dy = surg_mu.shape[0]

            cte = 0.5*(dy*np.log(2*np.pi)+ np.sum(np.log(sigma)))

            return -0.5*np.sum(((self.y - surg_mu) ** 2) / (sigma)) - cte

    def log_posterior(self, pr):
        """
        Evaluates the log-posterior using the neural network surrogate model.
        
        Args:
            pr: Current parameters.
        
        Returns:
            Log-posterior value.
        """
        return self.log_likelihood(pr) + self.log_prior_alpha(pr)

    def run_sampler(self, n_chains, verbose=True):
        """
        Run the Metropolis-Hastings sampling process.
        
        Args:
            n_chains: Number of MCMC chains (iterations) to run.
            verbose: Whether to print progress and acceptance rate during sampling (default True).
        
        Returns:
            alpha_samp: Sampled parameter values.
            dt_tracker: Step size progression over the chains.
        """
        # Initialize parameters
        alpha = np.random.uniform(-1, 1, size=self.nparam)  # Start with random parameters
        alpha_samp = np.zeros((n_chains, alpha.shape[0]))
        dt_tracker = np.zeros((n_chains, 1))
        acceptance_rate = 0

        for i in range(n_chains):
            # Compute the current log-posterior
            log_posterior_current = self.log_posterior(alpha)

            # Propose new alpha values
            alpha_proposal = self.proposals(alpha)
            log_posterior_proposal = self.log_posterior(alpha_proposal)

            # Compute the acceptance ratio
            a = min(1, np.exp(log_posterior_proposal - log_posterior_current))

            # Accept or reject the proposal
            if np.random.rand() < a:
                alpha = alpha_proposal
                acceptance_rate += 1

            # Adaptive step size adjustment
            self.dt = self.dt + self.dt * (a - 0.234) / (i + 1)
            dt_tracker[i] = self.dt

            # Store the accepted sample
            alpha_samp[i, :] = alpha

            # Print progress every 10% of the iterations
            if verbose and i % (n_chains // 10) == 0:
                print(f"Iteration {i}, Acceptance Rate: {acceptance_rate / (i + 1):.3f}, Step Size: {self.dt:.4f}")

        if verbose:
            print(f"Final Acceptance Rate: {acceptance_rate / n_chains:.3f}")

        return alpha_samp, dt_tracker

