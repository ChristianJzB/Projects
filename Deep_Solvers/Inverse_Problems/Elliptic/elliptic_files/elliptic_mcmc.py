import torch
from Base.mcmc import MetropolisHastings


class EllpiticMCMC(MetropolisHastings):
    def __init__(self, surrogate,surrogate_type = "fem" ):
        super(EllpiticMCMC, self).__init__()
        self.surrogate_type = surrogate_type
        self.surrogate = surrogate

    def log_prior(self, theta):
        if not ((theta >= -1) & (theta <= 1)).all():
            return -torch.inf

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
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1)
        surg = self.surrogate(data).detach().reshape(-1, 1)
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def dgala_log_likelihood(self, theta):
        """
        Evaluates the log-likelihood given a dgala.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1)
        surg_mu, surg_sigma = self.surrogate(data)

        surg_mu = surg_mu.view(-1, 1)
        surg_sigma = surg_sigma[:, :, 0].view(-1, 1)

        sigma = self.observation_noise ** 2 + surg_sigma
        dy = surg_mu.shape[0]

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(sigma)))

        return -0.5 * torch.sum(((self.observations_values - surg_mu.reshape(-1, 1)) ** 2) / sigma)- cte
    
    def log_likelihood(self, theta):
        if self.surrogate_type == "fem":
            return self.fem_log_likelihood(theta=theta)
        if self.surrogate_type == "nn":
            return self.nn_log_likelihood(theta=theta)
        if self.surrogate_type == "dgala":
            return self.dgala_log_likelihood(theta=theta)