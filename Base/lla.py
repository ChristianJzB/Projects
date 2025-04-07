import torch
from torch.nn.utils import parameters_to_vector
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from  .utilities import get_decorated_methods
from .utilities import FeatureExtractor

from copy import deepcopy
from math import sqrt, pi

class dgala(torch.nn.Module):
    def __init__(self, dga, sigma_noise=1., prior_precision=1.,prior_mean=0., last_layer_name = "output_layer"):
        super(dgala, self).__init__()

        self.dgala = deepcopy(dga)
        self.model = FeatureExtractor(deepcopy(dga.model), last_layer_name = last_layer_name)
        self._device = next(dga.model.parameters()).device
        self.lossfunc = torch.nn.MSELoss(reduction ='mean')
        
        self.loss = 0
        self.temperature = 1
        self.H = None
        self.mean = None
        self.n_params = None
        self.n_data = {key: None for key in self.dgala.lambdas.keys()}

        self._prior_precision = torch.tensor([prior_precision], device=self._device)
        self._prior_mean = torch.tensor([prior_mean], device=self._device)
        self._sigma_noise = torch.tensor(sigma_noise,device=self._device).float()

        if hasattr(self.dgala, "chunks"):
            self.chunks = self.dgala.chunks
            self.gamma = self.dgala.gamma.clone()
        else:
            self.chunks = None

    @property
    def prior_precision(self):
        return self._prior_precision
    
    @prior_precision.setter
    def prior_precision(self, new_prior_precision):
        if isinstance(new_prior_precision, torch.Tensor):
            self._prior_precision = new_prior_precision.to(self._device)
        else:
            self._prior_precision = torch.tensor([new_prior_precision], device=self._device).float()

    @property
    def sigma_noise(self):
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, new_sigma_noise):
        if isinstance(new_sigma_noise, torch.Tensor):
            self._sigma_noise = new_sigma_noise.to(self._device)
        else:
            self._sigma_noise = torch.tensor(new_sigma_noise, device=self._device,requires_grad=True).float()

    @property
    def posterior_precision(self):
        """Diagonal posterior precision \\(p\\)."""
        return self._H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def posterior_covariance(self):
        """Diagonal posterior variance \\(p^{-1}\\).""" 
        post_scale = _precision_to_scale_tril(self.posterior_precision)
        return post_scale @ post_scale.T

    @property
    def _H_factor(self):
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 / self.temperature
    
    @property
    def prior_precision_diag(self):
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar, layer-wise, or diagonal prior precision."""

        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params, device=self._device)

        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision
        
    @property
    def scatter(self):
        """Computes the _scatter_, a term of the log marginal likelihood that
        corresponds to L-2 regularization:
        `scatter` = \\((\\theta_{MAP} - \\mu_0)^{T} P_0 (\\theta_{MAP} - \\mu_0) \\)."""

        delta = (self.mean - self.prior_mean)
        return (delta * self.prior_precision_diag) @ delta

    @property
    def log_det_prior_precision(self):
        """Compute log determinant of the prior precision
        \\(\\log \\det P_0\\)"""

        return self.prior_precision_diag.log().sum()

    @property
    def log_det_posterior_precision(self):
        """Compute log determinant of the posterior precision
        \\(\\log \\det P\\) which depends on the subclasses structure
        used for the Hessian approximation."""
        return self.posterior_precision.logdet()

    @property
    def log_det_ratio(self):
        """Compute the log determinant ratio, a part of the log marginal likelihood.
        \\[
            \\log \\frac{\\det P}{\\det P_0} = \\log \\det P - \\log \\det P_0
        \\]log"""
        return self.log_det_posterior_precision - self.log_det_prior_precision
    
    @property
    def log_likelihood(self):
        """Compute log likelihood on the training data after `.fit()` has been called.
        The log likelihood is computed on-demand based on the loss and, for example,
        the observation noise, which makes it differentiable in the latter for
        iterative updates."""

        factor = -self._H_factor
        total_log_likelihood = 0.0

        for key, loss_value in self.loss.items():
            # Compute the normalizer term for Gaussian likelihood
            n_data_key = self.n_data[key]  # Number of data points for this key
            normalizer = n_data_key * torch.log(self.sigma_noise*sqrt(2*pi))

            # Compute log likelihood contribution for this key
            log_likelihood_key = factor *n_data_key* loss_value - normalizer

            # Accumulate total log likelihood
            total_log_likelihood += log_likelihood_key

        return total_log_likelihood
    

    def fit(self,fit_data):
        """Fit the local Laplace approximation at the parameters of the model."""
        
        self.class_methods = get_decorated_methods(self.dgala, decorator = "use_laplace")

       # assert set(self.class_methods) == set([element for sublist in fit_data["class_method"].values() for element in sublist])

        self.dgala.model.eval()
        #self.mean = parameters_to_vector(self.dgala.model.output_layer.parameters()).detach()
        self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()
        self.n_params = len(self.mean)
        self.prior_mean = self._prior_mean
        self._init_H()

        # Dynamically pass the `data_fit` contents as *args
        data_fit_args = fit_data.get("data_fit", {})
        unpacked_args = []
        for key, value in data_fit_args.items():
            if isinstance(value, tuple):
                # Unpack the tuple and add its elements individually
                unpacked_args.extend(value)
            else:
                # Add non-tuple values directly
                unpacked_args.append(value)

        loss = self.dgala.losses(*unpacked_args,loss_fn = self.lossfunc)

        self.loss = {key:loss.item() for key,loss in loss.items()}

        self.full_Hessian(fit_data)
        
    def _init_H(self):
        self.H = torch.zeros(self.n_params,self.n_params,device=self._device)

    def gradient_outograd(self, y, x):
        grad_p = torch.autograd.grad(outputs=y, 
                                 inputs=x, create_graph=True, allow_unused=True, materialize_grads=True)
        return [grp.detach() for grp in grad_p]
    
    def full_Hessian(self,fit_data, damping_factor=1e-6):
        parameters_ = list(self.dgala.model.output_layer.parameters())
        damping = torch.eye(self.n_params,device=self._device)*damping_factor

        for key,dt_fit in fit_data["data_fit"].items():
            dt_fit = dt_fit[1] if isinstance(dt_fit, tuple) else dt_fit

            for z,clm in enumerate(fit_data["class_method"][key]):
                self.dgala.model.zero_grad()
                fout = getattr(self.dgala, clm)(dt_fit)

                if isinstance(fout, tuple):  # Check if fout is a tuple
                    for i, f_out_indv in enumerate(fout):  # Iterate over fout if it's a tuple
                        indv_h = self.compute_hessian(f_out_indv,parameters_,key)
                        self. H += (indv_h + damping)*self.dgala.lambdas[fit_data["outputs"][key][i]]
                        self.n_data[fit_data["outputs"][key][i]] = f_out_indv.shape[0]
                else:
                    indv_h = self.compute_hessian(fout,parameters_,key)
                    self. H += (indv_h + damping)*self.dgala.lambdas[fit_data["outputs"][key][z]]
                    self.n_data[fit_data["outputs"][key][z]] = fout.shape[0]
                
    def compute_hessian (self,output,parameters_,key):
        hessian_loss = torch.zeros(self.n_params,self.n_params,device = self._device)

        if self.chunks: 
            nitems_chunk = output.shape[0] // self.chunks
            chunk_counter = 0

        for i,fo in enumerate(output):
            grad_p = self.gradient_outograd(fo,parameters_)
            
            ndim = grad_p[0].shape[0]

            reshaping_grads = [g.reshape(ndim,-1) for g in grad_p]
            # Concatenate along the parameter axis
            jacobian_matrix = torch.cat(reshaping_grads, dim=1).flatten().unsqueeze(0) 

            hessian_loss +=  jacobian_matrix.T @ jacobian_matrix

            if self.chunks and (i + 1) % nitems_chunk == 0 and key == "pde":
                hessian_loss *= self.gamma[chunk_counter]
                chunk_counter += 1
        return hessian_loss
        

    def __call__(self, x):
        """Compute the posterior predictive on input data `X`."""
        f_mu, f_var = self._glm_predictive_distribution(x)
        return f_mu, f_var


    def _glm_predictive_distribution(self, X):
        Js, f_mu = self.last_layer_jacobians(X)
        f_var = self.functional_variance(Js)
        if f_mu.shape[-1] > 1:
            f_var = torch.diagonal(f_var, dim1 = 1, dim2 = 2)
        return f_mu.detach(), f_var.detach()

    def last_layer_jacobians(self, x):
        """
        Compute Jacobians \\(\\nabla_{\\theta_\\textrm{last}} f(x;\\theta_\\textrm{last})\\) 
        only at current last-layer parameter \\(\\theta_{\\textrm{last}}\\).
        """
        f, phi = self.model.forward_with_features(x)
        bsize = phi.shape[0]
        output_size = f.shape[-1]

        if self.model.last_layer.bias is not None:
            phi = torch.cat([phi, torch.ones(f.shape[0],1).to(self._device)], dim=1)
        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=x.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
        return Js, f.detach()
    

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)
    
  
    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        """Compute the Laplace approximation to the log marginal likelihood
        """
        # update prior precision (useful when iterating on marglik)
        if prior_precision is not None:
            self.prior_precision = prior_precision

        # update sigma_noise (useful when iterating on marglik)
        if sigma_noise is not None:
            self.sigma_noise = sigma_noise

        return self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter)
    
    def optimize_marginal_likelihoodb(self, error_tolerance=1e-3, max_iter=300, lr=1e-2):
        """Optimize the log prior and log sigma by maximizing the marginal likelihood."""
        #log_prior_prec = self.prior_precision.log().requires_grad_(True)

        log_sigma_noise = self.sigma_noise.log().requires_grad_(True)
        log_prior_prec = self.prior_precision.log().requires_grad_(True)

        hyper_optimizer = torch.optim.Adam([log_prior_prec, log_sigma_noise], lr=lr)

        error,n_iter = float('inf'),0  # Initialize error

        while error > error_tolerance and n_iter < max_iter:
            #prev_log_prior, prev_log_sigma = log_prior_prec.detach().clone(), log_sigma_noise.detach().clone()
            prev_log_prior, prev_log_sigma = log_prior_prec.detach().clone(), log_sigma_noise.detach().clone()

            hyper_optimizer.zero_grad()

            # Calculate negative marginal likelihood
            neg_marglik = -self.log_marginal_likelihood(log_prior_prec.exp(), log_sigma_noise.exp())
            neg_marglik.backward(retain_graph=True)

            # Perform optimization step
            hyper_optimizer.step()

            # Calculate the error based on the change in hyperparameters
            error = 0.5 * (torch.abs(log_prior_prec - prev_log_prior) + torch.abs(log_sigma_noise - prev_log_sigma)).item()
            n_iter += 1

            # Optional: log progress for monitoring
            if n_iter % 100 == 0:
                print(f"Iteration {n_iter}, Error: {error:.5f}, neg_marglik: {neg_marglik.item():.5f}")

        self.prior_precision = log_prior_prec.detach().exp()
        self.sigma_noise = log_sigma_noise.detach().exp()

        if n_iter == max_iter:
            print(f"Maximum iterations ({max_iter})reached, sigma : {self.sigma_noise.item()}, prior: {self.prior_precision.item()}.")

    def optimize_marginal_likelihood(self, error_tolerance=1e-3, max_iter=300, lr=1e-2):
        """Optimize the log prior and log sigma by maximizing the marginal likelihood."""
        #log_prior_prec = self.prior_precision.log().requires_grad_(True)

        log_sigma_noise = self.sigma_noise.log().requires_grad_(True)

        hyper_optimizer = torch.optim.Adam([ log_sigma_noise], lr=lr)

        error,n_iter = float('inf'),0  # Initialize error

        while error > error_tolerance and n_iter < max_iter:
            #prev_log_prior, prev_log_sigma = log_prior_prec.detach().clone(), log_sigma_noise.detach().clone()
            prev_log_sigma = log_sigma_noise.detach().clone()

            hyper_optimizer.zero_grad()

            # Calculate negative marginal likelihood
            neg_marglik = -self.log_marginal_likelihood(sigma_noise=log_sigma_noise.exp())
            neg_marglik.backward(retain_graph=True)

            # Perform optimization step
            hyper_optimizer.step()

            # Calculate the error based on the change in hyperparameters
            error = torch.abs(log_sigma_noise - prev_log_sigma).item()
            n_iter += 1

            # Optional: log progress for monitoring
            if n_iter % 100 == 0:
                print(f"Iteration {n_iter}, Error: {error:.5f}, neg_marglik: {neg_marglik.item():.5f}")

        #self.prior_precision = log_prior_prec.detach().exp()
        self.sigma_noise = log_sigma_noise.detach().exp()

        if n_iter == max_iter:
            print(f"Maximum iterations ({max_iter})reached, sigma : {self.sigma_noise.item()}, prior: {self.prior_precision.item()}.")
