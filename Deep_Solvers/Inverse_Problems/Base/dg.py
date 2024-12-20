import torch
import torch.nn as nn
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from Base.utilities import FeatureExtractor

from torch.nn.utils import parameters_to_vector

from typing import Tuple, Callable, Optional
from copy import deepcopy
from Base.deep_models import DNN,WRFNN, MDNN
from math import sqrt, pi
import numpy as np


def _uplad_model(config):
    if config.nn_model == "NN":
        model = DNN(**config.model)
    elif config.nn_model == "WRF":
        model = WRFNN(**config.model)
    elif config.nn_model == "MDNN":
        model = MDNN(**config.model)
    else:
        raise NotImplementedError(f"Model {config.nn_model} not supported yet!")
    return model


class deepGalerkin(torch.nn.Module):
    def __init__(self, config, device):
        super(deepGalerkin, self).__init__()
        self.config = config
        self.device = device
        self.lambdas = dict(config.lambdas)
                    
        # Initialize model
        self.model = _uplad_model(config).to(device)

    def laplace_approx():
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Pass through the original function's behavior
                return func(*args, **kwargs)
            wrapper.use_laplace = True  # Add the attribute to the wrapped function
            return wrapper  # Return the wrapped function
        return decorator

    def losses(self,*args):
        raise NotImplementedError("Subclasses should implement this!")
    
    def total_loss(self,*args,update_weights = False, **kwargs):
        losses = self.losses(*args,**kwargs)

        if update_weights:
            self.grad_weights(losses)

        total_loss = 0
        for key, loss in losses.items():
            total_loss += self.lambdas[key] * loss
        return total_loss, losses

    def grad_weights(self,losses):
        alpha = self.config.alpha
        grads = []
        for _,loss in losses.items():
            # Zero out previous gradients (important to ensure we don't accumulate)
            self.model.zero_grad()
            
            # Compute gradients using autograd.grad instead of backward
            grads_wrt_params = torch.autograd.grad(loss, self.model.parameters(), 
                                    create_graph=True, allow_unused = True,materialize_grads=True)
            
            # Flatten all gradients into a single vector
            grads_wrt_params = parameters_to_vector(grads_wrt_params)

            # Compute L2 norm of the gradient vector
            grads.append(grads_wrt_params.norm(p=2).item())  # Store the L2 norm value

            del grads_wrt_params  # free memory

        # Compute global weights (normalize gradients)
        grad_sum = np.mean(grads)

        lambda_hat =  {key : grad_sum / grad for grad,key in zip(grads,losses.keys())}

        # Update global weights using a moving average
        self.lambdas = { key : (alpha * self.lambdas[key] + (1 - alpha) * lambda_hat[key]) for key in losses.keys()}
        
        


class dgala(torch.nn.Module):
    def __init__(self, dga, sigma_noise=1., prior_precision=1.,prior_mean=0.,last_layer_name=None):
        super(dgala, self).__init__()

        self.dgala = deepcopy(dga)
        self.model = FeatureExtractor(deepcopy(dga.model), last_layer_name = last_layer_name)
        self._device = next(dga.model.parameters()).device

        self.lossfunc = torch.nn.MSELoss(reduction ='sum')
        self.jacobians_gn = dict()
        self.loss_laplacian = dict()
        self.H = None
        self.hessian_structure = None
        self.loss = 0
        self.temperature = 1

        self.mean = None
        self.n_params = None
        self.n_layers = None
        # ignore checks of prior mean setter temporarily, check on .fit()
        self._prior_precision = torch.tensor([prior_precision], device=self._device)
        self._prior_mean = torch.tensor([prior_mean], device=self._device)
        self.sigma_noise = torch.tensor(sigma_noise,device=self._device).float().clone().detach()

    
    @property
    def posterior_precision(self):
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return self._H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def posterior_covariance(self):
        """Diagonal posterior variance \\(p^{-1}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        if self.hessian_structure == "diag":
            return 1 / self.posterior_precision

        elif self.hessian_structure == "full":
            
            post_scale = _precision_to_scale_tril(self.posterior_precision)
            return post_scale @ post_scale.T

    @property
    def _H_factor(self):
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 / self.temperature
    
    @property
    def prior_precision_diag(self):
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar, layer-wise, or diagonal prior precision.

        Returns
        -------
        prior_precision_diag : torch.Tensor
        """
        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params, device=self._device)

        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision
        
    @property
    def scatter(self):
        """Computes the _scatter_, a term of the log marginal likelihood that
        corresponds to L-2 regularization:
        `scatter` = \\((\\theta_{MAP} - \\mu_0)^{T} P_0 (\\theta_{MAP} - \\mu_0) \\).

        Returns
        -------
        [type]
            [description]
        """
        delta = (self.mean - self.prior_mean)
        return (delta * self.prior_precision_diag) @ delta

    @property
    def log_det_prior_precision(self):
        """Compute log determinant of the prior precision
        \\(\\log \\det P_0\\)

        Returns
        -------
        log_det : torch.Tensor
        """
        return self.prior_precision_diag.log().sum()

    @property
    def log_det_posterior_precision(self):
        """Compute log determinant of the posterior precision
        \\(\\log \\det P\\) which depends on the subclasses structure
        used for the Hessian approximation.

        Returns
        -------
        log_det : torch.Tensor
        """
        if self.hessian_structure == "diag":
            return self.posterior_precision.log().sum()

        elif self.hessian_structure == "full":
            return self.posterior_precision.logdet()

    @property
    def log_det_ratio(self):
        """Compute the log determinant ratio, a part of the log marginal likelihood.
        \\[
            \\log \\frac{\\det P}{\\det P_0} = \\log \\det P - \\log \\det P_0
        \\]

        Returns
        -------
        log_det_ratio : torch.Tensor
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision
    
    @property
    def log_likelihood(self):
        """Compute log likelihood on the training data after `.fit()` has been called.
        The log likelihood is computed on-demand based on the loss and, for example,
        the observation noise which makes it differentiable in the latter for
        iterative updates.

        Returns
        -------
        log_likelihood : torch.Tensor
        """
        factor = - self._H_factor
        # loss used is just MSE, need to add normalizer for gaussian likelihood
        c = self.n_data  * torch.log(self.sigma_noise * sqrt(2 * pi))
        return factor * self.loss - c

    

    def fit(self,fit_data, override = True):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
            train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
            override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        """
        if not override:
            raise ValueError('Last-layer Laplace approximations do not support `override=False`.')
        
        self.model.eval()

        if self.model.last_layer is None:
            X = pde["data_set"][pde["PDE"][0]]
            self.n_data = pde["data_set"][pde["PDE"][0]].shape[0]
            with torch.no_grad():
                try:
                    self.model.find_last_layer(X[:1].to(self._device))
                except (TypeError, AttributeError):
                    self.model.find_last_layer(X.to(self._device))
            params = parameters_to_vector(self.model.last_layer.parameters()).detach()
            self.n_params = len(params)
            self.prior_precision = self._prior_precision
            self.prior_mean = self._prior_mean
            self._init_H()

        self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()

        if self.hessian_structure == 'diag':
            self.diagonal_Hessian(pde)

        elif self.hessian_structure == "full":
            self.full_Hessian(pde)
        
    def _init_H(self):
        if self.hessian_structure == "diag":
            self.H = torch.zeros(self.n_params,device=self._device)

        elif self.hessian_structure == "full":
            self.H = torch.zeros(self.n_params,self.n_params,device=self._device)
        
    def jacobians_GN (self,pde):
        for cond in pde["PDE"]:
            self.model.zero_grad()

            fout = getattr(self.model.model, cond)(pde["data_set"][cond])
            features = self.model._features[self.model._last_layer_name]

            loss_f = self.lossfunc(fout,torch.zeros_like(fout))
            self.loss += loss_f

            bsize,output_size = fout.shape[0],  fout.shape[-1]

            if isinstance(self.model.last_layer, nn.Linear):

                last_layer_weights = self.model.last_layer._parameters["weight"]
                last_layer_bias = self.model.last_layer._parameters["bias"]
                parameters_list = [last_layer_weights,last_layer_bias]

                jacobian = torch.zeros(bsize,last_layer_weights.numel() + last_layer_bias.numel(), )

                for i,fo in enumerate(fout):
                    grad_p = torch.autograd.grad(outputs=fo, 
                                                inputs=parameters_list, 
                                                create_graph=True, 
                                                allow_unused=True, 
                                                materialize_grads=True)

                    jacobian[i, :last_layer_weights.numel()] = grad_p[0].detach().view( -1)  # Weights
                    jacobian[i, last_layer_weights.numel():] = grad_p[1].detach().view( -1)  # Bias

            if isinstance(self.model.last_layer, Dense):

                last_layer_g = self.model.last_layer._parameters["g"]
                last_layer_v = self.model.last_layer._parameters["v"]
                last_layer_bias = self.model.last_layer._parameters["bias"]

                parameters_list = [last_layer_g,last_layer_v,last_layer_bias]
                total_g, total_v, total_bias= last_layer_g.numel(),last_layer_v.numel(),last_layer_bias.numel()

                jacobian = torch.zeros(bsize,total_g + total_v + total_bias, )

                for i,fo in enumerate(fout):
                    grad_p = torch.autograd.grad(outputs=fo, 
                                                inputs=parameters_list, 
                                                create_graph=True, 
                                                allow_unused=True, 
                                                materialize_grads=True)

                    jacobian[i, :total_g] = grad_p[0].detach().view( -1)  # Weights
                    jacobian[i, total_g:total_g + total_v] = grad_p[1].detach().view( -1)  # Bias
                    jacobian[i, total_g + total_v:] = grad_p[2].detach().view( -1)  # Bias


            identity = torch.eye(output_size).unsqueeze(0).tile(bsize, 1, 1)
            Js = torch.einsum('kp,kij->kijp', jacobian, identity).reshape(bsize, output_size, -1)

            # Store the Jacobians for each PDE condition
            self.jacobians_gn[cond] = Js


    def diagonal_Hessian(self,pde):
        self.jacobians_GN(pde)
        for cond in pde["PDE"]:
            #self.H += torch.sum(self.jacobians_gn[cond]*self.loss_laplacian[cond]*self.jacobians_gn[cond],axis=0)
            self.H += torch.sum(self.jacobians_gn[cond]*self.jacobians_gn[cond],axis=0)

    def full_Hessian(self,pde, damping_factor=1e-6):
        self.jacobians_GN(pde)
        for cond in pde["PDE"]:
            jacobian = self.jacobians_gn[cond]
            
            # Sum up the outer products of the Jacobian with itself
            self.H += torch.sum(torch.einsum('bce,bdf->bef', jacobian, jacobian), axis=0)
        
        # Add some regularization to avoid singularity
        self.H += torch.eye(self.H.shape[0], device=self._device) * damping_factor
        self.jacobians_gn = dict()

    def __call__(self, x):
        """Compute the posterior predictive on input data `X`.

        Parameters
        ----------
        x : torch.Tensor
            `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here

        Returns
        -------
        predictive: torch.Tensor or Tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax).
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
        """
        f_mu, f_var = self._glm_predictive_distribution(x)
        return f_mu, f_var


    def _glm_predictive_distribution(self, X):
        Js, f_mu = self.last_layer_jacobians(X)
        f_var = self.functional_variance(Js)
        if f_mu.shape[-1] > 1:
            f_var = torch.diagonal(f_var, dim1 = 1, dim2 = 2)
        return f_mu.detach(), f_var.detach()

    def last_layer_jacobians(self, x):
        """Compute Jacobians \\(\\nabla_{\\theta_\\textrm{last}} f(x;\\theta_\\textrm{last})\\) 
        only at current last-layer parameter \\(\\theta_{\\textrm{last}}\\).

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, last-layer-parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        f, phi = self.model.forward_with_features(x)
        bsize = phi.shape[0]
        output_size = f.shape[-1]

        if isinstance(self.model.last_layer, Dense):

            G = self.model.last_layer._parameters["g"].detach()
            V = self.model.last_layer._parameters["v"].detach()

            dydG = (phi @ V).view(bsize, 1)  # Shape (batch_size, 1)
            dydV = (G.item() * phi).view(bsize, 20) 

            jacobian = torch.zeros(bsize, 21)  # 1 for G, 20 for V, and 1 for bias

            # Fill in the Jacobian
            jacobian[:, 0] = dydG.view(-1)  # Fill ∂y/∂G
            jacobian[:, 1:] = dydV  # Fill ∂y/∂V

            phi = jacobian.clone()

        if self.model.last_layer.bias is not None:
            phi = torch.cat([phi, torch.ones(f.shape[0],1).to(self._device)], dim=1)
        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=x.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
        # if self.model.last_layer.bias is not None:
        #     Js = torch.cat([Js, identity], dim=2)
        return Js, f.detach()
    

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        if self.hessian_structure == "diag":
            return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_covariance, Js)

        elif self.hessian_structure == "full":
            return torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)
    
  
    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        """Compute the Laplace approximation to the log marginal likelihood subject
        to specific Hessian approximations that subclasses implement.
        Requires that the Laplace approximation has been fit before.
        The resulting torch.Tensor is differentiable in `prior_precision` and
        `sigma_noise` if these have gradients enabled.
        By passing `prior_precision` or `sigma_noise`, the current value is
        overwritten. This is useful for iterating on the log marginal likelihood.

        Parameters
        ----------
        prior_precision : torch.Tensor, optional
            prior precision if should be changed from current `prior_precision` value
        sigma_noise : [type], optional
            observation noise standard deviation if should be changed

        Returns
        -------
        log_marglik : torch.Tensor
        """
        # update prior precision (useful when iterating on marglik)
        if prior_precision is not None:
            self.prior_precision = prior_precision

        # update sigma_noise (useful when iterating on marglik)
        if sigma_noise is not None:
            self.sigma_noise = sigma_noise

        return self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter)
    
    def optimize_marginal_likelihood(self, error_tolerance=1e-3, max_iter=300, lr=1e-2):
        """Optimize the log prior and log sigma by maximizing the marginal likelihood."""

        log_prior_prec = self.prior_precision.log()
        log_prior_prec.requires_grad = True

        log_sigma_noise = self.sigma_noise.log()
        log_sigma_noise.requires_grad = True

        hyper_optimizer = torch.optim.Adam([log_prior_prec, log_sigma_noise], lr=lr)

        error,n_iter = float('inf'),0  # Initialize error

        while error > error_tolerance and n_iter < max_iter:
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
        self.log_sigma_noise = log_sigma_noise.detach().exp()

        if n_iter == max_iter:
            print(f"Maximum iterations ({max_iter})reached, sigma : {self.sigma_noise.item()}, prior: {self.prior_precision.item()}.")
