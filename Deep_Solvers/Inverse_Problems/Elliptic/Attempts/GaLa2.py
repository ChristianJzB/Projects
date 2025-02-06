from math import sqrt, pi

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from typing import Tuple, Callable, Optional
from copy import deepcopy

from NN import Dense

class FeatureExtractor(nn.Module):
    """Feature extractor for a PyTorch neural network.
    A wrapper which can return the output of the penultimate layer in addition to
    the output of the last layer for each forward pass. If the name of the last
    layer is not known, it can determine it automatically. It assumes that the
    last layer is linear and that for every forward pass the last layer is the same.
    If the name of the last layer is known, it can be passed as a parameter at
    initilization; this is the safest way to use this class.
    Based on https://gist.github.com/fkodom/27ed045c9051a39102e8bcf4ce31df76.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    last_layer_name : str, default=None
        if the name of the last layer is already known, otherwise it will
        be determined automatically.
    """
    def __init__(self, model: nn.Module, last_layer_name: Optional[str] = None) -> None:
        super().__init__()
        self.model = model
        self._features = dict()
        if last_layer_name is None:
            self.last_layer = None
        else:
            self.set_last_layer(last_layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. If the last layer is not known yet, it will be
        determined when this function is called for the first time.

        Parameters
        ----------
        x : torch.Tensor
            one batch of data to use as input for the forward pass
        """
        if self.last_layer is None:
            # if this is the first forward pass and last layer is unknown
            out = self.find_last_layer(x)
        else:
            # if last and penultimate layers are already known
            out = self.model(x)
        return out

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass which returns the output of the penultimate layer along
        with the output of the last layer. If the last layer is not known yet,
        it will be determined when this function is called for the first time.

        Parameters
        ----------
        x : torch.Tensor
            one batch of data to use as input for the forward pass
        """
        out = self.forward(x)
        features = self._features[self._last_layer_name]
        return out, features

    def set_last_layer(self, last_layer_name: str) -> None:
        """Set the last layer of the model by its name. This sets the forward
        hook to get the output of the penultimate layer.

        Parameters
        ----------
        last_layer_name : str
            the name of the last layer (fixed in `model.named_modules()`).
        """
        # set last_layer attributes and check if it is linear
        self._last_layer_name = last_layer_name
        self.last_layer = dict(self.model.named_modules())[last_layer_name]
        if not isinstance(self.last_layer, (nn.Linear, Dense)):
            raise ValueError('Use model with a linear last layer.')

        # set forward hook to extract features in future forward passes
        self.last_layer.register_forward_hook(self._get_hook(last_layer_name))

    def _get_hook(self, name: str) -> Callable:
        def hook(_, input, __):
            # only accepts one input (expects linear layer)
            self._features[name] = input[0].detach()
        return hook

    def find_last_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Automatically determines the last layer of the model with one
        forward pass. It assumes that the last layer is the same for every
        forward pass and that it is an instance of `torch.nn.Linear`.
        Might not work with every architecture, but is tested with all PyTorch
        torchvision classification models (besides SqueezeNet, which has no
        linear last layer).

        Parameters
        ----------
        x : torch.Tensor
            one batch of data to use as input for the forward pass
        """
        if self.last_layer is not None:
            raise ValueError('Last layer is already known.')

        act_out = dict()
        def get_act_hook(name):
            def act_hook(_, input, __):
                # only accepts one input (expects linear layer)
                try:
                    act_out[name] = input[0].detach()
                except (IndexError, AttributeError):
                    act_out[name] = None
                # remove hook
                handles[name].remove()
            return act_hook

        # set hooks for all modules
        handles = dict()
        for name, module in self.model.named_modules():
            handles[name] = module.register_forward_hook(get_act_hook(name))

        # check if model has more than one module
        # (there might be pathological exceptions)
        if len(handles) <= 2:
            raise ValueError('The model only has one module.')

        # forward pass to find execution order
        out = self.model(x)

        # find the last layer, store features, return output of forward pass
        keys = list(act_out.keys())
        for key in reversed(keys):
            layer = dict(self.model.named_modules())[key]
            if len(list(layer.children())) == 0:
                self.set_last_layer(key)

                # save features from first forward pass
                self._features[key] = act_out[key]

                return out

        raise ValueError('Something went wrong (all modules have children).')
        
        
        
class llaplace:

    def __init__(self, model, sigma_noise=1., prior_precision=1.,prior_mean=0.,last_layer_name=None):

        self.model = FeatureExtractor(deepcopy(model), last_layer_name = last_layer_name)
        self._device = next(model.parameters()).device
        self.lossfunc = torch.nn.MSELoss(reduction ='sum')
        self.jacobians_gn = dict()
        self.loss_laplacian = dict()
        self.H = None
        self.hessian_structure = None
        self.loss = 0
        self.temperature = 1

        if self.model.last_layer is None:
            self.mean = None
            self.n_params = None
            self.n_layers = None
            # ignore checks of prior mean setter temporarily, check on .fit()
            self._prior_precision = torch.tensor([prior_precision], device=self._device)
            self._prior_mean = torch.tensor([prior_mean], device=self._device)
            self.sigma_noise = torch.tensor(sigma_noise,device=self._device).float().clone().detach()
        else:
            self.n_params = len(parameters_to_vector(self.model.last_layer.parameters()))
            self.prior_precision = torch.tensor([prior_precision], device=self._device)
            self.prior_mean = torch.tensor([prior_mean], device=self._device)
            self.mean = self.prior_mean
            self.sigma_noise = torch.tensor(sigma_noise,device=self._device).float()
            #self._init_H()
    
    @property
    def posterior_precision(self):
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        if self.hessian_structure == "diag":
            return self._H_factor* self.H + self.prior_precision_diag
        elif self.hessian_structure == "full":
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

    

    def fit(self,pde, hessian_structure ='diag', override = True):
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
        
        self.hessian_structure = hessian_structure

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