import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from typing import Tuple, Callable, Optional
from copy import deepcopy

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
        if not isinstance(self.last_layer, nn.Linear):
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
        self.loss = torch.nn.MSELoss(reduction ='sum')
        self.jacobians_gn = dict()
        self.loss_laplacian = dict()
        self.H = None
        self.hessian_structure = None

        if self.model.last_layer is None:
            self.mean = None
            self.n_params = None
            self.n_layers = None
            # ignore checks of prior mean setter temporarily, check on .fit()
            self._prior_precision = torch.tensor([prior_precision], device=self._device)
            self._prior_mean = torch.tensor([prior_mean], device=self._device)
            self.sigma_noise = torch.tensor(sigma_noise).float()
        else:
            self.n_params = len(parameters_to_vector(self.model.last_layer.parameters()))
            self.prior_precision = torch.tensor([prior_precision], device=self._device)
            self.prior_mean = torch.tensor([prior_mean], device=self._device)
            self.mean = self.prior_mean
            self.sigma_noise = torch.tensor(sigma_noise).float()
            #self._init_H()

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
            N = pde["data_set"][pde["PDE"][0]].shape[0]
            with torch.no_grad():
                try:
                    self.model.find_last_layer(X[:1].to(self._device))
                except (TypeError, AttributeError):
                    self.model.find_last_layer(X.to(self._device))
            params = parameters_to_vector(self.model.last_layer.parameters()).detach()
            self.n_params = len(params)
            #self.n_layers = len(list(self.model.last_layer.parameters()))
            # here, check the already set prior precision again
            self.prior_precision = self._prior_precision
            self.prior_mean = self._prior_mean
            self._init_H()

        #super().fit(train_loader, override=override)
        self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()

        if self.hessian_structure == 'diag':
            self.diagonal_Hessian(pde)

        elif self.hessian_structure == "full":
            self.full_Hessian(pde)
            
        self.H = (1/2) * self.H
        
    def _init_H(self):
        if self.hessian_structure == "diag":
            self.H = torch.zeros(self.n_params)

        elif self.hessian_structure == "full":
            self.H = torch.zeros(self.n_params,self.n_params)

        
    def jacobians_GN (self,pde):
        for cond in pde["PDE"]:
            self.model.zero_grad()
            # if cond == "de":
            #     fout = getattr(self.model.model, 'de')(pde["data_set"][cond])
            # else:
            #     fout = self.model(pde["data_set"][cond])

            fout = getattr(self.model.model, cond)(pde["data_set"][cond])
            features = self.model._features[self.model._last_layer_name]

            loss_f = self.loss(fout,torch.zeros_like(fout))
            df = grad(loss_f, fout, create_graph=True)[0]
            ddf = grad(df, fout, torch.ones_like(df))[0]

            self.jacobians_gn[cond] = torch.cat((features,torch.ones_like(fout)),1) 
            self.loss_laplacian[cond] = ddf
                    

    def diagonal_Hessian(self,pde):
        self.jacobians_GN(pde)
        for cond in pde["PDE"]:
            self.H += torch.sum(self.jacobians_gn[cond]*self.loss_laplacian[cond]*self.jacobians_gn[cond],axis=0)

    def full_Hessian(self,pde):
        self.jacobians_GN(pde)
        for cond in pde["PDE"]:
            self.H += torch.sum(torch.einsum("bcd,ba->bcd",torch.einsum('bc,bd->bcd', self.jacobians_gn[cond], self.jacobians_gn[cond]), self.loss_laplacian[cond]),axis=0)

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

        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=x.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
        if self.model.last_layer.bias is not None:
            Js = torch.cat([Js, identity], dim=2)
        return Js, f.detach()
    

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        if self.hessian_structure == "diag":
            return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_covariance, Js)

        elif self.hessian_structure == "full":
            return torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)

        
    
    @property
    def posterior_precision(self):
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        if self.hessian_structure == "diag":
            return self._H_factor() * self.H + self.prior_precision_diag()

        elif self.hessian_structure == "full":
            return self._H_factor() * self.H + torch.diag(self.prior_precision_diag())


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


    def _H_factor(self):
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 

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

