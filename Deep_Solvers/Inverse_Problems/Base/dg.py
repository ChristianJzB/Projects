import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from Base.deep_models import DNN,WRFNN, MDNN

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