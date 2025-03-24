import time
import wandb

import torch
import numpy as np
from scipy.stats import qmc

from files.burguers import Burgers
from files.heat import Heat


def samples_param(size, nparam = 1, min_val=0.0001, max_val=0.05,seed = 0):
    """Sample parameters uniformly from a specified range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(min_val, max_val, size=(size, nparam))

def generate_data(size, nparam = 1, seed = 0, burgers=True):
        #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        xy = sampler.random(n=size)

        if burgers:
            xy[:,0] = xy[:,0]*2-1
            xy[:,1] = xy[:,1]*0.5

        param = samples_param(size=size, nparam= nparam)
        xy_tensor = torch.Tensor(xy)
        param_tensor = torch.Tensor(param)
        x,y  = xy_tensor[:,0].reshape(-1,1),xy_tensor[:,1].reshape(-1,1)

        data_int = torch.cat([xy_tensor, param_tensor], axis=1).float()

        ini_c = torch.cat([x,torch.zeros_like(x).float(),param_tensor],axis = 1).float()

        if burgers:
            left_bc = torch.cat([torch.ones_like(x).float()*(-1),y, param_tensor],axis = 1).float()
        else: 
            left_bc = torch.cat([torch.zeros_like(x).float(),y, param_tensor],axis = 1).float()


        right_bc = torch.cat([torch.ones_like(x).float(),y, param_tensor],axis = 1).float()

        return data_int,ini_c, left_bc, right_bc 

# Helper function for activation functions
def _get_pde(problem):
    if problem == "burgers":
        return Burgers
    elif problem == "heat":
        return Heat
    else:
        raise ValueError(f"Unknown problem: {problem}")

def train_dga(config, device):
    """Train the PINN using the Adam optimizer with early stopping, computing test loss after each epoch."""
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)
    seed = config.seed
    #torch.manual_seed(config.seed)

    start_scheduler = int(config.epochs * config.start_scheduler )

    problem = _get_pde(config.dga)
    type_problem = True if config.dga=="burgers" else False

    dg_model = problem(config=config, device=device)

    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_model.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)  # Exponential decay scheduler

    for epoch in range(config.epochs):
        epoch_train_loss = 0.0  # Accumulate loss over all batches for this epoch

        data_int,ini_c, left_bc, right_bc = generate_data(size= config.batch_size,seed = seed + epoch, burgers= type_problem)
        data_int,ini_c, left_bc, right_bc = data_int.to(device),ini_c.to(device),left_bc.to(device), right_bc.to(device)

        optimizer.zero_grad()

        start_time = time.time()
        total_loss,losses = dg_model.total_loss(data_int, ini_c,left_bc, right_bc, loss_fn)
        loss_computation_time = time.time() - start_time  # Time taken for loss computation

        total_loss.backward()
        optimizer.step()
        
        # Accumulate the batch loss into the epoch loss
        epoch_train_loss += total_loss.item()

        # Scheduler step
        if epoch >= start_scheduler and (epoch - start_scheduler) % config.scheduler_step == 0:
            scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "loss_computation_time": loss_computation_time,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"loss_{key}": value.item() for key, value in losses.items()},
        })

    optimizer = torch.optim.LBFGS(
        dg_model.model.parameters(), lr=config.learning_rate, max_iter=50000, max_eval=None, tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps,line_search_fn="strong_wolfe" 
        )
    
    print("Starting Training: LBFGS optimizer")

    data_int,ini_c, left_bc, right_bc = generate_data(size= config.batch_size,seed = seed + epoch, burgers= type_problem)
    data_int,ini_c, left_bc, right_bc = data_int.to(device),ini_c.to(device),left_bc.to(device), right_bc.to(device)    

    def loss_func_train():
        optimizer.zero_grad()
        total_loss,losses = dg_model.total_loss(data_int, ini_c,left_bc, right_bc, loss_fn)

        total_loss.backward() 

        return total_loss

    optimizer.step(loss_func_train) 


    # Save final model
    torch.save(dg_model, f"./models/dnn_models/{wandb_config.name}.pth")
    wandb.save(f"./models/dnn_models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()

    return dg_model

######################### Data Training Elliptic

def samples_param_elliptic(size, nparam, min_val=-1, max_val=1,seed = 65647437836358831880808032086803839626):
    """Sample parameters uniformly from a specified range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(min_val, max_val, size=(size, nparam))


def generate_data_elliptic(size, param = None, nparam = 2, seed = 65647437836358831880808032086803839626):
        #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        x = sampler.random(n=size)

        if param is None:
            param = samples_param_elliptic(size=size, nparam= nparam)
        else:
            param = param[:size,:]

        x_tensor = torch.Tensor(x)
        param_tensor = torch.Tensor(param)

        data_int = torch.cat([x_tensor, param_tensor], axis=1).float()
        left_bc = torch.cat([torch.zeros_like(x_tensor).float(), param_tensor], axis=1).float()
        right_bc = torch.cat([torch.ones_like(x_tensor).float(), param_tensor], axis=1).float()

        return data_int, left_bc, right_bc  