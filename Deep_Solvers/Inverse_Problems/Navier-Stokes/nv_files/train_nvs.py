import time
import wandb

import torch
import numpy as np

from nv_files.NavierStokes import NavierStokes
from nv_files.data_generator import solve_poisson_fft,compute_velocity,UniformSampler

from nv_files.Field_Generator import omega0_samples_torch


def setup_initial_conditions():
    """Compute initial points and conditions."""
    numpy_x = np.load("./data/numpy_x.npy")
    numpy_y = np.load("./data/numpy_y.npy")
    numpy_w = np.load("./data/numpy_w.npy")

    X, Y = torch.meshgrid(torch.tensor(numpy_x), torch.tensor(numpy_y),indexing="ij")
    dx, dy = X[1, 0] - X[0, 0], Y[0, 1] - Y[0, 0]
    w0 = torch.tensor(numpy_w).unsqueeze(-1)

    psi = solve_poisson_fft(w0, dx, dy)
    u0, v0 = compute_velocity(psi, dx, dy)

    initial_points = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros_like(X.reshape(-1, 1))])
    initial_condition = torch.hstack([w0.reshape(-1, 1), u0.reshape(-1, 1), v0.reshape(-1, 1)])

    return initial_points, initial_condition


def train_pinn_nvs(config,device):
    # Weights & Biases
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    initial_points, initial_condition = setup_initial_conditions()

    dg_NVs = NavierStokes(config=config, device=device)

    # Setup optimizer and scheduler
    batch_size_interior = config.chunks*config.points_per_chunk
    start_scheduler = int(config.iterations * config.start_scheduler )

    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_NVs.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)

    # Domain and sampler setup
    dom = torch.tensor([[0, 2 * torch.pi], [0, 2 * torch.pi],[0,config.time_domain]]).to(device)
    samples_interior = iter(UniformSampler(dom, batch_size_interior))

    # Training loop
    for epoch in range(config.iterations):
        epoch_train_loss = 0.0
        update_weights = (epoch % config.weights_update == 0)
        
        batch = next(samples_interior).to(device)
        _, indices = batch[:, -1].sort()  # Sort based on the time column
        sorted_batch = batch[indices]    # Rearrange rows based on sorted indices

        sorted_batch[:,2] /= config.time_domain

        optimizer.zero_grad()

        # Timer for loss computation
        start_time = time.time()
        total_loss,losses = dg_NVs.total_loss(sorted_batch,initial_condition,initial_points,
                                              loss_fn = loss_fn,update_weights=update_weights)
        loss_computation_time = time.time() - start_time  # Time taken for loss computation

        total_loss.backward()
        optimizer.step()

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
            **{f"weight_{key}": value for key, value in dg_NVs.lambdas.items()}
        })
        # Save the model checkpoint
        if (epoch % 1000 == 0) and (epoch != 0):
            torch.save(dg_NVs, f"./models/{wandb_config.name}_epoch{epoch}.pth")

    # Save final model
    torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")
    wandb.save(f"./models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()
    return dg_NVs


def initial_conditions_samples(config):
    X = torch.linspace(0, 1, config.dim_initial_condition)*2*torch.pi  # Spatial grid in X direction
    Y = torch.linspace(0, 1, config.dim_initial_condition)*2*torch.pi  # Spatial grid in Y direction
    X, Y = torch.meshgrid(X, Y, indexing='ij' )  # Create meshgrid for X, Y

    dx, dy = X[1, 0] - X[0, 0], Y[0, 1] - Y[0, 0]

    torch.manual_seed(config.seed)  # Replace 42 with your desired seed value

    # Generate uniformly distributed values for `theta` in the range [-1, 1]
    theta = torch.rand(config.NKL, 2, config.samples_size_initial) * 2 - 1  # Uniform(-1, 1)

    w0 = omega0_samples_torch(X, Y, theta)
        
    """Compute initial points and conditions."""
    psi = solve_poisson_fft(w0, dx, dy)
    u0, v0 = compute_velocity(psi, dx, dy)

    initial_points = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros_like(X.reshape(-1, 1))])
    
    w0,u0,v0 = w0.reshape(-1, config.samples_size_initial),u0.reshape(-1, config.samples_size_initial),v0.reshape(-1, config.samples_size_initial)

    return initial_points,w0,u0,v0,theta

def parameters_data(theta, data):
    # Repeat the flattened `theta` for the number of rows in `data`
    theta_ = theta.view(-1).repeat(data.size(0), 1)  # Flatten and repeat

    # Concatenate `data` and `theta_` along the last dimension
    return torch.cat((data, theta_), dim=1)


from nv_files.train_nvs import initial_conditions_samples,parameters_data

def train_dg(config,device):
    """Train the PINN model with a timer for loss computation."""
    # Initialize W&B and environment
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)


    batch_size_interior = config.chunks*config.points_per_chunk
    batch_size_initial = config.samples_size_initial
    start_scheduler = int(config.iterations * config.start_scheduler )

    # Load data and prepare initial conditions
    initial_points,w0,u0,v0,theta = initial_conditions_samples(config)

    dg_NVs = NavierStokes(config=config, device=device)

    # Setup optimizer and scheduler
    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_NVs.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)

    # Domain and sampler setup
    dom = torch.tensor([[0, 2 * torch.pi], [0, 2 * torch.pi],[0,config.time_domain]]).to(device)
    samples_interior = iter(UniformSampler(dom, batch_size_interior))

    # Training loop
    for epoch in range(config.iterations):
        epoch_train_loss = 0.0
        epoch_train_indvls = {"nvs":0, "cond":0, "u0":0, "v0":0, "w0":0}

        for indx in range(w0.shape[1]):
            update_weights = ((epoch % config.weights_update == 0) and (indx == 0))

            initial_condition = torch.hstack([w0[:,indx].reshape(-1,1),u0[:,indx].reshape(-1,1),v0[:,indx].reshape(-1,1)])
            initial_points_ = parameters_data(theta=theta[:,:,indx], data=initial_points).to(device)

            batch = next(samples_interior).to(device)
            _, indices = batch[:, -1].sort()  # Sort based on the time column
            sorted_batch = batch[indices]    # Rearrange rows based on sorted indices

            sorted_batch = parameters_data(theta=theta[:,:,indx], data=sorted_batch).to(device)

            optimizer.zero_grad()

            # Timer for loss computation
            start_time = time.time()
            total_loss,losses = dg_NVs.total_loss(sorted_batch,initial_condition,initial_points_,
                                                  loss_fn = loss_fn,update_weights=update_weights)
            loss_computation_time = time.time() - start_time  # Time taken for loss computation

            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_indvls = {key: epoch_train_indvls[key] + losses[key].item() for key in epoch_train_indvls.keys()}

        # Scheduler step
        if epoch >= start_scheduler and (epoch - start_scheduler) % config.scheduler_step == 0:
            scheduler.step()

        # Average loss
        epoch_train_loss /= batch_size_initial
        epoch_train_indvls = {key: epoch_train_indvls[key]/batch_size_initial for key in epoch_train_indvls.keys()}

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "loss_computation_time": loss_computation_time,
            "learning_rate": scheduler.get_last_lr()[0],
            **epoch_train_indvls,
            **{f"weight_{key}": value for key, value in dg_NVs.lambdas.items()}
        })

        # Save the model checkpoint
        if (epoch % 100 == 0) and (epoch != 0):
            torch.save(dg_NVs, f"./models/{wandb_config.name}_epoch{epoch}.pth")

    # Save final model
    torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")
    wandb.save(f"./models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()
    return dg_NVs