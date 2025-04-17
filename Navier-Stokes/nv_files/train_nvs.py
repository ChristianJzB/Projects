import time
import wandb

import torch
import numpy as np

from nv_files.NavierStokes import NavierStokes,Vorticity
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

# Function to select indices
def sample_indices_ic(size, N, seed):
    with torch.random.fork_rng():  # Isolate RNG
        torch.manual_seed(seed)  # Set the seed
        indices = torch.randperm(size)[:N]  # Generate N random indices
    return indices

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
    samples_interior = iter(UniformSampler(dom, batch_size_interior,rng_seed= config.seed))

    # Training loop
    for epoch in range(config.iterations):
        epoch_train_loss = 0.0
        update_weights = (epoch % config.weights_update == 0)
        
        batch = next(samples_interior).to(device)
        _, indices = batch[:, -1].sort()  # Sort based on the time column
        sorted_batch = batch[indices]    # Rearrange rows based on sorted indices

        sorted_batch[:,2] /= config.time_domain

        indices = sample_indices_ic(initial_points.shape[0],batch_size_interior, config.seed + epoch)
        sampled_points = initial_points[indices]
        sampled_condition = initial_condition[indices]

        optimizer.zero_grad()

        # Timer for loss computation
        start_time = time.time()
        total_loss,losses = dg_NVs.total_loss(sorted_batch,sampled_condition,sampled_points,
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
            torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")

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

def data_set_cat(data_set, theta):
    theta = theta.view(-1,theta.shape[-1])  # Flatten and repeat

    # Repeat `ini` for the number of columns in `theta`
    num_cols = theta.shape[1]
    data_set_r = data_set.repeat(num_cols, 1)  # Shape (16000, 3)

    # Repeat each column of `theta` to match rows in `ini`
    theta_ = theta.T.repeat_interleave(data_set.shape[0], dim=0)  # Shape (16000, 100)

    # Concatenate along the last dimension
    result = torch.cat((data_set_r, theta_), dim=1)  # Shape (16000, 103)
    return result

def data_set_preparing(config,batch, initial_points,w0,u0,v0,theta,batch_size_interior,epoch):
    initial_points_ = data_set_cat(initial_points,theta)

    indices = sample_indices_ic(initial_points_.shape[0],config.batch_ic, config.seed + epoch)

    initial_points_ = initial_points_[indices]

    w0_,u0_,v0_ = w0.T.reshape(-1,1)[indices],u0.T.reshape(-1,1)[indices],v0.T.reshape(-1,1)[indices]

    initial_condition = torch.hstack([w0_,u0_,v0_])

    batch = data_set_cat(batch,theta)

    _, indices = batch[:, 2].sort()  # Sort based on the time column

    sorted_batch = batch[indices]    # Rearrange rows based on sorted indices

    indices = sample_indices_ic(sorted_batch.shape[0],batch_size_interior, config.seed + epoch)
    sorted_batch = sorted_batch[indices]

    return sorted_batch,initial_points_,initial_condition



def train_dg(config,device):
    """Train the PINN model with a timer for loss computation."""
    # Initialize W&B and environment
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)


    batch_size_interior = config.chunks*config.points_per_chunk
    start_scheduler = int(config.iterations * config.start_scheduler )

    # Load data and prepare initial conditions
    initial_points,w0,u0,v0,theta = initial_conditions_samples(config)
    initial_points,w0,u0,v0,theta = initial_points.to(device),w0.to(device),u0.to(device),v0.to(device),theta.to(device)
    
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
        update_weights = (epoch % config.weights_update == 0)

        batch = next(samples_interior).to(device)

        sorted_batch,initial_points_,initial_condition = data_set_preparing(config,batch, 
                                                    initial_points,w0,u0,v0,theta,batch_size_interior,epoch)
        
        sorted_batch,initial_points_,initial_condition  = sorted_batch.to(device),initial_points_.to(device),initial_condition.to(device) 

        optimizer.zero_grad()

        # Timer for loss computation
        start_time = time.time()
        total_loss,losses = dg_NVs.total_loss(sorted_batch,initial_condition,initial_points_,
                                                loss_fn = loss_fn,update_weights=update_weights)
        loss_computation_time = time.time() - start_time  # Time taken for loss computation

        total_loss.backward()
        optimizer.step()

        # Scheduler step
        if epoch >= start_scheduler and (epoch - start_scheduler) % config.scheduler_step == 0:
            scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss.item(),
            "loss_computation_time": loss_computation_time,
            "learning_rate": scheduler.get_last_lr()[0],
            **{key: loss.item() for key,loss in losses.items()},
            **{f"weight_{key}": value for key, value in dg_NVs.lambdas.items()}
        })

        # Save the model checkpoint
        if (epoch % 100 == 0) and (epoch != 0):
            torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")

    # Save final model
    torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")
    wandb.save(f"./models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()
    return dg_NVs


########### Vorticity ########################
##############################################
def setup_ic_vortice():
    """Compute initial points and conditions."""
    numpy_x = np.load("./data/numpy_x.npy")
    numpy_y = np.load("./data/numpy_y.npy")
    numpy_w = np.load("./data/numpy_w.npy")

    X, Y = torch.meshgrid(torch.tensor(numpy_x), torch.tensor(numpy_y),indexing="ij")
    w0 = torch.tensor(numpy_w).unsqueeze(-1)

    initial_points = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros_like(X.reshape(-1, 1))])

    return initial_points, w0.reshape(-1, 1)

# Function to select indices
def sample_indices_ic(size, N, seed):
    with torch.random.fork_rng():  # Isolate RNG
        torch.manual_seed(seed)  # Set the seed
        indices = torch.randperm(size)[:N]  # Generate N random indices
    return indices

def train_pinn_vortice(config,device):
    # Weights & Biases
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    initial_points, initial_condition = setup_ic_vortice()

    dg_NVs = Vorticity(config=config, device=device)

    # Setup optimizer and scheduler
    batch_size_interior = config.chunks*config.points_per_chunk
    start_scheduler = int(config.iterations * config.start_scheduler )

    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_NVs.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)

    # Domain and sampler setup
    dom = torch.tensor([[0, 2 * torch.pi], [0, 2 * torch.pi],[0,config.time_domain]]).to(device)
    samples_interior = iter(UniformSampler(dom, batch_size_interior,rng_seed= config.seed))

    # Training loop
    for epoch in range(config.iterations):
        epoch_train_loss = 0.0
        update_weights = (epoch % config.weights_update == 0)
        
        batch = next(samples_interior).to(device)
        _, indices = batch[:, -1].sort()  # Sort based on the time column
        sorted_batch = batch[indices]    # Rearrange rows based on sorted indices

        sorted_batch[:,2] /= config.time_domain

        indices = sample_indices_ic(initial_points.shape[0],batch_size_interior, config.seed + epoch)
        sampled_points = initial_points[indices]
        sampled_condition = initial_condition[indices]

        optimizer.zero_grad()

        # Timer for loss computation
        start_time = time.time()
        total_loss,losses = dg_NVs.total_loss(sorted_batch,sampled_condition,sampled_points,
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
            torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")

    # Save final model
    torch.save(dg_NVs, f"./models/{wandb_config.name}.pth")
    wandb.save(f"./models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()
    return dg_NVs




def ic_vort_samples(config):
    X = torch.linspace(0, 1, config.dim_initial_condition)*2*torch.pi  # Spatial grid in X direction
    Y = torch.linspace(0, 1, config.dim_initial_condition)*2*torch.pi  # Spatial grid in Y direction
    X, Y = torch.meshgrid(X, Y, indexing='ij' )  # Create meshgrid for X, Y

    dx, dy = X[1, 0] - X[0, 0], Y[0, 1] - Y[0, 0]

    torch.manual_seed(config.seed)  # Replace 42 with your desired seed value

    # Generate uniformly distributed values for `theta` in the range [-1, 1]
    theta = torch.rand(config.KL_expansion, 2, config.samples_size_initial) * 2 - 1  # Uniform(-1, 1)

    w0 = omega0_samples_torch(X, Y, theta, d=config.d, tau=config.tau)
        
    """Compute initial points and conditions."""
    #psi = solve_poisson_fft(w0, dx, dy)

    initial_points = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros_like(X.reshape(-1, 1))])
    w0 = w0.reshape(-1, config.samples_size_initial)
    #,psi.reshape(-1, config.samples_size_initial)
    #return initial_points,w0,psi,theta
    return initial_points,w0,theta

def data_set_cat(data_set, theta):
    theta = theta.view(-1,theta.shape[-1])  # Flatten and repeat

    # Repeat `ini` for the number of columns in `theta`
    num_cols = theta.shape[1]
    data_set_r = data_set.repeat(num_cols, 1)  # Shape (16000, 3)

    # Repeat each column of `theta` to match rows in `ini`
    theta_ = theta.T.repeat_interleave(data_set.shape[0], dim=0)  # Shape (16000, 100)

    # Concatenate along the last dimension
    result = torch.cat((data_set_r, theta_), dim=1)  # Shape (16000, 103)
    return result

def data_vor_set_preparing(config,batch, initial_points,w0,theta,batch_size_interior,epoch):
    initial_points_ = data_set_cat(initial_points,theta)
    batch_ic = config.chunks*config.points_per_chunk 

    indices = sample_indices_ic(initial_points_.shape[0],batch_ic, config.seed + epoch)

    initial_points_ = initial_points_[indices]

    #w0_,psi_ = w0.T.reshape(-1,1)[indices],psi.T.reshape(-1,1)[indices]

    #initial_condition = torch.hstack([w0_,psi_])
    initial_condition = w0.T.reshape(-1,1)[indices]

    batch = data_set_cat(batch,theta)

    _, indices = batch[:, 2].sort()  # Sort based on the time column

    sorted_batch = batch[indices]    # Rearrange rows based on sorted indices

    indices = sample_indices_ic(sorted_batch.shape[0],batch_size_interior, config.seed + epoch)
    sorted_batch = sorted_batch[indices]

    return sorted_batch,initial_points_,initial_condition

def ic_vort_test_set(config):
    X = torch.linspace(0, 1, config.dim_initial_condition)*2*torch.pi  # Spatial grid in X direction
    Y = torch.linspace(0, 1, config.dim_initial_condition)*2*torch.pi  # Spatial grid in Y direction
    X, Y = torch.meshgrid(X, Y, indexing='ij' )  # Create meshgrid for X, Y

    dx, dy = X[1, 0] - X[0, 0], Y[0, 1] - Y[0, 0]

    torch.manual_seed(config.seed +1)  # Replace 42 with your desired seed value

    # Generate uniformly distributed values for `theta` in the range [-1, 1]
    theta = torch.rand(config.KL_expansion, 2, config.samples_size_initial) * 2 - 1  # Uniform(-1, 1)

    w0 = omega0_samples_torch(X, Y, theta, d=config.d, tau=config.tau)
        
    """Compute initial points and conditions."""
    #psi = solve_poisson_fft(w0, dx, dy)

    initial_points = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.zeros_like(X.reshape(-1, 1))])
    w0 = w0.reshape(-1, config.samples_size_initial)
    #w0,psi = w0.reshape(-1, config.samples_size_initial),psi.reshape(-1, config.samples_size_initial)
    #return initial_points,w0,psi,theta
    return initial_points,w0,theta

def test_valuation(config,model,initial_points,w0,theta):
    test_w, test_phi = [], []
    for i in range(config.samples_size_initial):
        wo_ = w0[:,i].reshape(-1,1)
        #phi_ = phi[:,i].reshape(-1,1)
        data_test = theta[:,:,i].reshape(-1,1).T.repeat_interleave(initial_points.shape[0], dim=0)
        data_test = torch.cat((initial_points, data_test), dim=1)  # Shape (16000, 103)

        wpred = model.w(data_test)
        #phipred = model.phi(data_test)
        rel_error_w = torch.linalg.norm(wpred - wo_,ord = 2)/ torch.linalg.norm(wo_,ord = 2)
        #rel_error_phi = torch.linalg.norm(phipred - phi_,ord = 2)/ torch.linalg.norm(phi_,ord = 2)
        test_w.append(rel_error_w.item())
        #test_phi.append(rel_error_phi.item())
    return np.mean(test_w)
    #return np.mean(test_w),np.mean(test_phi)



def train_vorticity_dg(config,device):
    """Train the PINN model with a timer for loss computation."""
    # Initialize W&B and environment
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)


    batch_size_interior = config.chunks*config.points_per_chunk
    start_scheduler = int(config.iterations * config.start_scheduler )

    # Load data and prepare initial conditions
    initial_points,w0,theta = ic_vort_samples(config)
    initial_points,w0,theta = initial_points.to(device),w0.to(device),theta.to(device)

    ip_test,w0_test,theta_test = ic_vort_test_set(config)
    ip_test,w0_test,theta_test = ip_test.to(device),w0_test.to(device),theta_test.to(device)

    dg_NVs = Vorticity(config=config, device=device)

    # Setup optimizer and scheduler
    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_NVs.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)

    # Domain and sampler setup
    dom = torch.tensor([[0, 2 * torch.pi], [0, 2 * torch.pi],[0,config.time_domain]]).to(device)
    samples_interior = iter(UniformSampler(dom, batch_size_interior,device = device))

    # Training loop
    for epoch in range(config.iterations):
        update_weights = (epoch % config.weights_update == 0)

        batch = next(samples_interior)

        sorted_batch,initial_points_,initial_condition = data_vor_set_preparing(config,batch, 
                                                    initial_points,w0,theta,batch_size_interior,epoch)
        
        sorted_batch,initial_points_,initial_condition  = sorted_batch.to(device),initial_points_.to(device),initial_condition.to(device) 

        optimizer.zero_grad()

        # Timer for loss computation
        #start_time = time.time()
        total_loss,losses = dg_NVs.total_loss(sorted_batch,initial_condition,initial_points_,
                                                loss_fn = loss_fn,update_weights=update_weights)

        #loss_computation_time = time.time() - start_time  # Time taken for loss computation

        total_loss.backward()
        optimizer.step()

        # if (epoch % 1000 == 0) and (epoch != 0):
        #     test_w = test_valuation(config, dg_NVs,ip_test,w0_test,theta_test)
        #     wandb.log({"test_w":test_w})

        # Scheduler step
        if epoch >= start_scheduler and (epoch - start_scheduler) % config.scheduler_step == 0:
            scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss.item(),
            #"loss_computation_time": loss_computation_time,
            "learning_rate": scheduler.get_last_lr()[0],
            **{key: loss.item() for key,loss in losses.items()},
            **{f"weight_{key}": value for key, value in dg_NVs.lambdas.items()}
        })

        # Save the model checkpoint
        if (epoch % 100 == 0) and (epoch != 0):
            torch.save(dg_NVs, f"./Navier-Stokes/models/{wandb_config.name}.pth")

    # Save final model
    torch.save(dg_NVs, f"./Navier-Stokes/models/{wandb_config.name}.pth")
    wandb.save(f"./Navier-Stokes/models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()
    return dg_NVs


