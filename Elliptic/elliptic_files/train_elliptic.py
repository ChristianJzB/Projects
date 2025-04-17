import time
import wandb

import torch
from torch.utils.data import Dataset,DataLoader

import numpy as np
from scipy.stats import qmc

from .FEM_Solver import FEMSolver
from .elliptic import Elliptic


def samples_param(size, nparam, min_val=-1, max_val=1,seed = 65647437836358831880808032086803839626):
    """Sample parameters uniformly from a specified range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(min_val, max_val, size=(size, nparam))

class dGDataset(Dataset):
    def __init__(self, size,param = None,nparam = 2):
        self.nparam = nparam
        self.size = size

        self.data_int, self.left_bc, self.right_bc = self.generate_data(size, param)

    def generate_data(self, size, param = None, seed = 65647437836358831880808032086803839626):
        #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        x = sampler.random(n=size)

        if param is None:
            param = samples_param(size=size, nparam=self.nparam)
        else:
            param = param[:size,:]

        x_tensor = torch.Tensor(x)
        param_tensor = torch.Tensor(param)

        data_int = torch.cat([x_tensor, param_tensor], axis=1).float()
        left_bc = torch.cat([torch.zeros_like(x_tensor).float(), param_tensor], axis=1).float()
        right_bc = torch.cat([torch.ones_like(x_tensor).float(), param_tensor], axis=1).float()

        return data_int, left_bc, right_bc

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data_int[index], self.left_bc[index], self.right_bc[index]

def generate_data(size, param = None, nparam = 2, seed = 65647437836358831880808032086803839626):
        #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        x = sampler.random(n=size)

        if param is None:
            param = samples_param(size=size, nparam= nparam)
        else:
            param = param[:size,:]

        x_tensor = torch.Tensor(x)
        param_tensor = torch.Tensor(param)

        data_int = torch.cat([x_tensor, param_tensor], axis=1).float()
        left_bc = torch.cat([torch.zeros_like(x_tensor).float(), param_tensor], axis=1).float()
        right_bc = torch.cat([torch.ones_like(x_tensor).float(), param_tensor], axis=1).float()

        return data_int, left_bc, right_bc  


def generate_test_data(size, lam = 1/4, M = 2,param = None,vert=30, nparam=2):
    """
    Generate test data using the FEM solver for a specified number of samples and parameters.

    Parameters:
    - size (int): Number of test samples to generate.
    - vert (int): Number of vertices for the FEMSolver mesh.
    - nparam (int): Number of parameters for the test.

    Returns:
    - x_test (np.ndarray): The test points (vertices).
    - test_samples_param (np.ndarray): Parameters for the test.
    - test_data (np.ndarray): FEM solver results for each test sample.
    """
    # Sample parameters
    if param is None:
        test_samples_param = samples_param(size, nparam=nparam)
    else:
        test_samples_param = param[:size,:]

    solver = FEMSolver(np.zeros(nparam), lam, M, vert=vert)

    # Preallocate array for test data
    test_data = np.zeros((size, vert + 1))

    # Loop through and solve for each parameter set
    for i in range(size):
        try:
            # Solve the FEM problem
            solver.theta = test_samples_param[i, :]  # Update the parameter vector
            solver.uh = None
            solution = solver.solve()
            test_data[i, :] = solution.x.array
        except Exception as e:
            print(f"FEM solver failed for sample {i}: {str(e)}")
            continue

    # Get the test points (vertices)
    try:
        x_test = solver.solution_array()[0][:, 0].reshape(-1, 1)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve test points: {str(e)}")

    return x_test, test_samples_param, test_data


def train_elliptic(config, device):
    """Train the PINN using the Adam optimizer with early stopping, computing test loss after each epoch."""
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    torch.manual_seed(config.seed)
    start_scheduler = int(config.epochs * config.start_scheduler )

    data_parameters = samples_param(config.nn_samples*2, nparam=config.KL_expansion)
    
    param_train, param_test = data_parameters[:config.nn_samples,:],  data_parameters[config.nn_samples:,:]

    dataset = dGDataset(size = config.nn_samples, param=param_train)

    x_val,param_val, sol_val = generate_test_data(config.nn_samples,param =param_test, vert=30,
                                                  nparam=config.KL_expansion,M = config.KL_expansion)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    dg_elliptic = Elliptic(config=config, device=device)

    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_elliptic.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)  # Exponential decay scheduler

    for epoch in range(config.epochs):
        epoch_train_loss = 0.0  # Accumulate loss over all batches for this epoch
        
        for data_int, left_bc, right_bc in dataloader:
            data_int, left_bc, right_bc = data_int.to(device), left_bc.to(device), right_bc.to(device)

            optimizer.zero_grad()

            start_time = time.time()
            total_loss,losses = dg_elliptic.total_loss(data_int, left_bc, right_bc, loss_fn)
            loss_computation_time = time.time() - start_time  # Time taken for loss computation

            total_loss.backward()
            optimizer.step()
            
            # Accumulate the batch loss into the epoch loss
            epoch_train_loss += total_loss.item()

        # Calculate the average loss for the epoch
        epoch_train_loss /= len(dataloader)

        # Compute the test loss at the end of the epoch
        test_loss_current = compute_mean_error(dg_elliptic.model, param_val, x_val, sol_val)

        # Scheduler step
        if epoch >= start_scheduler and (epoch - start_scheduler) % config.scheduler_step == 0:
            scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "test_loss": test_loss_current,
            "loss_computation_time": loss_computation_time,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"loss_{key}": value.item() for key, value in losses.items()},
        })
        # Save the model checkpoint
        if (epoch % 1000 == 0) and (epoch != 0):
            torch.save(dg_elliptic, f"./Elliptic/models/{wandb_config.name}.pth")

    # Save final model
    torch.save(dg_elliptic, f"./Elliptic/models/{wandb_config.name}.pth")
    wandb.save(f"./Elliptic/models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()

    return dg_elliptic


def compute_mean_error(model, parameters_test, t, y_numerical):
    """
    Compute the mean error between the numerical solution and model predictions.

    Parameters:
    model : torch.nn.Module
        The trained PyTorch model for predictions.
    parameters_test : list of tuples
        Test parameters to be used for generating test data.
    t : numpy.ndarray
        Time or spatial variable.
    y_numerical : numpy.ndarray
        Numerical solution for comparison.
    w : any
        Weight or other identifier for mean_error storage.
    act : str
        Activation type for mean_error storage.

    Returns:
    None: Updates mean_error in place.
    """
    error = []
    
    for n, pr1 in enumerate(parameters_test):
        # Generate test data
        
        data_test = np.hstack((t, np.ones((t.shape[0],pr1.shape[0])) * pr1))
        
        # Get the numerical solution for this test case
        numerical_sol = y_numerical[n, :]

        # Predict using the model
        u_pred = model(torch.tensor(data_test).float()).detach().cpu().numpy()

        # Reshape the numerical solution to match the prediction shape
        numerical_sol = numerical_sol.reshape(u_pred.shape)

        # Calculate the error
        error.append(np.linalg.norm(numerical_sol - u_pred, ord=2) / np.linalg.norm(numerical_sol, ord=2))

    return np.mean(error)

