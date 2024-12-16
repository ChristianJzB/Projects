import numpy as np
#from pyDOE import lhs
from NN import DNN
from scipy.stats import qmc

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from elliptic_files.FEM_Solver import  RootFinder,FEMSolver



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


def generate_noisy_obs(obs, theta_t=np.array([0.098, 0.430]),lam = 1 /4, M = 2, vert=1000, mean=0, std=np.sqrt(1e-4)):
    """
    Generates noisy observations for given parameters and observation points.
    
    Parameters:
    - obs: Number of observation points (integer)
    - theta_t: True parameter values for the FEM solver (default is np.array([0.098, 0.430]))
    - vert: Number of vertices in the FEM mesh (default is 1000)
    - mean: Mean of the noise distribution (default is 0)
    - std: Standard deviation of the noise distribution (default is sqrt(1e-4))

    Returns:
    - obs_points: The observation points (shape: (obs, 1))
    - sol_test: Noisy solution points (shape: (obs, 1))
    """

    # Solve the FEM problem using the given theta values and roots
    solver = FEMSolver(theta_t, lam, M, vert=vert)
    solution = solver.solve()

    # Extract observation points and the solution points
    obs_points, sol_points = solver.solution_array()

    # Choose indices for the observation points, including start and end
    indices = np.linspace(0, obs_points.shape[0] - 1, num=obs, dtype=int)

    # Select the observation and solution points based on the indices
    obs_points, sol_points = obs_points[indices, 0], sol_points[indices]

    # Generate noise and add it to the solution points
    noise_sol_points = add_noise(sol_points, mean, std)

    # Ensure proper reshaping of observation and solution points
    obs_points = obs_points.reshape(-1, 1)
    sol_test = noise_sol_points.reshape(-1, 1)

    return obs_points, sol_test


def add_noise(solution, mean, std, seed = 65647437836358831880808032086803839626):
    """
    Adds Gaussian noise to the solution.

    Parameters:
    - solution: The clean solution points
    - mean: Mean of the noise
    - std: Standard deviation of the noise

    Returns:
    - Noisy solution points
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, std, solution.shape)
    return solution + noise


def compute_loss(pinn, data_int, left_bc, right_bc, loss_fn):
    """Compute the total loss for given data."""
    pde_pred = pinn.de(data_int)
    left_bc_pred = pinn.bc_l(left_bc)
    right_bc_pred = pinn.bc_r(right_bc)

    zeros = torch.zeros_like(pde_pred)

    loss_pde = loss_fn(pde_pred, zeros)
    loss_lbc = loss_fn(left_bc_pred, zeros)
    loss_rbc = loss_fn(right_bc_pred, zeros)

    total_loss = loss_pde + loss_lbc + loss_rbc
    return total_loss, loss_pde, loss_lbc, loss_rbc

def train_adam(dataloader, pinn, loss_fn, optimizer, epochs, parameters_test, t, y_numerical, device, patience=np.inf):
    """Train the PINN using the Adam optimizer with early stopping, computing test loss after each epoch."""
    print("Starting Adam Training")
    
    train_loss, test_loss = [], []
    
    # Variables for early stopping
    best_test_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0  # Accumulate loss over all batches for this epoch
        
        for data_int, left_bc, right_bc in dataloader:
            data_int, left_bc, right_bc = data_int.to(device), left_bc.to(device), right_bc.to(device)

            data_int = Variable(data_int, requires_grad=True)
            left_bc, right_bc = Variable(left_bc, requires_grad=True), Variable(right_bc, requires_grad=True)
            
            optimizer.zero_grad()

            total_loss, loss_pde, loss_lbc, loss_rbc = compute_loss(pinn, data_int, left_bc, right_bc, loss_fn)

            total_loss.backward()
            optimizer.step()
            
            # Accumulate the batch loss into the epoch loss
            epoch_train_loss += total_loss.item()

        # Calculate the average loss for the epoch
        epoch_train_loss /= len(dataloader)
        train_loss.append(epoch_train_loss)

        # Compute the test loss at the end of the epoch
        test_loss_current = compute_mean_error(pinn, parameters_test, t, y_numerical)
        test_loss.append(test_loss_current)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {epoch_train_loss:.5e}, Test Loss: {test_loss_current:.5e}')
        
        # Early stopping logic
        if test_loss_current < best_test_loss:
            best_test_loss = test_loss_current
            best_model = pinn.state_dict()  # Save the best model parameters
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1
        
        # If patience is exceeded, stop the training
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best test loss: {best_test_loss:.5e}")
            pinn.load_state_dict(best_model)  # Restore the best model
            break
    
    # Convert losses to numpy arrays
    return np.array(train_loss), np.array(test_loss)




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


def plot_mcmc_parameters(samples, theta_t, tracker, nobs, alpha_samples):
    """
    Plot MCMC parameter samples with trace plots, autocorrelation, KDE, and iteration tracker.
    
    Parameters:
    - samples: Array of MCMC samples.
    - theta_t: True parameter values for comparison.
    - tracker: Tracker for MCMC iterations.
    - nobs: Indices for the samples.
    - alpha_samples: The shape of the samples (used to iterate).
    """
    num_params = alpha_samples.shape[1]

    for i in range(num_params):
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(fr"MCMC: $\theta$ {i + 1}", fontsize=16)
        plt.subplots_adjust(hspace=0.25)

        # Trace Plot
        axs[0, 0].plot(samples[nobs[-1]][:, i], label='Chain', color='blue')
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_title("Chain")
        axs[0, 0].legend()

        # Autocorrelation Plot
        plot_acf(samples[nobs[-1]][:, i], lags=200, ax=axs[0, 1], color='red')
        axs[0, 1].set_xlabel(r"$\text{Lag}$")
        axs[0, 1].set_ylabel(r"$\text{Autocorrelation}$")
        axs[0, 1].set_title("Autocorrelation")

        # KDE Plot
        sns.kdeplot(samples[nobs[-1]][200:, i], ax=axs[1, 0], bw_adjust=2, color='orange')
        axs[1, 0].axvline(x=theta_t[i], color='black', label=r"$\hat{\theta}$", linestyle='--')
        axs[1, 0].set_xlabel(r"$\theta$")
        axs[1, 0].set_title("Density")
        #axs[1, 0].legend()

        # Iteration Tracker
        axs[1, 1].plot(tracker[nobs[-1]], color='black', label=fr'$dt$ Dynamic')
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_title(fr'$dt$ Dynamic')
        #axs[1, 1].legend()

        plt.show()

def histogram_(x, bins=100):
    # Calculate histogram
    counts, bin_edges = np.histogram(x, bins=bins)

    # Normalize counts to form a probability density
    counts = counts / (sum(counts) * np.diff(bin_edges))

    # Calculate the bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, counts


def plot_posterior_distributions(samples, nobs, samp_num=None,num_bins=100, sufix="", labelsuf="NN", a1=-1, b1=1, a2=-1, b2=1):
    """
    Plot posterior distributions for the parameters using the custom histogram_ function.
    
    Parameters:
    - samples: Array of MCMC samples.
    - nobs: Indices for the samples.
    - samp_num: Numerical samples for comparison.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Posterior Distributions {sufix}", color="blue", fontsize=16)
    plt.subplots_adjust(hspace=0.25)

    # Define number of bins for histogram

    # Plotting for parameter 1 (theta_1)
    for obs in nobs:
        bin_centers, counts = histogram_(samples[obs][:, 0], bins=num_bins)
        axs[0].plot(bin_centers, counts, linestyle='-', alpha=0.6, label=f"{labelsuf}({str(obs)})")
    
    # Plot for the comparison sample (samp_num)
    if samp_num is not None:
        bin_centers, counts = histogram_(samp_num[:, 0], bins=num_bins)
        axs[0].plot(bin_centers, counts, linestyle="--", color="black", alpha=0.6, linewidth=0.75, label="FEM")
    
    axs[0].set_xlabel(r"$\theta_{1}$")
    axs[0].set_title(r"$\theta_1$")
    axs[0].legend(frameon=False)
    axs[0].set_xlim((a1, b1))

    # Plotting for parameter 2 (theta_2)
    for obs in nobs:
        bin_centers, counts = histogram_(samples[obs][:, 1], bins=num_bins)
        axs[1].plot(bin_centers, counts, linestyle='-', alpha=0.6, label=f"{labelsuf}({str(obs)})")

    if samp_num is not None:
        bin_centers, counts = histogram_(samp_num[:, 1], bins=num_bins)
        axs[1].plot(bin_centers, counts, linestyle="--", color="black", alpha=0.6, linewidth=0.75, label="FEM")

    axs[1].set_xlabel(r"$\theta_{2}$")
    axs[1].set_title(r"$\theta_2$")
    axs[1].legend(frameon=False)
    axs[1].set_xlim((a2, b2))

    plt.show()


def plot_loss_dynamics(hl,w, nobs, sample_size, model_dir='./Models/', optimizer="adam",
                        adltrain=0.09, adltest=0.09):
    """
    Function to plot training and validation loss dynamics for ADAM optimizer 
    with optional LBFGS optimizer and varying batch sizes.
    
    Parameters:
    - w: Number of weights (integer)
    - nobs: Number of observations (integer)
    - sample_size: List of batch sizes
    - model_dir: Directory where the model loss files are stored (default: './Models/')
    - file_prefix: Prefix for the model loss files (default: 'model_')
    - include_lbfgs: Whether to include LBFGS results (default: True)
    - adltrain: Y-axis limit for ADAM training loss (default: 0.09)
    - adltest: Y-axis limit for ADAM test loss (default: 0.09)
    - lbfgsltrain: Y-axis limit for LBFGS training loss (default: 0.05)
    - lbfgsltest: Y-axis limit for LBFGS test loss (default: 0.05)
    """
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5 ))
    fig.suptitle(f"NN: Weights = {w} and N={nobs}", color="b", fontsize=14)
    

    # Loop through each batch size in sample_size
    for ss in sample_size:
        # Load the data for ADAM optimizer (training and validation loss)
        adam_train_loss = np.load(f'{model_dir}{optimizer}_only_train_hiddenl{hl}_w{w}_N{nobs}_batch{ss}.npy')
        adam_test_loss = np.load(f'{model_dir}{optimizer}_only_test_hiddenl{hl}_w{w}_N{nobs}_batch{ss}.npy')

        # Plot Training Loss for ADAM optimizer
        axs[0].plot(adam_train_loss, label=f"Batch Size: {ss}")
        axs[0].set_title(f"Training Loss: {optimizer} optimizer (batch {ss})")
        axs[0].set_ylabel("MSE")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylim((0, adltrain))

        # Plot Validation Loss for ADAM optimizer
        axs[1].plot(adam_test_loss, label=f"Batch Size: {ss}")
        axs[1].set_title(f"Validation Loss: {optimizer} optimizer (batch {ss})")
        axs[1].set_ylabel("MSE")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylim((0, adltest))

    # Add a global legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=len(sample_size), frameon=False, fontsize=10)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Show the plot
    plt.show()



def plot_nn_vs_fem(weights, param_test, x_test, sol_test, sample_size, model_path_template,model_class, hn=1,nobs=500,samples=2):
    """
    This function compares the predicted solution of a neural network model with the FEM numerical solution
    and visualizes them along with errors.

    Parameters:
    -----------
    - weights : list of int
        The list of weights used in the neural network layers.
    - param_test : np.array
        The parameter test set (theta values) for testing.
    - x_test : np.array
        The test input data points for evaluating the model.
    - sol_test : np.array
        The true solution (numerical solution from FEM).
    - sample_size : list of int
        Different batch sizes to evaluate the neural networks.
    - model_path_template : str
        A template for the model path that will load the pre-trained models.
        Example: './Models/1dElliptic_PDE2_w{w}_N{nobs}_batch{ss}.pt'
    - nobs : int, optional (default=500)
        Number of observations.
    """
    
    # Define the layout for the plot
    fig, axs = plt.subplots(samples, len(weights), figsize=(15, 8), sharex=True, sharey=True, layout="constrained")
    colors = ["blue", "orange", "green"]
    
    fig.suptitle(f"NN: N={nobs}", color="red")
    
    # Loop over different weights
    for j, w in enumerate(weights):
        for n, pr1 in enumerate(param_test[:samples]):
            # Prepare the test data by adding parameters as columns
            data_test = np.hstack((x_test, np.ones_like(x_test) * pr1[0], np.ones_like(x_test) * pr1[1]))
            numerical_sol = sol_test[n, :]
            
            # Plot the title of each subplot
            axs[0, j].set_title(fr"Number of weights of NN: {w}")
            
            # Loop through different sample sizes
            for i, (ss, cl) in enumerate(zip(sample_size, colors)):
                layers = [3] + hn*[w] + [1]  # Define layers for DNN
                
                # Load the pre-trained model
                model = model_class(layers)
                model_path = model_path_template.format(w=w, nobs=nobs, ss=ss)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                
                # Predict the output using the model
                u_pred = model(torch.tensor(data_test).float()).detach().cpu().numpy()
                numerical_sol = numerical_sol.reshape(u_pred.shape)
                
                # Calculate the relative error
                error = np.linalg.norm(numerical_sol - u_pred, ord=2) / (np.linalg.norm(numerical_sol, ord=2) * 30)
                
                # Plot the NN prediction vs the numerical solution
                axs[n, j].scatter(x_test, u_pred, label=fr"NN_{ss}: e = {error:.2e}", color=cl, alpha=0.6)
            
            # Plot the FEM (numerical) solution
            axs[n, j].plot(x_test, numerical_sol, label="FEM", color="black", linestyle="--", linewidth=1)
            
            # Add annotations and labels
            axs[n, 0].annotate(fr"$\theta_1$ = {pr1[0]:.1g}, $\theta_2$ = {pr1[1]:.1g}", xy=(0, 0), xytext=(0.5, 0))
            axs[-1, j].set_xlabel("x")
            axs[n, 0].set_ylabel("u(x)")
            axs[n, j].legend(frameon=False)
    
    plt.show()


def compute_max_error(N, NNpr,hn=1,modelclass=DNN,vert=30, grid=75, model_dir='./Models/', model_prefix='1dElliptic_adam_PDE2',batch = 150):
    """
    Compute the maximum error between the FEM solution and a surrogate neural network for different parameters and observation sizes.
    
    Parameters:
    - parameter: A list of parameter values (1D grid).
    - N: A list of observation sizes for loading pre-trained neural networks.
    - npr: Number of neurons per layer in the surrogate model.
    - solver: An initialized FEM solver object.
    - vert: Number of vertices for the FEM solver (default: 30).
    - grid: Grid size for the parameter space (default: 75).
    - layers: List specifying the architecture of the neural network (default: [3, npr, 1]).
    - model_dir: Directory where the pre-trained models are stored (default: './Models/').
    - model_prefix: Prefix for the model filename (default: '1dElliptic_adam_PDE2').
    
    Returns:
    - results: A 3D array (grid x grid x len(N)) of maximum errors.
    """

    layers = [3] + hn*[NNpr] + [1]
    parameter  = np.linspace(-1,1,grid)

    # Initialize results array
    results = np.zeros((grid, grid, len(N)))

    # Create meshgrid for plotting later
    X, Y = np.meshgrid(parameter, parameter)

    solver = FEMSolver(np.zeros(2), vert = vert)

    # Main loop over parameter space
    for i, par1 in enumerate(parameter):
        for j, par2 in enumerate(parameter):
            pr = np.array([par1, par2])

            # Update the solver with new parameter values
            solver.theta = pr  # Update the parameter vector
            solver.uh = None  # Reset the FEM solution
            solver.solve()

            # Get FEM solution arrays
            x_FEM, y_FEM = solver.solution_array()
            x_FEM, y_FEM = x_FEM[:, 0].reshape(-1, 1), y_FEM.reshape(-1, 1)

            # Loop over observation sizes N
            for z, nobs in enumerate(N):
                model_path = f"{model_dir}/{model_prefix}_w{NNpr}_N{nobs}_batch{batch}.pt"

                try:
                    # Load pre-trained neural network model for given N
                    model = modelclass(layers)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()

                    # Prepare the input data for the surrogate model
                    data = torch.tensor(np.hstack((x_FEM, np.ones((x_FEM.shape[0], pr.shape[0])) * pr))).float()
                    surg = model(data).detach().cpu().numpy().reshape(-1, 1)

                    # Compute the maximum error between FEM and surrogate
                    max_error = np.max(np.abs(y_FEM - surg))
                    results[i, j, z] = max_error

                except FileNotFoundError:
                    print(f"Model file not found for N={nobs} at path {model_path}")
                    results[i, j, z] = np.nan  # Mark missing model data

    return results, X, Y


def compute_relative_error(N, NNpr,hn=1,modelclass=DNN,vert=30, grid=75, model_dir='./Models/', model_prefix='1dElliptic_adam_PDE2',batch = 150):
    """
    Compute the maximum error between the FEM solution and a surrogate neural network for different parameters and observation sizes.
    
    Parameters:
    - parameter: A list of parameter values (1D grid).
    - N: A list of observation sizes for loading pre-trained neural networks.
    - npr: Number of neurons per layer in the surrogate model.
    - solver: An initialized FEM solver object.
    - vert: Number of vertices for the FEM solver (default: 30).
    - grid: Grid size for the parameter space (default: 75).
    - layers: List specifying the architecture of the neural network (default: [3, npr, 1]).
    - model_dir: Directory where the pre-trained models are stored (default: './Models/').
    - model_prefix: Prefix for the model filename (default: '1dElliptic_adam_PDE2').
    
    Returns:
    - results: A 3D array (grid x grid x len(N)) of maximum errors.
    """

    layers = [3] + hn*[NNpr] + [1]
    parameter  = np.linspace(-1,1,grid)

    # Initialize results array
    results = np.zeros((grid, grid, len(N)))

    # Create meshgrid for plotting later
    X, Y = np.meshgrid(parameter, parameter)

    solver = FEMSolver(np.zeros(2), vert = vert)

    # Main loop over parameter space
    for i, par1 in enumerate(parameter):
        for j, par2 in enumerate(parameter):
            pr = np.array([par1, par2])

            # Update the solver with new parameter values
            solver.theta = pr  # Update the parameter vector
            solver.uh = None  # Reset the FEM solution
            solver.solve()

            # Get FEM solution arrays
            x_FEM, y_FEM = solver.solution_array()
            x_FEM, y_FEM = x_FEM[:, 0].reshape(-1, 1), y_FEM.reshape(-1, 1)

            # Loop over observation sizes N
            for z, nobs in enumerate(N):
                model_path = f"{model_dir}/{model_prefix}_w{NNpr}_N{nobs}_batch{batch}.pt"

                try:
                    # Load pre-trained neural network model for given N
                    model = modelclass(layers)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()

                    # Prepare the input data for the surrogate model
                    data = torch.tensor(np.hstack((x_FEM, np.ones((x_FEM.shape[0], pr.shape[0])) * pr))).float()
                    surg = model(data).detach().cpu().numpy().reshape(-1, 1)

                    # Compute the maximum error between FEM and surrogate
                    rel_error = np.linalg.norm(y_FEM - surg, ord=2)/np.linalg.norm(y_FEM, ord=2)
                    results[i, j, z] = rel_error

                except FileNotFoundError:
                    print(f"Model file not found for N={nobs} at path {model_path}")
                    results[i, j, z] = np.nan  # Mark missing model data

    return results, X, Y


def plot_max_errors(N, results, X, Y,NNpr):
    """
    Plot the maximum error between FEM and surrogate models for each observation size N.

    Parameters:
    - parameter: A list of parameter values (1D grid).
    - N: A list of observation sizes.
    - results: A 3D array (grid x grid x len(N)) of maximum errors.
    - X: Meshgrid of parameter 1.
    - Y: Meshgrid of parameter 2.
    """

    # Plotting
    fig, axs = plt.subplots(1, len(N), figsize=(15, 3), layout="constrained", sharey=True)
    fig.suptitle(fr"NN: {NNpr}, $\max_{{x}} | u(x, \theta) - \hat{{u}}(x, \theta)|$", fontsize=12)

    z_min, z_max = results.min(), results.max()

    for z, key in enumerate(N):
        Z = results[:, :, z]  # Extract the results for current N

        # Set the color scale limits based on the first plot
        # Plot the heatmap for each N
        pcm = axs[z].pcolormesh(X, Y, Z, shading='auto', vmin=z_min, vmax=z_max)
        axs[z].set_xlabel(r"$\theta_1$")
        axs[z].set_title(f"NN({key})")
        plt.colorbar(pcm, ax=axs[z])

    axs[0].set_ylabel(r"$\theta_2$")  # Set common y-axis label
    plt.show()

