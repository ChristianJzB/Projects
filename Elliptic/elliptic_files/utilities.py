import numpy as np
from scipy.stats import qmc

import torch
# import torch.optim as optim
# from torch.utils.data import Dataset
# from torch.autograd import Variable

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from .FEM_Solver import FEMSolver

from .train_elliptic import train_elliptic,generate_data,samples_param


def generate_data_elliptic(size, param = None, nparam = 2, seed = 65647437836358831880808032086803839626):
        #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        x = sampler.random(n=size)

        if param is None:
            param = samples_param(size=size, nparam= nparam,seed=seed)
        else:
            param = param[:size,:]

        x_tensor = torch.Tensor(x)
        param_tensor = torch.Tensor(param)

        data_int = torch.cat([x_tensor, param_tensor], axis=1).float()
        left_bc = torch.cat([torch.zeros_like(x_tensor).float(), param_tensor], axis=1).float()
        right_bc = torch.cat([torch.ones_like(x_tensor).float(), param_tensor], axis=1).float()

        return data_int, left_bc, right_bc  


def deepgala_data_fit(samples,nparameters,device,seed = 65647437836358831880808032086803839626):
    # data_parameters = samples_param(samples*2, nparam=nparameters)
    # param_train, param_test = data_parameters[:samples,:],  data_parameters[samples:,:]
    data_int,left_bc,right_bc = generate_data_elliptic(samples, nparam=nparameters,seed=seed)
    data_int,left_bc,right_bc  = data_int.to(device),left_bc.to(device),right_bc.to(device)
    
    dgala_data = {"data_fit": {"pde":data_int, "left_bc":left_bc,"right_bc":right_bc}, 
                "class_method": {"pde": ["elliptic_pde"], "left_bc":["u"],"right_bc":["u"]},
                "outputs": {"pde": ["elliptic"], "left_bc": ["ubcl"],"right_bc":["ubcr"]}}
    return dgala_data

def generate_noisy_obs(obs, theta_t=np.array([0.098, 0.430]),vert=1000, mean=0, std=np.sqrt(1e-4)):
    """
    Generates noisy observations for given parameters and observation points.
    """

    # Solve the FEM problem using the given theta values and roots
    solver = FEMSolver(theta_t,vert=vert)
    solution = solver.solve()

    # Extract observation points and the solution points
    obs_points, sol_points = solver.solution_array()

    # Choose indices for the observation points, including start and end
    obs_points = np.linspace(0.2,0.8,obs).reshape(-1,1)

    # Select the observation and solution points based on the indices
    sol_points = solver.eval_at_points(obs_points)

    # Generate noise and add it to the solution points
    noise_sol_points = add_noise(sol_points, mean, std)

    # Ensure proper reshaping of observation and solution points
    obs_points = obs_points.reshape(-1, 1)
    sol_test = noise_sol_points.reshape(-1, 1)

    return obs_points, sol_test

def add_noise(solution, mean, std, seed = 0):
    """
    Adds Gaussian noise to the solution.
    """
    # rng = np.random.default_rng(seed)
    # noise = rng.normal(mean, std, solution.shape)
    np.random.seed(seed)
    noise = np.random.normal(mean, std, solution.shape)
    return solution + noise


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


def compute_max_error(N,vert=30, grid=75):
    """
    Compute the maximum error between the FEM solution and a surrogate neural network for different parameters and observation sizes.
    """
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

            data = torch.tensor(np.hstack((x_FEM, np.ones((x_FEM.shape[0], pr.shape[0])) * pr))).float()

            # Loop over observation sizes N
            for z, sample in enumerate(N):
                model_path = f"./models/MDNN_s{sample}.pth"

                try:
                    elliptic = torch.load(model_path)
                    elliptic.eval()

                    surg = elliptic.model(data).detach().cpu().numpy().reshape(-1, 1)

                    # Compute the maximum error between FEM and surrogate
                    max_error = np.max(np.abs(y_FEM - surg))
                    results[i, j, z] = max_error

                except FileNotFoundError:
                    print(f"Model file not found for N={sample} at path {model_path}")
                    results[i, j, z] = np.nan  # Mark missing model data

    return results, X, Y


def plot_max_errors(N, results, X, Y,NNpr):
    """
    Plot the maximum error between FEM and surrogate models for each observation size N.
    """
    # Plotting
    fig, axs = plt.subplots(1, len(N), figsize=(15, 3), layout="constrained", sharey=True)
    fig.suptitle(fr"NN: {NNpr}, $\max_{{x}} | u(x, \theta) - \hat{{u}}(x, \theta)|$", fontsize=12)


    for z, key in enumerate(N):
        Z = results[:, :, z]  # Extract the results for current N

        # Plot the heatmap for each N
        pcm = axs[z].pcolormesh(X, Y, Z, shading='auto')
        axs[z].set_xlabel(r"$\theta_1$")
        axs[z].set_title(f"NN({key})")
        plt.colorbar(pcm, ax=axs[z])

    axs[0].set_ylabel(r"$\theta_2$")  # Set common y-axis label
    plt.show()

