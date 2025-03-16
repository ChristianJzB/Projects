
import torch
import numpy as np
from nv_files.Pseudo_Spectral_Solver import VorticitySolver2D
from nv_files.Field_Generator import omega0_samples_torch
from nv_files.train_nvs import ic_vort_samples,data_vor_set_preparing
from nv_files.data_generator import UniformSampler


def generate_noisy_obs(obs,noise_level = 1e-3,NKL = 2 ,dim_obs = 128, seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.linspace(0, 1, dim_obs)*2*torch.pi  # Spatial grid in X direction
    Y = torch.linspace(0, 1, dim_obs)*2*torch.pi  # Spatial grid in Y direction
    X, Y = torch.meshgrid(X, Y, indexing='ij' )  # Create meshgrid for X, Y

    # Generate uniformly distributed values for `theta` in the range [-1, 1]
    theta = torch.rand(NKL, 2, 1) * 2 - 1  # Uniform(-1, 1)

    w0 = omega0_samples_torch(X, Y, theta,d = 5,tau=np.sqrt(2))

    def force_function(X, Y):
        return  (np.sin(X + Y) + np.cos(X + Y))

    solver = VorticitySolver2D(N=dim_obs, L=2*np.pi, T=2.0, nu=1e-2, dt=5e-4,num_sol=100, method='CN',force = force_function)

    obs_res = solver.run_simulation( np.array(w0[:,:,0]))

    noise = np.random.normal(0, np.sqrt(noise_level), obs_res[-1].shape)

    noisy_obs = obs_res[-1] + noise

    obs_input_ = torch.cat((X.reshape(-1,1), Y.reshape(-1,1), 2*torch.ones_like(X.reshape(-1,1))), dim=1)
    
    noisy_obs_ = noisy_obs.reshape(-1,1)

    # Randomly select 10 unique row indices
    indices = torch.randperm(noisy_obs_.shape[0])[:obs]

    # Sort the indices in ascending order
    sorted_indices = indices.sort().values

    obs_input = obs_input_[sorted_indices]
    noisy_obs = noisy_obs_[sorted_indices]

    return obs_input,noisy_obs


def deepgala_data_fit(config,device):
    initial_points,w0,theta = ic_vort_samples(config)

    batch_size_interior = config.chunks*config.points_per_chunk

    dom = torch.tensor([[0, 2 * torch.pi], [0, 2 * torch.pi],[0,config.time_domain]])

    samples_interior = iter(UniformSampler(dom, batch_size_interior))

    batch = next(samples_interior)

    sorted_batch,initial_points_,initial_condition = data_vor_set_preparing(config,batch, 
                                                    initial_points,w0,theta,batch_size_interior,0)
    sorted_batch,initial_points_,initial_condition = sorted_batch.to(device),initial_points_.to(device),initial_condition.to(device)
    data_trainig = {"data_fit": {"pde":sorted_batch, "initial_conditions":(initial_condition,initial_points_)}, 
                "class_method": {"pde": ["nv_pde"], "initial_conditions":["w"]},
                "outputs": {"pde": ["nvs", "cond"], "initial_conditions": ["w0"]}}
    
    return data_trainig

                                        