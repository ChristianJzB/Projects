import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from FEM_Solver import  RootFinder,FEMSolver
from Deep_Solvers.Inverse_Problems.Base.MCMC import MetropolisHastingsSampler, MoreauYosidaPrior
from GaLa import llaplace
from GaLa2 import llaplace as llaplace2

from utilities import *
from NN import DNN

import pyro
from pyro.infer import MCMC, mcmc
import pyro.distributions as dist

import numpy as np
import time

#np.random.seed(1234)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


lam = 1 / 4
M = 2

finder = RootFinder(lam, M)
roots = finder.find_roots()

def k_function(data_domain,w=roots):
    
    x = data_domain[:,0].reshape(-1,1)
    theta = data_domain[:,1:].reshape(x.shape[0],-1)

    A = torch.sqrt(1 / ( (1/8)*(5 + (w / 2)**2) +  (torch.sin(2*w) / (4*w))*((w / 4)**2 - 1) - (torch.cos(2*w)/8)))
    
    bn =  A*(torch.sin(w*x) + ((w)/(4))*torch.cos(w*x))

    an = torch.sqrt(8 / (w**2 + 16))

    return torch.sum(an*bn*theta,dim=1)

def de(self,data_domain,k = k_function,roots = torch.tensor(roots,dtype=float).reshape(1,-1)):
    """ The pytorch autograd version of calculating residual """
    u = self(data_domain)

    du = torch.autograd.grad(
        u, data_domain, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    k = k(data_domain,roots)
    
    ddu_x = torch.autograd.grad(
        torch.exp(k).reshape(-1,1)*du[:,0].reshape(-1,1),data_domain, 
        grad_outputs=torch.ones_like(du[:,0].reshape(-1,1)),
        retain_graph=True,
        create_graph=True
        )[0]
        
    return ddu_x[:,0].reshape(-1,1) + 4*data_domain[:,0].reshape(-1,1)


def bc_l(self,data_bc):
    u = self(data_bc)
    return u.reshape(-1,1)

def bc_r(self,data_bc):
    u = self(data_bc)
    return u.reshape(-1,1)- 2

DNN.de = de
DNN.bc_l = bc_l
DNN.bc_r = bc_r


# epochs = 500

# N = [500,1000,2000,3000,4000,5000]
# sample_size = [75,100,125,150]
# weights = [5,10,15,20]

# lr = 0.01

# data_parameters = samples_param(10000, nparam=2)

# param_train, param_test = data_parameters[:5000,:],  data_parameters[5000:,:]


# for w in weights:
#     for nobs in N:
#         dataset = dGDataset(size = nobs, param=param_train)

#         x_val,param_val, sol_val = generate_test_data(nobs,param =param_test, vert=30)

#         for ss in sample_size:
#             layers = [3] + 1*[w] + [1]

#             model = DNN(layers).to(device)
#             torch.save(model.state_dict(),f"/exports/eddie/scratch/s2113174/experiments/models/prueba_w{w}.pt")

#             dataloader = DataLoader(dataset, batch_size=ss, shuffle=False)

#             loss = torch.nn.MSELoss(reduction ='mean')

#             optimizer = torch.optim.Adam(model.parameters(), lr = lr)

#             loss_adam = train_adam(dataloader,model, loss, optimizer, epochs,param_val,x_val,sol_val, device)

#             optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5000, max_eval=None, 
#                 tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps,line_search_fn="strong_wolfe" )

#             loss_lbfgs = train_lbfgs(dataloader,model, loss, optimizer,param_val,x_val,sol_val, device)
            
#             np.save(f'./models/adam_train_w{w}_N{nobs}_batch{ss}.npy', loss_adam[0])
#             np.save(f'./models/adam_test_w{w}_N{nobs}_batch{ss}.npy', loss_adam[1]) 
#             np.save(f'./models/lbfgs_train_w{w}_N{nobs}_batch{ss}.npy', loss_lbfgs[0]) 
#             np.save(f'./models/lbfgs_test_w{w}_N{nobs}_batch{ss}.npy', loss_lbfgs[1]) 

#             path = f"./models/1dElliptic_PDE2_w{w}_N{nobs}_batch{ss}.pt"
#             torch.save(model.state_dict(),path)

################################# Inverse Problem
# Define the model using a coarse FEM solver

def model(surrogate_fem_solver,synthetic_data, points,noise_std, device='cpu'):


    # Prior for the parameters theta (assuming 2 parameters)
    theta = pyro.sample("theta", dist.Uniform(-1, 1).expand([2]))


    # Use a coarse mesh FEM solver as the surrogate model
    surrogate_fem_solver.theta = theta.detach().cpu().numpy()  # Update the parameter vector

    surrogate_fem_solver.uh = None

    surrogate_fem_solver.solve()

    # Get model predictions at the given points
    model_predictions = torch.Tensor(surrogate_fem_solver.eval_at_points(points)).to(device)

    # Likelihood: Gaussian likelihood with a fixed standard deviation
    y = pyro.sample("obs", dist.Normal(model_predictions, noise_std).to_event(1), obs=torch.Tensor(synthetic_data).to(device))

    return y


def model_dgala(surrogate,synthetic_data, points, sig,device='cpu'):

    # Prior for the parameters theta (assuming 2 parameters)
    theta = pyro.sample("theta", dist.Uniform(-1, 1).expand([2]))

    data_test = torch.cat([points, theta.repeat(points.size(0), 1)], dim=1)

    surg_mu, surg_sigma = surrogate(data_test)

    surg_mu = surg_mu.view(-1, 1)

    surg_sigma = surg_sigma[:, :, 0].view(-1, 1)

    sigma = sig ** 2 + surg_sigma

    # Likelihood: Gaussian likelihood with a fixed standard deviation
    y = pyro.sample("obs", dist.Normal(surg_mu, torch.sqrt(sigma)).to_event(1), obs=torch.Tensor(synthetic_data).to(device))

    return y

def model_nn(surrogate,synthetic_data, points, sig,device='cpu'):

    # Prior for the parameters theta (assuming 2 parameters)
    theta = pyro.sample("theta", dist.Uniform(-1, 1).expand([2]))

    data_test = torch.cat([points, theta.repeat(points.size(0), 1)], dim=1)

    surg_mu = surrogate(data_test)

    surg_mu = surg_mu.view(-1, 1)

    # Likelihood: Gaussian likelihood with a fixed standard deviation
    y = pyro.sample("obs", dist.Normal(surg_mu, sig).to_event(1), obs=torch.Tensor(synthetic_data).to(device))

    return y


obs = 10
theta_th=np.array([0.098, 0.430])
mean, var = 0,[1e-5,1e-4,5e-4,1e-3,5e-3,1e-2]
N = [150,500,1000,1500,2000,2500]
weights = [5,10,20,50]
hidden_layers = [1,2]



for vr in var:
    st = np.sqrt(vr)
    obs_points, obs_sol = generate_noisy_obs(obs, theta_t=theta_th, mean=mean, std=st,vert=200)

    sampler = MetropolisHastingsSampler(obs_points, obs_sol,
                                        sig = st,numerical = True, vert=50,device=device)

    samp_num,dt_tracker_num = sampler.run_sampler(n_chains=100000)

    np.save(f'./models/FEM_var{vr}_adam_Samples.npy', samp_num)

    del sampler, samp_num, dt_tracker_num

    for hl in hidden_layers:
        for w in weights:
            # Timing results storage
            timing_results = {
                'CHR_RWMH':{"NN":[],"dGaLA2":[], "dGaLA":[]},
                'Pyro_RWMH': {"NN":[],"dGaLA2":[], "dGaLA":[]}
            }

            layers = [3] + hl*[w] + [1]

            for nobs in N:
                model = DNN(layers).to(device)
                path = f"./models/1dElliptic_adam_PDE2_hiddenl{hl}_w{w}_N{nobs}_batch150.pt"
                model.load_state_dict(torch.load(path))
                model.eval()

                #####  NN SAmpling using own sampler
                print("NN Sampling")
                sampler = MetropolisHastingsSampler(obs_points, obs_sol,surrogate=model, sig = st,device=device)
                
                start_time_mh = time.time()  # Start timing
                nsamples, dt_progres = sampler.run_sampler(n_chains=100000)
                end_time_mh = time.time()  # End timing

                np.save(f'./models/NN_hl{hl}_w{w}_var{vr}_{nobs}_Samples.npy', nsamples)
                del sampler, nsamples, dt_progres

                elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
                timing_results['CHR_RWMH']["NN"].append(elapsed_time_mh)  # Store timing


                #####  NN SAmpling using pyro with Uniform prior
                # kernel = mcmc.RandomWalkKernel(model_nn, target_accept_prob=0.234)

                # mcmc_sampler = MCMC(kernel, num_samples=1000000, warmup_steps=5000)

                # start_time_mh = time.time()  # Start timing
                # mcmc_sampler.run(model,torch.tensor(obs_sol).float().to(device),
                #                  torch.tensor(obs_points).float().to(device),
                #                  torch.sqrt(torch.tensor(vr).float().to(device)))
                # end_time_mh = time.time()  # End timing

                # elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
                # timing_results['Pyro_RWMH']["NN"].append(elapsed_time_mh)  # Store timing

                # # Get the results (posterior samples)
                # samples = mcmc_sampler.get_samples()
                # samples = samples["theta"].numpy()

                # np.save(f'./models/NN_hl{hl}_w{w}_var{vr}_{nobs}_Samples_unipyro.npy', samples)
                # del kernel, mcmc_sampler, samples

                ######## Deep GaLA
                data_int,left_bc,right_bc = generate_data(nobs)
                data_int,left_bc,right_bc  = data_int.to(device),left_bc.to(device),right_bc.to(device)

                pde = {"PDE":["de","bc_l","bc_r"], 
                    "data_set":{"de" : Variable(data_int,requires_grad=True),
                    "bc_l":left_bc,
                    "bc_r" :right_bc}}
                
                try:
                    llp2 = llaplace2(model)

                    llp2.fit(pde=pde, hessian_structure = "full")

                    llp2.optimize_marginal_likelihood()

                    #### SAmpling DeepGaLA2 with own sampler
                    sampler = MetropolisHastingsSampler(obs_points, obs_sol,surrogate=llp2, sig = st, 
                                                        mean=False,device=device)
                    
                    start_time_mh = time.time()  # Start timing
                    nsamples, dt_progres = sampler.run_sampler(n_chains=100000)
                    end_time_mh = time.time()  # End timing

                    elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
                    timing_results['CHR_RWMH']["dGaLA2"].append(elapsed_time_mh)  # Store timing

                    np.save(f'./models/dGaLA2_hl{hl}_w{w}_var{vr}_{nobs}_Samples.npy', nsamples)
                    del sampler, nsamples, dt_progres

                    #### SAmpling DeepGaLA2 with pyro
                    # kernel = mcmc.RandomWalkKernel(model_dgala, target_accept_prob=0.234)

                    # mcmc_sampler = MCMC(kernel, num_samples=1000000, warmup_steps=5000)

                    # start_time_mh = time.time()  # Start timing
                    # mcmc_sampler.run(llp2,torch.tensor(obs_sol).float().to(device),
                    #              torch.tensor(obs_points).float().to(device),
                    #              torch.sqrt(torch.tensor(vr).float().to(device)))
                    # end_time_mh = time.time()  # End timing
                    # elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
                    # timing_results['Pyro_RWMH']["dGaLA2"].append(elapsed_time_mh)  # Store timing

                    # # Get the results (posterior samples)
                    # samples = mcmc_sampler.get_samples()
                    # samples = samples["theta"].numpy()

                    # np.save(f'./models/dGaLA2_hl{hl}_w{w}_var{vr}_{nobs}_Samples_pyro_uniform.npy', samples)
                    # del kernel, mcmc_sampler, samples

                except:
                    print("deepGALA2 failed for hl:{hl} and w:{w}")

                ##### Deep GaLA
                llp = llaplace(model)
                llp.fit(pde=pde, hessian_structure = "full")

                log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
                hyper_optimizer = torch.optim.Adam([log_prior.to(device), log_sigma.to(device)], lr=1e-2)

                error = 1.0  # Initialize error to start the loop
                n_iter = 0

                while error > 1e-3 or n_iter == 3000:
                    # Clone the values at the start of the iteration to use later for error calculation
                    prev_log_prior, prev_log_sigma = log_prior.clone(), log_sigma.clone()

                    hyper_optimizer.zero_grad()

                    # Calculate negative marginal likelihood
                    neg_marglik = -llp.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                    neg_marglik.backward(retain_graph=True)

                    # Perform optimization step
                    hyper_optimizer.step()

                    # Calculate error directly in PyTorch
                    error = 0.5 * (torch.abs(log_prior - prev_log_prior) + torch.abs(log_sigma - prev_log_sigma)).item()
                    n_iter +=1

                #### SAmpling DeepGaLA with CHR
                sampler = MetropolisHastingsSampler(obs_points, obs_sol,surrogate=llp, sig = st, 
                                                    mean=False,device=device)
                
                start_time_mh = time.time()  # Start timing
                nsamples, dt_progres = sampler.run_sampler(n_chains=100000)
                end_time_mh = time.time()  # End timing
                elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
                timing_results['Pyro_RWMH']["dGaLA"].append(elapsed_time_mh)  # Store timing


                np.save(f'./models/dGaLA_hl{hl}_w{w}_var{vr}_{nobs}_Samples.npy', nsamples)

                del sampler, nsamples, dt_progres

                # #### SAmpling DeepGaLA with pyro
                # kernel = mcmc.RandomWalkKernel(model_dgala, target_accept_prob=0.234)

                # mcmc_sampler = MCMC(kernel, num_samples=1000000, warmup_steps=5000)

                # start_time_mh = time.time()  # Start timing
                # mcmc_sampler.run(llp,torch.tensor(obs_sol).float().to(device),
                #                 torch.tensor(obs_points).float().to(device),
                #                 torch.sqrt(torch.tensor(vr).float().to(device)))
                # end_time_mh = time.time()  # End timing
                # elapsed_time_mh = end_time_mh - start_time_mh  # Calculate elapsed time
                # timing_results['Pyro_RWMH']["dGaLA"].append(elapsed_time_mh)  # Store timing

                # # Get the results (posterior samples)
                # samples = mcmc_sampler.get_samples()
                # samples = samples["theta"].numpy()

                # np.save(f'./models/dGaLA_hl{hl}_w{w}_var{vr}_{nobs}_Samples_pyro_uniform.npy', samples)
                # del kernel, mcmc_sampler, samples

            np.save(f'./models/time_results_NN__hl{hl}_w{w}.npy', timing_results)