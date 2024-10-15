import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from FEM_Solver import  RootFinder,FEMSolver
from MetropolisHastings import MetropolisHastingsSampler
from GaLa import llaplace

from utilities import *
from NN import DNN

import numpy as np

#np.random.seed(1234)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def tang_eq(x):
    return np.tan(x) - 8*(x/(x**2 - 16))

intervals = [(1,2),(2,3),(3.8,4.1),(4.1,4.6),(4.7,4.8),(7.2,7.3),(7.3,8),(10,10.2),(10.2,11.5),(13,13.2),(14,14.5)]
n = 2

root_finder = RootFinder(n, tang_eq, intervals)
roots = root_finder.find_roots()
root_finder.plot_equation()

print(f"The first {n} roots are: {roots}")

theta_t = np.array([0.09762701, 0.43037873, 0.20552675, 0.08976637, -0.1526904, 0.29178823, -0.12482558, 0.783546, 0.92732552, -0.23311696])

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
##### Samples from FEM

obs = 10
theta_th=np.array([0.098, 0.430])
mean, var = 0,[1e-4,5e-4,1e-3,5e-3]
N = [150,500,1000,1500]


#### Samples From Surrogate
layers = [3] + 1*[20] + [1]

for vr in var:
    st = np.sqrt(vr)
    obs_points, obs_sol = generate_noisy_obs(obs, theta_t=theta_th, mean=mean, std=st,vert=200)

    sampler = MetropolisHastingsSampler(obs_points, obs_sol, 
                                       sig = st,numerical = True, roots=roots, vert=50,device=device)

    samp_num,dt_tracker_num = sampler.run_sampler(n_chains=100000)

    np.save(f'./models/FEM_var{vr}_adam_Samples.npy', samp_num)

    del sampler, samp_num, dt_tracker_num

    for nobs in N:
        model = DNN(layers).to(device)
        model.load_state_dict(torch.load(f"./models/1dElliptic_adam_PDE2_w20_N{nobs}_batch150.pt"))
        model.eval()

        sampler = MetropolisHastingsSampler(obs_points, obs_sol,surrogate=model, sig = st)
        nsamples, dt_progres = sampler.run_sampler(n_chains=100000)

        np.save(f'./models/NN_var{vr}_{nobs}_Samples.npy', nsamples)
        np.save(f'./models/NN_var{vr}_{nobs}_Step.npy', dt_progres)
        del sampler, nsamples, dt_progres

        ##### Deep GaLA
        data_int,left_bc,right_bc = generate_data(nobs)
        data_int,left_bc,right_bc  = data_int.to(device),left_bc.to(device),right_bc.to(device)

        pde = {"PDE":["de","bc_l","bc_r"], 
            "data_set":{"de" : Variable(data_int,requires_grad=True),
            "bc_l":left_bc,
            "bc_r" :right_bc}}
        
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

        sampler = MetropolisHastingsSampler(obs_points, obs_sol,surrogate=llp, sig = st, mean=False)
        nsamples, dt_progres = sampler.run_sampler(n_chains=100000)

        np.save(f'./models/dGaLA_var{vr}_{nobs}_Samples.npy', nsamples)
        np.save(f'./models/dGaLA_var{vr}_{nobs}_Step.npy', dt_progres)

        del sampler, nsamples, dt_progres



# ##### Deep GaLA
# deepGala = dict()

# for nobs in N:
#     data_int,left_bc,right_bc = generate_data(nobs)
#     data_int,left_bc,right_bc  = data_int.to(device),left_bc.to(device),right_bc.to(device)

#     model = DNN(layers).to(device)
#     path = f"./models/1dElliptic_adam_PDE2_w20_N{nobs}_batch150.pt"
#     model.load_state_dict(torch.load(path))
#     model.eval()

#     pde = {"PDE":["de","bc_l","bc_r"], 
#         "data_set":{"de" : Variable(data_int,requires_grad=True),
#         "bc_l":left_bc,
#         "bc_r" :right_bc}}

#     llp = llaplace(model)
#     llp.fit(pde=pde, hessian_structure = "full")

#     log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
#     hyper_optimizer = torch.optim.Adam([log_prior.to(device), log_sigma.to(device)], lr=1e-2)

#     error = 1.0  # Initialize error to start the loop
#     n_iter = 0
#     while error > 1e-3 or n_iter == 3000:
#         # Clone the values at the start of the iteration to use later for error calculation
#         prev_log_prior, prev_log_sigma = log_prior.clone(), log_sigma.clone()

#         hyper_optimizer.zero_grad()

#         # Calculate negative marginal likelihood
#         neg_marglik = -llp.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
#         neg_marglik.backward(retain_graph=True)

#         # Perform optimization step
#         hyper_optimizer.step()

#         # Calculate error directly in PyTorch
#         error = 0.5 * (torch.abs(log_prior - prev_log_prior) + torch.abs(log_sigma - prev_log_sigma)).item()
#         n_iter +=1
#     deepGala[str(nobs)] = llp



# # ###### Samples Deep GaLA
# for vr in var:
#     st = np.sqrt(vr)

#     obs_points, obs_sol = generate_noisy_obs(obs, theta_t=theta_th, mean=mean, std=st,vert=200)

#     for nobs in N:
#         dgl = deepGala[str(nobs)]

#         sampler = MetropolisHastingsSampler(obs_points, obs_sol,surrogate=dgl, sig = st, mean=False)
#         nsamples, dt_progres = sampler.run_sampler(n_chains=100000)

#         np.save(f'./models/dGaLA_var{vr}_{nobs}_Samples.npy', nsamples)
#         np.save(f'./models/dGaLA_var{vr}_{nobs}_Step.npy', dt_progres)