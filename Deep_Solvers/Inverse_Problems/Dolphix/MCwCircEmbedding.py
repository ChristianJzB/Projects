#!/work/sc073/sc073/s2079009/miniconda3/bin/python

from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import math
import time
import logging
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/s2079009/MAC-MIGS/PhD/PhD_code/')
from utils import PDE_solver
from utils import cov_functions
from utils import circ_embedding
from utils import periodisation_smooth

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def MC_simulation(N, x, y, a, alpha, var, cov_fun, cov_fun_per, mesh_size=8, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
    '''
    Performs a standard Monte Carlo simulation for computing E[p(x,y)].

    :param N: the number of samples to use per simulation.
    :param m_KL: number of KL modes to include in KL-expansion.
    :param x: the x point at which to compute E[p(x,y)].
    :param y: the y point at which to compute E[p(x,y)].
    :param a: the RHS constant of the PDE.
    :param alpha: constant for computing E[Q_M - Q].
    :param var: the sample variance used in computing RMSE.
    :param mesh_size (default 8): the step size to be used for computing FEM approximation.
    :param pol_degree (default 1): degree of polynomial to be used to computed FEM approximation.
    ::param rho (default 0.3): correlation length of the covariance function.
    :param sigma (default 1): variance of the covariance function.
    :param nu (default 0.5): smoothness parameter of covariance function.

    :return: 
        rmse - root mean squared error between approximation and exact value.
        total_time - time taken to compute MC estimate.
    '''

    # values for computing approximation on 2 different grids
    p_hat = 0
    p_hat_2 = 0

    # time the duration of simulation
    total_time = 0

    egnv, m_per = \
        circ_embedding.find_cov_eigenvalues(mesh_size, mesh_size, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)
    

    # FEM setup on two different grids
    V, f, bc = \
        PDE_solver.setup_PDE(m=mesh_size, pol_degree=pol_degree, a=a)

    V_2, f_2, bc_2 = \
        PDE_solver.setup_PDE(m=mesh_size//2, pol_degree=pol_degree, a=a)

    # Monte Carlo simulation for computing E[p(x,y)] 
    for i in tqdm(range(N)):
        t0 = time.time()

        xi = rng.standard_normal(size=4*m_per*m_per)
        w = xi * (np.sqrt(np.real(egnv)))
        w = interfaces.scipy_fft.fft2(w.reshape((2*m_per, 2*m_per)), norm='ortho')
        w = np.real(w) + np.imag(w)
        Z1 = w[:mesh_size+1, :mesh_size+1].reshape((mesh_size+1)*(mesh_size+1))

        # Compute solution on current grid with current k
        k = PDE_solver.k_RF(Z1, mesh_size)
        u = Function(V)
        v = TestFunction(V)
        F = k * dot(grad(u), grad(v)) * dx - f * v * dx
        solve(F == 0, u, bc)
        
        # quantity of interest
        p_hat += u(x, y)
        # p_hat += norm(u, 'L2')

        t1 = time.time()
        total_time += t1-t0

        # Compute solution for finer mesh (4*M)
        Z2 = w[:mesh_size+1:2, :mesh_size+1:2].reshape(((mesh_size//2)+1)*((mesh_size//2)+1))

        # Compute solution for finer mesh (4*M)
        k_2 = PDE_solver.k_RF(Z2, mesh_size//2)
        u_2 = Function(V_2)
        v_2 = TestFunction(V_2)
        F_2 = k_2 * dot(grad(u_2), grad(v_2)) * dx - f_2 * v_2 * dx
        solve(F_2 == 0, u_2, bc_2)

        # quantity of interest
        p_hat_2 += u_2(x, y)
        # p_hat_2 += norm(u_2, 'L2')

    # Monte Carlo estimates of E[p(x, y)]
    exp_est = p_hat/N
    exp_est_2 = p_hat_2/N

    # extrapolation value of discretisation error E[Q_M-Q]
    exp_true = (exp_est_2-exp_est) / (1 - 4 ** (-alpha))
    # compute discretisation error
    disc_error = exp_true ** 2  
    
    # Root Mean Squared Error of E[p(x,y)]
    rmse = np.sqrt(1 / N * var + disc_error)

    print(f'Sample error is {1/N*var}.')
    print(f'Discretisation error is is {disc_error}.')
    print(f'Exp estimate is {exp_est}.')
    print(f'Finer exp estimate is {exp_est_2}.')
    print(f'RMSE is {rmse}.')
    print(f'Time is {total_time}.')

    return rmse, total_time

def main():
    # the value at which to compute E[p(x,y)]
    x_val = 7 / 15
    y_val = 7 / 15
    # choose RHS constant for the ODE
    a_val = 1
    # polynomial degree for computing FEM approximations
    pol_degree_val = 1
    # variance of random field
    sigma = 1
    # correlation length of random field
    rho = 0.03
    # smoothness parameters of random field
    nu = 1.5
    # norm to be used in covariance function of random field
    p_val = 2

    var_val = 0.0018769639289535666
    alpha_val = 0.3570868564486739
    C_alpha = -3.053440188533286

    # list of desired accuracies
    epsilon = 0.05

    mesh_sizes = [2**n for n in range(4, 9)]
        
    print(f'Starting simulation for epsilon = {epsilon}.')
    # number of samples needed to reach epsilon accuracy
    N_val = 3*math.ceil(2*epsilon ** (-2) * var_val)
    # mesh size needed to reach epsilon accuracy
    mesh_size_val = \
        math.ceil((math.sqrt(2)*np.exp(C_alpha)/epsilon)**(1/(2*alpha_val)))
    mesh_size_val = min(mesh_sizes, key=lambda x:abs(x-mesh_size_val))
    print(f'N = {N_val} and mesh size = {mesh_size_val}.')

    # compute time needed for computing E[p(x, y)] with reference value
    t0_val = time.time()

    # MC simulation for current epsilon
    rmse_val, time_val = \
        MC_simulation(N_val, x_val, y_val, a_val, alpha_val, var_val, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun,mesh_size_val, pol_degree_val, sigma, rho, nu, p_val)

    t1_val = time.time()
    print(f'Total time is {t1_val-t0_val}\n')

    print(f'rmse_val = {rmse_val}')
    print(f'time_val = {time_val}')

if __name__ == "__main__":
    main()
