import numpy as np
import cupy as cp
from cupy.fft import rfft2, irfft2, fftfreq, rfftfreq


class VorticitySolver2D:
    def __init__(self, N, T, nu, dt, num_sol = 10,L = 1, method = 'CN', force=None):
        """
        Initialize the Vorticity Solver with necessary parameters.
        
        Parameters:
            N (int): Grid size (number of grid points along one dimension).
            L (float): Domain size (assumed square domain).
            T (float): Final time for simulation.
            nu (float): Viscosity of the fluid.
            dt (float): Time step.
        """
        self.N = N
        self.L = L
        self.T = T
        self.nu = nu
        self.dt = dt
        self.num_sol = num_sol
        self.method = method
        self.force = force

        # Add the forcing term (if provided)
        if self.force is not None:
            # Create the grid (X, Y)
            X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
            self.f_hat = np.fft.rfft2(self.force(X, Y)) # Apply the force
        
        # Grid setup
        self.kx = 2 * torch.pi*np.fft.fftfreq(N, d=self.L / N)
        self.ky = 2 * torch.pi*np.fft.rfftfreq(N, d=self.L / N)
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.k_squared = self.kx**2 + self.ky**2

        # Avoid division by zero for the Laplace operator
        self.laplace_operator = self.k_squared.copy()
        self.laplace_operator[0, 0] = 1  # Regularize the zero mode
        
        # Dealiasing filter
        self.dealias_filter = self.brick_wall_filter_2d((N, N))
        
        # Initialize other variables
        self.time_array = np.linspace(dt, T, int(T / dt))
        self.points = np.linspace(0, len(self.time_array) - 1, self.num_sol, dtype=int)

        # RK4 parameters
        self.alphas = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
        self.betas = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
        self.gammas = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

    def brick_wall_filter_2d(self, grid_shape):
        """
        Implements the 2/3 rule dealiasing filter for a 2D real-to-complex Fourier transform grid.
        
        Parameters:
        grid_shape (tuple): The shape of the real-space grid (n, m), where `n` is the number of rows
                             and `m` is the number of columns.
        
        Returns:
        ndarray: A 2D binary array (filter) for dealiasing in Fourier space.
        """
        n, m = grid_shape
        filter_ = np.zeros((n, m // 2 + 1))
        kx_max = int(2 / 3 * n // 2)  # Cutoff for rows (kx)
        ky_max = int(2 / 3 * (m // 2 + 1))  # Cutoff for columns (ky)

        filter_[:kx_max, :ky_max] = 1
        filter_[-kx_max:, :ky_max] = 1

        return filter_

    def compute_cfl_time_step(self, u, v):
        """
        Computes the maximum stable time step according to the CFL condition.
        
        Parameters:
        u (ndarray): x-component of velocity (physical space).
        v (ndarray): y-component of velocity (physical space).
        
        Returns:
        float: Maximum allowable time step according to the CFL condition.
        """
        dx = self.L / self.N
        dy = dx  # since LxL domain, dx = dy
        delta = min(dx, dy)

        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))

        dt_cfl = min(dx / u_max, dy / v_max, delta**2 / (2 * self.nu))
        return dt_cfl

    def initialize_vorticity(self, w0):
        """
        Initialize the vorticity field.
        
        Parameters:
        w_ref (ndarray): Reference vorticity field for initialization.
        
        Returns:
        ndarray: Initial vorticity field (w0).
        """
        w_hat = np.fft.rfft2(w0)  # Fourier transform of initial vorticity
        w_hat *= self.dealias_filter  # Apply dealiasing filter
        return w_hat

    def solve_poisson(self, w_hat):
        """
        Solves Poisson equation for stream function in Fourier space.
        
        Parameters:
        w_hat (ndarray): Vorticity field in Fourier space.
        
        Returns:
        ndarray: Stream function in Fourier space.
        """
        return w_hat / self.laplace_operator

    def compute_velocity(self, psi_hat):
        """
        Compute velocity field in Fourier space from stream function.
        
        Parameters:
        psi_hat (ndarray): Stream function in Fourier space.
        
        Returns:
        tuple: u_hat, v_hat (velocity components in Fourier space).
        """
        u_hat = 1j * self.ky * psi_hat  # u = dpsi/dy
        v_hat = -1j * self.kx * psi_hat  # v = -dpsi/dx

        u = np.fft.irfft2(u_hat, s=(self.N, self.N))
        v = np.fft.irfft2(v_hat, s=(self.N, self.N))
        
        return u_hat, v_hat, u, v

    def apply_nonlinear_term(self, u, v, w_hat):
        """
        Compute and return the nonlinear term in Fourier space.
        
        Parameters:
        u (ndarray): x-component of velocity (physical space).
        v (ndarray): y-component of velocity (physical space).
        w_hat (ndarray): Vorticity in Fourier space.
        
        Returns:
        ndarray: Nonlinear term in Fourier space.
        """
        dw_dx_hat = 1j * self.kx * w_hat
        dw_dy_hat = 1j * self.ky * w_hat
        nonlinear_term = u * np.fft.irfft2(dw_dx_hat, s=(self.N, self.N)) + v * np.fft.irfft2(dw_dy_hat, s=(self.N, self.N))
        nonlinear_term = np.fft.rfft2(nonlinear_term) * self.dealias_filter
        # Add the forcing term (if provided)
        if self.force is not None:
            nonlinear_term -= self.f_hat       # Add forcing term in Fourier space
        return nonlinear_term

    def crank_nicholson_step(self, w_hat):
        """
        Perform a time step for solving the vorticity equation.
        
        Parameters:
        w_hat (ndarray): Current vorticity in Fourier space.
        
        Returns:
        ndarray: Updated vorticity in Fourier space.
        """
        # Solve Poisson for stream function
        psi_hat = self.solve_poisson(w_hat)
        
        # Compute velocity
        u_hat, v_hat, u, v = self.compute_velocity(psi_hat)

        # Compute nonlinear term and update w_hat
        nonlinear_term_hat = self.apply_nonlinear_term(u, v, w_hat)
        
        # Apply CFL condition for time step
        dt_max = self.compute_cfl_time_step(u, v)
        if self.dt > dt_max:
            print(f"Warning: Time step {self.dt} exceeds the CFL limit {dt_max}. Reducing time step.")
            self.dt = dt_max  # Adjust time step if needed

        # Update vorticity in Fourier space using Crank-Nicolson scheme
        w_hat = ((1 - 0.5 * self.dt * self.nu * self.laplace_operator) * w_hat - self.dt * nonlinear_term_hat) / (1 + 0.5 * self.dt * self.nu * self.laplace_operator)
        return w_hat
    

    def rk4_step(self, w_hat):
        """
        Performs one RK4 step to advance the simulation.
        
        Parameters:
        w_hat (ndarray): Current vorticity in Fourier space.
        u (ndarray): x-component of velocity (physical space).
        v (ndarray): y-component of velocity (physical space).
        
        Returns:
        ndarray: Updated vorticity field in Fourier space.
        """
        h = 0
        for k in range(len(self.betas)):

            psi_hat = self.solve_poisson(w_hat)
        
            # Compute velocity
            u_hat, v_hat, u, v = self.compute_velocity(psi_hat)

            # Compute nonlinear term and update w_hat
            nonlinear_term_hat = self.apply_nonlinear_term(u, v, w_hat)
            

            dt_max = self.compute_cfl_time_step(u, v)
            if self.dt > dt_max:
                print(f"Warning: Time step {self.dt} exceeds the CFL limit {dt_max}. Reducing time step.")
                self.dt = dt_max  # Adjust time step if needed

            h = -nonlinear_term_hat + self.betas[k] * h

            mu = 0.5 * self.dt * (self.alphas[k + 1] - self.alphas[k])
            w_hat = ((1 - mu * self.nu * self.laplace_operator) * w_hat + self.gammas[k] * self.dt * h) / (1 + mu * self.nu * self.laplace_operator)

        return w_hat


    def run_simulation(self, w_ref):
        """
        Run the vorticity solver over the specified time domain.
        
        Parameters:
        w_ref (ndarray): Reference vorticity field for initialization.
        """
        w_list = []
        w_list.append(w_ref)

        w_hat = self.initialize_vorticity(w_ref)

        for time in self.time_array:
             
            if self.method == 'RK4':
                w_hat = self.rk4_step(w_hat)
            elif self.method == 'CN':
                w_hat = self.crank_nicholson_step(w_hat)
        
            
            if time in self.time_array[self.points]:
                w = np.fft.irfft2(w_hat, s=(self.N, self.N))
                w_list.append(w)

        return w_list
    


class NVSolver2D:
    def __init__(self, N, T, nu, dt, num_sol=10, L=1, method='CN', force=None):
        """
        Initialize the Vorticity Solver with necessary parameters.
        
        Parameters:
            N (int): Grid size (number of grid points along one dimension).
            L (float): Domain size (assumed square domain).
            T (float): Final time for simulation.
            nu (float): Viscosity of the fluid.
            dt (float): Time step.
        """
        self.N = N
        self.L = L * 2 * cp.pi
        self.T = T
        self.nu = nu
        self.dt = dt
        self.num_sol = num_sol
        self.method = method
        self.force = force

        # Add the forcing term (if provided)
        if self.force is not None:
            X, Y = cp.meshgrid(cp.linspace(0, self.L, self.N), cp.linspace(0, self.L, self.N))
            self.f_hat = rfft2(self.force(X, Y))  # Apply the force

        # Grid setup
        self.kx = fftfreq(N, d=self.L / N)
        self.ky = rfftfreq(N, d=self.L / N)
        self.kx, self.ky = cp.meshgrid(self.kx, self.ky, indexing="ij")
        self.k_squared = self.kx**2 + self.ky**2

        # Avoid division by zero for the Laplace operator
        self.laplace_operator = self.k_squared.copy()
        self.laplace_operator[0, 0] = 1  # Regularize the zero mode

        # Dealiasing filter
        self.dealias_filter = self.brick_wall_filter_2d((N, N))

        # Initialize other variables
        self.w_list = []
        self.time_array = cp.linspace(dt, T, int(T / dt))
        self.points = cp.linspace(0, len(self.time_array) - 1, self.num_sol, dtype=int)

        # RK4 parameters
        self.alphas = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
        self.betas = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
        self.gammas = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

    def brick_wall_filter_2d(self, grid_shape):
        """
        Implements the 2/3 rule dealiasing filter for a 2D real-to-complex Fourier transform grid.
        """
        n, m = grid_shape
        filter_ = cp.zeros((n, m // 2 + 1), dtype=cp.float32)
        kx_max = int(2 / 3 * n // 2)  # Cutoff for rows (kx)
        ky_max = int(2 / 3 * (m // 2 + 1))  # Cutoff for columns (ky)

        filter_[:kx_max, :ky_max] = 1
        filter_[-kx_max:, :ky_max] = 1

        return filter_

    def compute_cfl_time_step(self, u, v):
        """
        Computes the maximum stable time step according to the CFL condition.
        """
        dx = self.L / self.N
        dy = dx  # since LxL domain, dx = dy
        delta = min(dx, dy)

        u_max = cp.max(cp.abs(u))
        v_max = cp.max(cp.abs(v))

        dt_cfl = min(dx / u_max, dy / v_max, delta**2 / (2 * self.nu))
        return dt_cfl

    def initialize_vorticity(self, w0):
        """
        Initialize the vorticity field.
        """
        w0 = cp.asarray(w0)
        #plan = cp.fft.get_fft_plan(w0)  # Precompute FFT plan

        w_hat = rfft2(w0)  # Fourier transform of initial vorticity
        w_hat *= self.dealias_filter  # Apply dealiasing filter
        self.w_list.append(w0)  # Keep result in CPU memory for later visualization
        return w_hat

    def solve_poisson(self, w_hat):
        """
        Solves Poisson equation for stream function in Fourier space.
        """
        return w_hat / self.laplace_operator

    def compute_velocity(self, psi_hat):
        """
        Compute velocity field in Fourier space from stream function.
        """
        u_hat = 1j * self.ky * psi_hat  # u = dpsi/dy
        v_hat = -1j * self.kx * psi_hat  # v = -dpsi/dx

        u = irfft2(u_hat, s=(self.N, self.N))
        v = irfft2(v_hat, s=(self.N, self.N))

        return u_hat, v_hat, u, v

    def apply_nonlinear_term(self, u, v, w_hat):
        """
        Compute and return the nonlinear term in Fourier space.
        """
        dw_dx_hat = 1j * self.kx * w_hat
        dw_dy_hat = 1j * self.ky * w_hat
        nonlinear_term = u * irfft2(dw_dx_hat, s=(self.N, self.N)) + v * irfft2(dw_dy_hat, s=(self.N, self.N))
        nonlinear_term = rfft2(nonlinear_term) * self.dealias_filter
        if self.force is not None:
            nonlinear_term -= self.f_hat  # Add forcing term in Fourier space
        return nonlinear_term

    def crank_nicholson_step(self, w_hat):
        """
        Perform a time step for solving the vorticity equation.
        """
        psi_hat = self.solve_poisson(w_hat)
        u_hat, v_hat, u, v = self.compute_velocity(psi_hat)
        nonlinear_term_hat = self.apply_nonlinear_term(u, v, w_hat)

        dt_max = self.compute_cfl_time_step(u, v)
        if self.dt > dt_max:
            print(f"Warning: Time step {self.dt} exceeds the CFL limit {dt_max}. Reducing time step.")
            self.dt = dt_max

        w_hat = ((1 - 0.5 * self.dt * self.nu * self.laplace_operator) * w_hat - self.dt * nonlinear_term_hat) / (
            1 + 0.5 * self.dt * self.nu * self.laplace_operator
        )
        return w_hat

    def rk4_step(self, w_hat):
        """
        Performs one RK4 step to advance the simulation.
        """
        h = 0
        for k in range(len(self.betas)):
            psi_hat = self.solve_poisson(w_hat)
            u_hat, v_hat, u, v = self.compute_velocity(psi_hat)
            nonlinear_term_hat = self.apply_nonlinear_term(u, v, w_hat)

            dt_max = self.compute_cfl_time_step(u, v)
            if self.dt > dt_max:
                print(f"Warning: Time step {self.dt} exceeds the CFL limit {dt_max}. Reducing time step.")
                self.dt = dt_max

            h = -nonlinear_term_hat + self.betas[k] * h
            mu = 0.5 * self.dt * (self.alphas[k + 1] - self.alphas[k])
            w_hat = ((1 - mu * self.nu * self.laplace_operator) * w_hat + self.gammas[k] * self.dt * h) / (
                1 + mu * self.nu * self.laplace_operator
            )

        return w_hat

    def run_simulation(self, w_ref):
        """
        Run the vorticity solver over the specified time domain.
        """
        w_hat = self.initialize_vorticity(w_ref)

        for time in self.time_array:
            if self.method == 'RK4':
                w_hat = self.rk4_step(w_hat)
            elif self.method == 'CN':
                w_hat = self.crank_nicholson_step(w_hat)

            if time in self.time_array[self.points]:
                w = irfft2(w_hat, s=(self.N, self.N))
                self.w_list.append(w)  # Store on CPU memory for post-processing

        return [cp.asnumpy(w) for w in self.w_list]


import torch
import torch.fft

class torch_NVSolver2D:
    def __init__(self, N, T, nu, dt, num_sol=10, L=1, method='CN', force=None, device='cpu'):
        self.device = torch.device(device)
        self.N = N
        self.L = L 
        self.T = T
        self.nu = nu
        self.dt = dt
        self.num_sol = num_sol
        self.method = method
        self.force = force

        if self.force is not None:
            X, Y = torch.meshgrid(torch.linspace(0, self.L, self.N, device=self.device), 
                                  torch.linspace(0, self.L, self.N, device=self.device), indexing='ij')
            self.f_hat = torch.fft.rfft2(self.force(X, Y)).to(self.device)

        self.kx = 2 * torch.pi *torch.fft.fftfreq(N, d=self.L / N, device=self.device)
        self.ky = 2 * torch.pi *torch.fft.rfftfreq(N, d=self.L / N, device=self.device)
        self.kx, self.ky = torch.meshgrid(self.kx, self.ky, indexing="ij")
        self.k_squared = self.kx**2 + self.ky**2
        self.laplace_operator = self.k_squared.clone()
        self.laplace_operator[0, 0] = 1  # Avoid division by zero

        self.dealias_filter = self.brick_wall_filter_2d((N, N)).to(self.device)

        # Initialize other variables
        self.time_array = np.linspace(dt, T, int(T / dt))
        self.points = np.linspace(0, len(self.time_array) - 1, self.num_sol, dtype=int)

    def brick_wall_filter_2d(self, grid_shape):
        n, m = grid_shape
        filter_ = torch.zeros((n, m // 2 + 1), dtype=torch.float32, device=self.device)
        kx_max = int(2 / 3 * n // 2)
        ky_max = int(2 / 3 * (m // 2 + 1))

        filter_[:kx_max, :ky_max] = 1
        filter_[-kx_max:, :ky_max] = 1
        return filter_

    def compute_cfl_time_step(self, u, v):
        dx = self.L / self.N
        dy = dx
        delta = min(dx,dy)
        eps = 1e-10  # Small number to prevent division by zero

        u_max = torch.max(torch.abs(u)).item() + eps   # Small number to prevent division by zero
        v_max = torch.max(torch.abs(v)).item()+ eps

        dt_cfl = min(dx / u_max, dy / v_max, delta**2 / (2 * self.nu))
        return dt_cfl

    def initialize_vorticity(self, w0):
        w0 = torch.as_tensor(w0, device=self.device)
        w_hat = torch.fft.rfft2(w0)
        w_hat *= self.dealias_filter
        return w_hat

    def solve_poisson(self, w_hat):
        return w_hat / self.laplace_operator

    def compute_velocity(self, psi_hat):
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat

        u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
        v = torch.fft.irfft2(v_hat, s=(self.N, self.N))
        return u_hat, v_hat, u, v

    def apply_nonlinear_term(self, u, v, w_hat):
        dw_dx_hat = 1j * self.kx * w_hat
        dw_dy_hat = 1j * self.ky * w_hat

        nonlinear_term = u * torch.fft.irfft2(dw_dx_hat, s=(self.N, self.N)) + v * torch.fft.irfft2(dw_dy_hat, s=(self.N, self.N))
        nonlinear_term = torch.fft.rfft2(nonlinear_term) * self.dealias_filter

        if self.force is not None:
            nonlinear_term -= self.f_hat
        return nonlinear_term

    def crank_nicholson_step(self, w_hat):
        psi_hat = self.solve_poisson(w_hat)
        _, _, u, v = self.compute_velocity(psi_hat)
        nonlinear_term_hat = self.apply_nonlinear_term(u, v, w_hat)

        dt_max = self.compute_cfl_time_step(u, v)
        if self.dt > dt_max:
            print(f"Warning: Time step {self.dt} exceeds CFL limit {dt_max}. Reducing time step.")
            self.dt = dt_max

        w_hat = ((1 - 0.5 * self.dt * self.nu * self.laplace_operator) * w_hat - self.dt * nonlinear_term_hat) / (
            1 + 0.5 * self.dt * self.nu * self.laplace_operator)
        return w_hat

    def run_simulation(self, w_ref):
        w_list = []
        w_list.append(torch.as_tensor(w_ref, device=self.device))

        w_hat = self.initialize_vorticity(w_ref)

        for time in self.time_array:
            w_hat = self.crank_nicholson_step(w_hat)
            if time in self.time_array[self.points]:
                w = torch.fft.irfft2(w_hat, s=(self.N, self.N))
                w_list.append(w)

        return w_list
