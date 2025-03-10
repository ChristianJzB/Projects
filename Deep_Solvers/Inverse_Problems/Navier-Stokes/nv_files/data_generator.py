import torch
import torch.fft as fft
from torch.utils.data import Dataset

# Solve Poisson equation using FFT (batched version)
def solve_poisson_fft(omega, dx, dy):
    """
    Solves ∇²ψ = -omega for a batch of vorticity fields.
    omega: Tensor of shape (X, Y, n_samples)
    """
    # Get grid size
    nx, ny, n_samples = omega.shape

    # Compute wavenumbers
    kx = fft.fftfreq(nx, dx) * 2 * torch.pi  # Wavenumbers in x
    ky = fft.fftfreq(ny, dy) * 2 * torch.pi  # Wavenumbers in y
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # Avoid division by zero at (kx, ky) = (0, 0)

    # FFT of vorticity (batched along the last dimension)
    omega_hat = fft.fft2(omega, dim=(0, 1))

    # Solve for streamfunction in Fourier space
    psi_hat = omega_hat / k2.unsqueeze(-1)  # Broadcast k2 over n_samples
    psi_hat[0, 0, :] = 0.0  # Enforce zero mean for each sample

    # Inverse FFT to get streamfunction in real space
    psi = fft.ifft2(psi_hat, dim=(0, 1)).real
    return psi

# Compute velocity field (batched version)
def compute_velocity(psi, dx, dy):
    """
    Computes velocity (u, v) for a batch of streamfunctions.
    psi: Tensor of shape (X, Y, n_samples)
    """
    # FFT of psi (batched along the last dimension)
    psi_hat = fft.fft2(psi, dim=(0, 1))

    # Compute wavenumbers
    nx, ny, n_samples = psi.shape
    kx = fft.fftfreq(nx, dx) * 2 * torch.pi
    ky = fft.fftfreq(ny, dy) * 2 * torch.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")

    # Spectral derivatives
    u_hat = 1j * ky.unsqueeze(-1) * psi_hat  # u = ∂ψ/∂y
    v_hat = -1j * kx.unsqueeze(-1) * psi_hat  # v = -∂ψ/∂x

    # Inverse FFT to get real-space velocities
    u = fft.ifft2(u_hat, dim=(0, 1)).real
    v = fft.ifft2(v_hat, dim=(0, 1)).real
    return u, v




class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_seed=1234):
        """
        Base class for samplers.
        
        :param batch_size: The size of the batch to be sampled.
        :param rng_seed: Random seed for reproducibility.
        """
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self.num_devices = torch.cuda.device_count()  # Gets the number of GPUs

    def __getitem__(self, index):
        """
        Generate one batch of data.

        :param index: Index for batch sampling, unused here but needed for Dataset API.
        :return: Batch of data
        """
        # Increment the seed (or set to something dynamic like current time)
        self.rng_seed += index
        batch = self.data_generation()
        return batch

    def data_generation(self):
        """
        Abstract method to generate data, to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, device,rng_seed=1234):
        super().__init__(batch_size, rng_seed)
        self.device = device
        self.dom = dom.to(self.device)  # (dim, 2), where each row is [min, max] for a dimension
        self.dim = dom.shape[0]

    def data_generation(self):
        """
        Generates batch_size random samples uniformly within the domain, respecting the RNG seed.
        
        :return: Tensor of shape (batch_size, dim)
        """
        # Generate random samples uniformly within the given domain (min, max)
        min_vals = self.dom[:, 0]
        max_vals = self.dom[:, 1]
        
        # Initialize random number generator with seed for reproducibility
        torch.manual_seed(self.rng_seed)  # Reset the seed for each batch
        
        # Generate random samples in the range [min_vals, max_vals] for each dimension
        # Using rand to create uniform samples in the [0, 1) range
        rand_vals = torch.rand(self.batch_size, self.dim).to(self.device)
        
        # Scale the values to be in the [min, max] range for each dimension
        batch = min_vals + rand_vals * (max_vals - min_vals)

        return batch