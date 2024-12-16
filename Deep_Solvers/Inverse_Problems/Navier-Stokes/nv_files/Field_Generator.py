import numpy as np
import torch


def compute_seq_pairs(N_KL, include_00=False):
    trunc_Nx = int(np.floor(np.sqrt(2 * N_KL)) + 1)
    pairs = []
    # Generate (i, j) pairs and their squared norms
    for i in range(trunc_Nx):
        for j in range(trunc_Nx):
            if i == 0 and j == 0 and not include_00:
                continue
            pairs.append([i, j, i**2 + j**2])

    # Sort pairs by the squared norm (third column) and select the first N_KL pairs
    pairs = sorted(pairs, key=lambda x: x[2])[:N_KL]
    
    # Return only the (i, j) pairs, discarding the norm
    return np.array(pairs)[:, :2]



def generate_omega0(X,Y, seq_pairs, d, tau,seed = None):
    rng = np.random.default_rng(seed)
    # Ensure theta has an even length
    assert len(seq_pairs) % 2 == 0
    N_KL = len(seq_pairs) // 2

    # Random coefficients a_k and b_k
    abk = rng.normal(0, 1, (N_KL, 2))
    
    # Initialize omega0 to zero
    omega0 = np.zeros_like(X)

    # Vectorized computation of omega0
    for i, (kx, ky) in enumerate(seq_pairs[:N_KL]):
        # Ensure conditions on kx and ky
        assert kx + ky > 0 or (kx + ky == 0 and kx > 0)
        
        ak, bk = abk[i]
        k_squared = kx**2 + ky**2
        normalization = 1 / (np.sqrt(2) * np.pi * (tau**2 + k_squared)**(d / 2))

        # Update omega0 with contributions from this (kx, ky) pair
        omega0 += normalization * (ak * np.cos(2*np.pi*(kx * X + ky * Y)) + bk * np.sin(2*np.pi*(kx * X + ky * Y)))

    return 7**(3/2)*omega0


def omega0_samples(X, Y, theta, d=5.0, tau=7.0):
    nKL = theta.shape[0]
    # Compute seq_pairs based on nKL
    seq_pairs = compute_seq_pairs(nKL)
    
    # Initialize omega0 with an extra dimension for multiple samples
    omega0 = np.zeros((X.shape[0], X.shape[1], theta.shape[-1]))
    
    # Outer loop to handle each sample dimension in abk
    for z in range(theta.shape[-1]):
        # Vectorized computation of omega0 for each sample
        for i, (kx, ky) in enumerate(seq_pairs):
            # Ensure conditions on kx and ky
            assert kx + ky > 0 or (kx + ky == 0 and kx > 0)
            ak, bk = theta[i, :, z]
            normalization = 1 / (np.sqrt(2) * np.pi * (tau**2 + kx**2 + ky**2 )**(d / 2))
            # Update omega0 for the z-th sample with contributions from this (kx, ky) pair
            omega0[:, :, z] += normalization * (ak * np.cos(kx * X + ky * Y) + bk * np.sin(kx * X + ky * Y))

    # Apply the scaling factor to the final result
    return 7**(3/2) * omega0

def omega0_samples_torch(X, Y, theta, d=5.0, tau=7.0):
    nKL = theta.shape[0]
    
    # Compute seq_pairs in numpy and then convert it to PyTorch
    seq_pairs = compute_seq_pairs(nKL)
    seq_pairs = torch.tensor(seq_pairs, device=X.device, dtype=X.dtype)  # Convert to PyTorch tensor
    
    # Initialize omega0 with an extra dimension for multiple samples, as a torch tensor
    omega0 = torch.zeros((X.shape[0], Y.shape[-1], theta.shape[-1]), device=X.device, dtype=X.dtype)
    
    # Outer loop to handle each sample dimension in theta
    for z in range(theta.shape[-1]):
        # Vectorized computation of omega0 for each sample
        for i, (kx, ky) in enumerate(seq_pairs):
            # Ensure conditions on kx and ky
            assert kx + ky > 0 or (kx + ky == 0 and kx > 0)
            
            # Get ak and bk for the current sample z and index i
            ak, bk = theta[i, :, z]
            
            # Compute normalization factor using torch functions
            k_squared = kx**2 + ky**2
            normalization = 1 / (torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype)) * torch.pi * (tau**2 + k_squared)**(d / 2))
            
            # Update omega0 for the z-th sample with contributions from this (kx, ky) pair
            omega0[:, :, z] += normalization * (ak * torch.cos(kx * X + ky * Y) + bk * torch.sin(kx * X + ky * Y))

    # Apply the scaling factor to the final result
    return 7**(3/2) * omega0