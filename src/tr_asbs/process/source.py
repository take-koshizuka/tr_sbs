# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import math
from typing import Dict
from torch.distributions import constraints

class Prior(nn.Module, ABC):
    """
    Common interface for prior distributions.
    """
    
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cov(self) -> float:
        """
        Characteristic covariance matrix of the prior.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log-density log p(x).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Draw samples.
        Should return a tensor whose leading dimension is n_samples.
        """
        raise NotImplementedError
    
    def replicate_sample(self, n_samples: int, duplicate_size: int) -> torch.Tensor:
        """
        Sample and replicate for path sampling.
        
        Args:
            n_samples: Number of independent samples (B).
            dupicate_size: Number of paths per sample (P).
        Returns:
            Tensor of shape (B * P, dim)
        """
        x = self.sample(n_samples)          # shape: (B, *S)
        B = x.shape[0]

        # (B, 1, *S)
        x = x.unsqueeze(1)

        # (B, P, *S)
        x = x.repeat(1, duplicate_size, *([1] * (x.dim() - 2)))

        # (B * P, *S)
        x = x.reshape(B * duplicate_size, *x.shape[2:])

        return x

    def score(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Default implementation of the score function:
        ∇_x log p(x), same shape as samples.

        Subclasses can override this if an analytic form is available.
        """
        grad_fxn = torch.func.grad(self.log_prob)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples)


class GaussianPrior(Prior):
    def __init__(self, dim: int, scale: float, device: str = "cpu"):
        self._dim = int(dim)
        self._scale = float(scale)
        self.device = device
        self._mean = torch.zeros(self._dim, device=self.device)
        self._cov = (self._scale ** 2) * torch.eye(self._dim, device=self.device)
        self.dist = torch.distributions.MultivariateNormal(
            self._mean,
            self._cov
        )
        
    @property
    def dim(self) -> int:
        return self._dim

    # --- properties ---
    @property
    def cov(self) -> float:
        return self._cov

    @property
    def mean(self):
        return self._mean

    # --- API ---
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(x)

    def sample(self, size: tuple) -> torch.Tensor:
        samples = self.dist.sample(size)
        return samples.to(self.device)

    def score(self, samples: torch.Tensor) -> torch.Tensor:
        # Analytic score for isotropic Gaussian
        return -samples / (self._scale ** 2)
    

class MeanFreePrior(torch.distributions.Distribution, Prior):
    """
    Implements a Gaussian distribution on the mean-free subspace.
    
    Energy function:
      E(x) = 1/(2σ^2) * Σ_i ||x_i - x_bar||^2
           = 1/(4σ^2 n) * Σ_{i,j} ||x_i - x_j||^2
    
    Args:
        n_particles (int): Number of particles (N).
        spatial_dim (int): Spatial dimension (D).
        scale (float): Standard deviation (sigma) on the subspace.
        device (str): Device.
    
    """
    arg_constraints = {}
    def __init__(self, n_particles: int, spatial_dim: int, scale: float, device: str = "cpu"):
        # event_shape is flattened (N*D)
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([n_particles * spatial_dim]))
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self._dim = n_particles * spatial_dim
        self.device = device
        self._sigma = float(scale)
        self._sigma2 = self._sigma ** 2

        self.k = (n_particles - 1) * spatial_dim # Effective degrees of freedom

        # Derive alpha from scale (sigma) for energy calculation:
        # sigma^2 = 1 / (2 * alpha * n)  =>  alpha = 1 / (2 * n * sigma^2)
        self._alpha = 1.0 / (2.0 * self.n_particles * self._sigma2)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def scale(self) -> float:
        """Returns the standard deviation (sigma) of this distribution."""
        return self._sigma
    
    @property
    def cov(self) -> torch.Tensor:
        # 粒子空間での射影行列 P = I - (1/N)J
        I = torch.eye(self.n_particles, device=self.device)
        J = torch.ones(self.n_particles, self.n_particles, device=self.device)
        P_particle = I - (J / self.n_particles)
        
        # 空間次元への拡張 (Kronecker product)
        # Cov = sigma^2 * (P (x) I_d)
        cov_matrix = self._sigma2 * torch.kron(P_particle, torch.eye(self.spatial_dim, device=self.device))
        
        return cov_matrix

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of a given particle configuration.
        """
        # Reshape x to (..., N, D)
        B = x.shape[0]
        x_reshaped = x.reshape(-1, self.n_particles, self.spatial_dim)

        # 1. Calculate Energy E(x)
        xc = x_reshaped - x_reshaped.mean(dim=-2, keepdims=True)
        
        # Sum of squared norms over spatial and particle dimensions
        quad = torch.sum(xc**2, dim=(-1, -2))              
        
        # E(x) = _alpha * n_particles * quad
        energy = self._alpha * self.n_particles * quad

        # 2. Calculate Log Normalization Constant (log Z)
        # log Z = (k/2) * log(2πσ^2)
        # Note: Previous formula derived from alpha is equivalent to this simpler form using k and sigma
        log_normalization_constant = 0.5 * self.k * math.log(2.0 * math.pi * self._sigma2)
        
        # Log probability: log p(x) = -E(x) - log Z
        log_px = -energy - log_normalization_constant
        
        return log_px.squeeze()

    def sample(self, size: tuple) -> torch.Tensor:
        """
        Samples from the mean-free Gaussian distribution.
        """
        # 1. Sample from unconstrained Gaussian distribution N(0, σ^2 I)
        # torch.manual_seed(0)
        sample_size = size + (self.n_particles, self.spatial_dim)
        samples = torch.randn(
            *sample_size, device=self.device
        )
        samples = samples * self._sigma
        samples = samples - samples.mean(dim=-2, keepdims=True)
        return samples.reshape(*sample_size)
    
class DeltaPrior(torch.distributions.Distribution, Prior):
    """
    <Delta prior with smoothing>
    A Delta distribution that concentrates mass on the configuration where
    “all particles collapse to the same point” is implemented as follows:
    - mean-free component:  x_i - x̄
    - mean component     :  x̄
    Each component is smoothed with a tiny-variance Gaussian.

    Log probability:
      log p(x) = - [ (1/(2 τ_c^2)) * Σ_i ||x_i - x̄||^2  +  (n/(2 τ_m^2)) * ||x̄||^2 ]
                 - (ND/2) log(2π) - (k/2) log(τ_c^2) - (D/2) log(τ_m^2)

    τ_c, τ_m should be very small (default 1e-6) so the behavior approximates
    a Delta while avoiding -inf.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(
        self,
        n_particles: int,
        spatial_dim: int,
        device: str = "cpu",
        tau_center: float = 1e-6,   # std of smoothing for mean-free part
        tau_mean: float = 1e-6,     # std of smoothing for mean part
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.device = device

        # degrees of freedom (mean-free dimensionality)
        self.k = (n_particles - 1) * spatial_dim

        # smoothing scales
        self.tau_center = float(tau_center)
        self.tau_mean = float(tau_mean)

        # precompute normalization constant
        ND = self.dim
        D = self.spatial_dim
        tc2 = self.tau_center ** 2
        tm2 = self.tau_mean ** 2
        self._logZ = 0.5 * ND * math.log(2.0 * math.pi) \
                   + 0.5 * self.k * math.log(tc2) \
                   + 0.5 * D * math.log(tm2)

    @property
    def scale(self) -> float:
        # Effectively a delta → 0 for compatibility
        return 0.0

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        # (..., N, D) → subtract mean to make it mean-free
        return x - x.mean(dim=-2, keepdims=True)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log p(x) for a sharply peaked Gaussian (τ → 0) smoothing of a Delta.
        - mean-free squared norm suppressed by τ_center
        - mean vector squared norm suppressed by τ_mean
        """
        x = x.reshape(-1, self.n_particles, self.spatial_dim)

        # mean-free component
        xc = self._center(x)
        quad_center = torch.sum(xc**2, dim=(-1, -2))  # Σ_i ||x_i - x̄||^2

        # mean component (D-dim): y_mean = √n * x̄ follows Gaussian with τ_mean
        mean = x.mean(dim=-2)                         # (..., D)
        quad_mean = (self.n_particles * torch.sum(mean**2, dim=-1))  # n * ||x̄||^2

        # energy
        energy = (0.5 / (self.tau_center**2)) * quad_center + (0.5 / (self.tau_mean**2)) * quad_mean

        # log prob
        log_px = -energy - self._logZ
        return log_px

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        True Delta: always return zeros.
        Shape: (n_samples, N, D)
        """
        return torch.zeros(n_samples, self.n_particles, self.spatial_dim, device=self.device)
