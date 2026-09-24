# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from abc import ABC, abstractmethod
from functools import lru_cache

import torch
import torch.nn as nn
import math
from typing import Dict, TYPE_CHECKING
from torch.distributions import constraints

from tr_asbs.utils.composition import composition_to_atomic_numbers
from tr_asbs.utils.geometry import is_mean_free_data

if TYPE_CHECKING:
    from tr_asbs.data.atomic_data import ThermoAtomicData


@lru_cache(maxsize=1)
def _thermo_atomic_data_type():
    """Import the optional atomistic data type only for atomic sources."""
    from tr_asbs.data.atomic_data import ThermoAtomicData

    return ThermoAtomicData

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


class HarmonicPrior(torch.distributions.Distribution, Prior):
    """
    Mean-free prior with a non-isotropic covariance coupling all particle
    pairs by a harmonic spring, following the reference ASBS implementation
    (facebookresearch/adjoint_samplers, CenteredParticlesHarmonic; see also
    arXiv:2304.02198). Unlike MeanFreePrior (isotropic per-coordinate noise,
    then mean-centered), this correlates particle positions so that samples
    already resemble a loosely-connected cluster rather than an unstructured
    cloud -- intended to reduce near-overlap configurations at the start of
    training and thereby the importance-weight variance they cause.

    Construction matches the reference exactly: A = 1.5*I - 0.5*J (J = all-
    ones) is not PSD for n_particles > 3 (its "all-particles-translate-
    together" eigenmode has a negative eigenvalue), so kron(A, I_d).inverse()
    is not a valid covariance either. torch.svd of that (indefinite) matrix
    still returns U, S >= 0 with cov ~= U @ diag(S) @ U.T only up to sign
    flips on the negative-eigenvalue directions -- i.e. the actual sampling
    distribution implicitly used by the reference code is the PSD matrix
    U[:, :rank] @ diag(S[:rank]) @ U[:, :rank].T, not the nominal (indefinite)
    one. log_prob is defined self-consistently with that same effective
    covariance (required since it feeds the Girsanov/importance-weight
    computation in integrator.run_with_logweights), not with the nominal one.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(self, n_particles: int, spatial_dim: int, scale: float = 1.0, device: str = "cpu"):
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self._dim = n_particles * spatial_dim
        self.device = device
        self._scale = float(scale)

        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([self._dim]))

        self.k = (n_particles - 1) * spatial_dim  # effective (mean-free) degrees of freedom

        # Raw (pre-centering) sampling covariance: full rank (dim, not k) --
        # the SVD of the nominal (indefinite) matrix doesn't itself restrict to
        # the mean-free subspace, that only happens via the explicit centering
        # step in sample() below, exactly as in the reference implementation.
        cov_nominal = self._compute_nominal_cov(n_particles, spatial_dim)
        U_raw, S_raw, _ = torch.linalg.svd(cov_nominal)
        self._A = (U_raw * S_raw.sqrt().unsqueeze(0)).to(device)  # (dim, dim), A @ A.T = U diag(S) U.T

        # True covariance of the actual (post-centering) samples: P @ cov_raw @ P,
        # where P is the same mean-centering projector as MeanFreePrior.cov.
        # This is what log_prob must be self-consistent with (it feeds the
        # Girsanov/importance-weight computation in integrator.run_with_logweights).
        cov_raw_effective = U_raw @ torch.diag(S_raw) @ U_raw.T
        I = torch.eye(n_particles)
        J = torch.ones(n_particles, n_particles)
        P_particle = I - J / n_particles
        P = torch.kron(P_particle, torch.eye(spatial_dim))
        cov_final = P @ cov_raw_effective @ P

        eigvals, eigvecs = torch.linalg.eigh(cov_final)
        # keep the k largest eigenvalues (rest are ~0, the centered-out translation modes)
        order = torch.argsort(eigvals, descending=True)
        self._U = eigvecs[:, order[: self.k]].to(device)   # (dim, k), orthonormal
        self._S = eigvals[order[: self.k]].clamp(min=1e-12).to(device)  # (k,)

    def _compute_nominal_cov(self, n_particles: int, spatial_dim: int) -> torch.Tensor:
        A = -0.5 * torch.ones(n_particles, n_particles)
        A[torch.arange(n_particles), torch.arange(n_particles)] = 1.0
        B = torch.eye(spatial_dim)
        return torch.kron(A, B).inverse()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def cov(self) -> torch.Tensor:
        # True covariance of the final (post-centering) samples.
        return (self._scale ** 2) * (self._U @ torch.diag(self._S) @ self._U.T)

    def sample(self, size: tuple) -> torch.Tensor:
        if not isinstance(size, tuple):
            size = (size,)
        B = 1
        for s in size:
            B *= s
        z = torch.randn(B, self._A.shape[1], device=self.device)
        samples = z @ self._A.T  # (B, dim), raw (pre-centering) Cov = A @ A.T
        samples = samples * self._scale
        samples = samples.reshape(*size, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(dim=-2, keepdim=True)
        return samples.reshape(*size, self._dim)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self._dim)
        # Coordinates in the orthonormal mean-free eigenbasis of the TRUE
        # (post-centering) covariance -- x is already mean-free, i.e. in the
        # span of self._U, so this is a lossless, isometric reparametrization
        # (no Jacobian correction needed).
        y = x_flat @ self._U  # (B, k)
        var = (self._scale ** 2) * self._S  # (k,)
        quad = (y ** 2 / var).sum(dim=-1)
        log_norm = 0.5 * self.k * math.log(2.0 * math.pi) + 0.5 * torch.log(var).sum()
        log_px = -0.5 * quad - log_norm
        return log_px.reshape(x.shape[:-1]) if x.dim() > 1 else log_px.squeeze()


class BaseSource(nn.Module, ABC):
    """Base class for sampling random atomic data (graph-batch representation,
    used by the Ala2/OpenMM+PaiNN pipeline -- see GaussianSource below).
    """

    def __init__(
        self,
        pbc: list = [False, False, False],
        cell=None,
        composition: str = None,
        num_atoms: int = None,
        temperature: float = 1.0,
        charge: int = 0,
        spin: int = 1,
        center: bool = True,
    ):
        if not ((num_atoms is None) ^ (composition is None)):
            raise ValueError(
                "Exactly one of num_particles or composition must be provided."
            )

        super().__init__()
        self.register_buffer("pbc", torch.as_tensor(pbc, dtype=torch.bool))
        if cell is None:
            cell = torch.zeros(3, 3, dtype=torch.float)
        self.register_buffer("cell", torch.as_tensor(cell, dtype=torch.float))
        if composition is not None:
            atomic_numbers = composition_to_atomic_numbers(composition)
            num_atoms = len(atomic_numbers)
        else:
            atomic_numbers = [0] * num_atoms  # Default to zero for unspecified
        self.register_buffer(
            "atomic_numbers", torch.as_tensor(atomic_numbers, dtype=torch.long)
        )
        self.num_atoms = num_atoms
        self.register_buffer(
            "temperature", torch.as_tensor(temperature, dtype=torch.float)
        )
        if charge is None:
            charge = torch.zeros(num_atoms, dtype=torch.long)
        self.register_buffer("charge", torch.as_tensor(charge, dtype=torch.long))
        if spin is None:
            spin = torch.ones(num_atoms, dtype=torch.long)
        self.register_buffer("spin", torch.as_tensor(spin, dtype=torch.long))
        self.center = center

    @abstractmethod
    def sample_single_data(self) -> ThermoAtomicData:
        """Sample a single atomic data instance."""

    def sample(self, shape) -> ThermoAtomicData:
        """Sample a batch of random atomic data."""
        if isinstance(shape, tuple):
            if len(shape) != 1:
                raise ValueError("Shape must be a single integer for batch size.")
            shape = shape[0]
        data_list = [self.sample_single_data() for _ in range(shape)]
        data = _thermo_atomic_data_type().from_data_list(data_list)
        if self.center and not is_mean_free_data(data):
            raise ValueError("Sampled data is not mean-free.")
        return data

    def replicate_sample(self, shape, num_repeats: int) -> ThermoAtomicData:
        """Sample a batch and replicate each sample num_repeats times."""
        data = self.sample(shape)
        repeated = []
        for graph_data in data.batch_to_atomicdata_list():
            repeated.extend([graph_data.clone() for _ in range(num_repeats)])
        return _thermo_atomic_data_type().from_data_list(repeated).to(data.pos.device)

    def log_prob(self, data: ThermoAtomicData) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement log_prob.")


class GaussianSource(BaseSource):
    """Isotropic Gaussian prior over atomic positions (graph-batch
    representation). Counterpart of GaussianPrior above, for the Ala2/
    OpenMM+PaiNN atomic pipeline.
    """

    def __init__(
        self,
        pbc: list = [False, False, False],
        cell=None,
        composition: str = None,
        num_atoms: int = None,
        temperature: float = 1.0,
        charge: int = 0,
        spin: int = 1,
        scale: float = 1.0,
        center: bool = True,
    ):
        super().__init__(
            pbc=pbc,
            cell=cell,
            composition=composition,
            num_atoms=num_atoms,
            temperature=temperature,
            charge=charge,
            spin=spin,
            center=center,
        )
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))

    def sample_single_data(self) -> ThermoAtomicData:
        pos = torch.randn(self.num_atoms, 3, dtype=torch.float, device=self.cell.device)
        pos = pos * self.scale
        if self.center:
            pos -= pos.mean(dim=0, keepdim=True)
        return _thermo_atomic_data_type().from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)

    def log_prob(self, data: ThermoAtomicData) -> torch.Tensor:
        # Independent isotropic Gaussian prior over all coordinates in each graph.
        sigma = self.scale.to(data.pos).clamp(min=1e-12)
        two_pi = torch.tensor(2 * torch.pi, device=data.pos.device, dtype=data.pos.dtype)
        out = []
        for graph_data in data.batch_to_atomicdata_list():
            x = graph_data.pos.reshape(-1)
            dim = x.numel()
            # Effective degrees of freedom for the normalization constant:
            # when self.center=True (the sampled/scored distribution only
            # ever occupies the mean-free subspace -- COM removal cuts
            # exactly `spatial_dim` dof regardless of atom count), matching
            # MeanFreePrior's k = (n_particles - 1) * spatial_dim convention
            # elsewhere in this file. Using the full ambient `dim` here would
            # add a fixed, non-cancelling bias to every log_prob call.
            k = dim - graph_data.pos.shape[-1] if self.center else dim
            quad = (x / sigma).pow(2).sum()
            log_det = k * torch.log(sigma)
            out.append(-0.5 * (quad + k * torch.log(two_pi) + 2 * log_det))
        return torch.stack(out, dim=0)
