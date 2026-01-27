# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from abc import ABC, abstractmethod

import ase.io
import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict
from torch.distributions import constraints

from tr_asbs.data.atomic_data import ThermoAtomicData
from tr_asbs.utils.composition import composition_to_atomic_numbers
from tr_asbs.utils.geometry import is_mean_free_data
from tr_asbs.utils.topology import pdb_to_bond_indices


class BaseSourceAtom(nn.Module, ABC):
    """Base class for sampling random atomic data."""

    def __init__(
        self,
        pbc: list[bool] | torch.Tensor = [False, False, False],  # [3,] array
        cell: list[list[float]] | torch.Tensor | None = None,  # [3, 3] array
        composition: str | None = None,
        num_atoms: int | None = None,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
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
        pass

    def sample(self, shape: int | tuple[int]) -> ThermoAtomicData:
        """Sample a batch of random atomic data."""
        if isinstance(shape, tuple):
            if len(shape) != 1:
                raise ValueError("Shape must be a single integer for batch size.")
            shape = shape[0]
        data_list = [self.sample_single_data() for _ in range(shape)]
        data = ThermoAtomicData.from_data_list(data_list)
        if self.center and not is_mean_free_data(data):
            raise ValueError("Sampled data is not mean-free.")
        return data
    
    def replicate_sample(
        self,
        batch_size: int | tuple[int],
        dupicate_size: int,
    ) -> ThermoAtomicData:
        """
        Sample a batch of random atomic data and expand it for path sampling.
        
        Args:
            batch_size: The original number of independent atomic configurations (B).
            batch_size_path: The number of paths/noise series per configuration (P).
            
        Returns:
            ThermoAtomicData: A batch of size B * P where the first B 
            configurations are sampled, and each is repeated P times.
        """
        if batch_size_path == 1:
            return self.sample(batch_size)
        
        # 1. まず、元のバッチサイズ B のデータ (独立な配置) をサンプリング
        original_data_list = [self.sample_single_data() for _ in range(batch_size)]
        
        expanded_data_list = []
        for data in original_data_list:
            for _ in range(dupicate_size):
                expanded_data_list.append(data.clone()) # データを複製してリストに追加
                
        # 3. 拡張されたリストを一つの巨大なバッチにまとめる
        data = ThermoAtomicData.from_data_list(expanded_data_list)
        if self.center and not is_mean_free_data(data):
            raise ValueError("Sampled data is not mean-free.")
            
        return data


class PeriodicUniformSourceAtom(BaseSourceAtom):
    def sample_single_data(self) -> ThermoAtomicData:
        if not self.pbc.all():
            raise ValueError("PeriodicUniformSourceAtom requires full PBCs.")
        pos = torch.rand(self.num_atoms, 3, dtype=torch.float, device=self.cell.device)
        pos = pos @ self.cell
        if self.center:
            pos -= pos.mean(dim=0, keepdim=True)
        return ThermoAtomicData.from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)


class GaussianSourceAtom(BaseSourceAtom):
    def __init__(
        self,
        pbc: list[bool] | torch.Tensor = [False, False, False],  # [3,] array
        cell: list[list[float]] | torch.Tensor | None = None,  # [3, 3] array
        composition: str | None = None,
        num_atoms: int | None = None,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
        scale: float | None = 1.0,
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
        return ThermoAtomicData.from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)


class HarmonicSourceAtom(BaseSourceAtom):
    def __init__(
        self,
        pbc: list[bool] | torch.Tensor = [False, False, False],  # [3,] array
        cell: list[list[float]] | torch.Tensor | None = None,  # [3, 3] array
        topology_pdb_file: str | None = None,
        num_atoms: int | None = None,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
        scale: float | None = 1.0,
        center: bool = True,
    ):
        composition = (
            str(ase.io.read(topology_pdb_file).symbols)
            if topology_pdb_file is not None
            else None
        )
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

        # Construct Laplacian
        if topology_pdb_file is not None:
            L = torch.zeros((self.num_atoms, self.num_atoms), dtype=torch.float)
            bond_indices = pdb_to_bond_indices(topology_pdb_file)
            L[bond_indices[0], bond_indices[0]] += 1
            L[bond_indices[1], bond_indices[1]] += 1
            L[bond_indices[0], bond_indices[1]] = -1
            L[bond_indices[1], bond_indices[0]] = -1
        else:  # Assume fully connected graph
            L = torch.eye(self.num_atoms, dtype=torch.float) * self.num_atoms
            L = L - torch.ones((self.num_atoms, self.num_atoms), dtype=torch.float)
        L = torch.kron(L, torch.eye(3, dtype=torch.float))  # Expand to 3D
        D, P = torch.linalg.eigh(L)
        D_inv = 1 / D
        D_inv[D < 1e-6] = 0.0  # Ignore small eigenvalues
        self.register_buffer("transform", P @ torch.diag(torch.sqrt(D_inv)))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))

    def sample_single_data(self) -> ThermoAtomicData:
        pos = (
            self.transform
            @ torch.randn(
                self.num_atoms * 3, dtype=torch.float, device=self.cell.device
            )
        ).reshape(self.num_atoms, 3)
        pos = pos * self.scale
        if self.center:
            pos -= pos.mean(dim=0, keepdim=True)
        return ThermoAtomicData.from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)


class PreGeneratedAtomsSource(BaseSourceAtom):
    """Source distribution defined by pre-generated atomic configurations."""

    def __init__(
        self,
        atoms_file: str,
        wrap: bool = False,
        center: bool = True,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
    ):
        nn.Module.__init__(self)
        self.atoms = ase.io.read(atoms_file, index=":")
        # Fix for empty last frame with PDB files
        if atoms_file.endswith(".pdb") and len(self.atoms[-1]) == 0:
            self.atoms = self.atoms[:-1]
        self.center = center
        self.temperature = temperature
        # Wrap and center positions if required
        for atom in self.atoms:
            positions = atom.get_positions()
            # cell = atom.get_cell().array
            # if wrap:
            #     positions = wrap_positions(positions, cell)
            if center:
                positions -= positions.mean(axis=0)
            atom.set_positions(positions)
            if "charge" not in atom.info:
                atom.info["charge"] = charge
            if "spin" not in atom.info:
                atom.info["spin"] = spin
        self.data_list = [
            ThermoAtomicData.from_ase(atoms, temperature) for atoms in self.atoms
        ]
        # Buffer for device management
        self.register_buffer("device_buffer", torch.tensor(0, dtype=torch.long))

    def sample_single_data(self) -> ThermoAtomicData:
        """Sample a single atomic data instance from the pre-generated list."""
        index = torch.randint(0, len(self.data_list), size=(1,)).item()
        batch = self.data_list[index]
        return batch.to(self.device_buffer.device)

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
