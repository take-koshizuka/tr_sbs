# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import matplotlib.pyplot as plt

from .bgflow_energy import bgflowEnergy
from .base_energy import BaseSymtheticEnergy
import wandb

# from https://github.com/noegroup/bgflow/blob/main/bgflow/distribution/energy/multi_double_well_potential.py
class MultiDoubleWellPotential(bgflowEnergy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via
    .. math::
        E_{DW}(d) = a * (d-d_{offset})^4 + b * (d-d_{offset})^2 + c.
    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._spatial_dim = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._spatial_dim)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)

class DoubleWellEnergy(BaseSymtheticEnergy):
    def __init__(
        self,
        n_particles,
        spatial_dim,
        ref_samples_path,
        device="cpu",
    ):
        # Set the name and dim
        super().__init__(n_particles, spatial_dim, ref_samples_path)
        self.device = device

        self.multi_double_well = MultiDoubleWellPotential(
            dim=self.n_particles * self.spatial_dim,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        return self.multi_double_well._energy(samples).squeeze(-1)

    def to(self, device):
        self.multi_double_well.to(device)
        
    def __repr__(self):
        return f"DW-{self.n_particles}-{self.spatial_dim})"
    
    def visualize(self, samples, title=None):
        test_data_smaller = self.sample(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        dist_samples = compute_distances(samples, self.n_particles, self.spatial_dim).detach().cpu()
        dist_test = compute_distances(test_data_smaller, self.n_particles, self.spatial_dim).detach().cpu()

        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = self.eval(samples).detach().detach().cpu()
        energy_test = self.eval(test_data_smaller).detach().detach().cpu()

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()
        if title is not None:
            wb = {f"figures/vis_{title}": [wandb.Image(fig)]}
        else:
            wb = {f"figures/vis": [wandb.Image(fig)]}

        return wb

# modified from https://github.com/noegroup/bgflow/blob/main/bgflow/utils/geometry.py
def compute_distances(x, n_particles, spatial_dim, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in spatial_dim.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, spatial_dim)
    # distances = torch.cdist(x, x)
    diff = x[:,:,None,:] - x[:,None,:,:]  # Shape: (batch, n_particles, n_particles, spatial_dim)
    distances = torch.sqrt(torch.sum(diff**2, axis=-1) + 1e-8)  # Shape: (batch, n_particles, n_particles)
    if remove_duplicates:
        distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances