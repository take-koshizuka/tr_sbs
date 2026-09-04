# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Morse-potential cluster energy -- a diagnostic substitute for
LennardJonesEnergy/LennardJonesEnergyNoGT, testing the hypothesis that LJ's
r^-12 repulsive core (a power-law divergence, which blows up *faster* than any
exponential as r -> 0) is what's driving the catastrophic Girsanov
log-importance-weight variance measured for LJ-13 (mean~300, std~2515,
max~40099 -- see scratch_diagnose_optimize_gamma.py). The Morse potential's
repulsive wall is exponential instead of power-law, i.e. genuinely softer at
short range, while everything else about the local well shape can be kept
identical.

Parametrization: eps (well depth) and rm (equilibrium distance) match
LennardJonesPotential's own defaults (eps=1.0, rm=1.0) so the energy SCALE is
directly comparable. The width parameter `a` is set so the potential's
curvature at the minimum, V''(rm), matches LJ's own V''(rm) = 72*eps/rm^2
exactly (Morse's V''(rm) = 2*eps*a^2, so a = 6/rm) -- this isolates the
comparison to "does the shape of the repulsive wall away from equilibrium
matter", without confounding it with an arbitrary difference in how stiff the
whole basin is.

IMPORTANT: a Morse-13 cluster is a genuinely different physical system from
LJ-13 -- NOT a drop-in replacement with the same equilibrium distribution.
The existing LJ-13 ground-truth reference samples (test_split_LJ13-1000.npy)
are NOT valid samples from a Morse-13 equilibrium distribution, so this class
follows the NoGT (LennardJonesEnergyNoGT) pattern -- energy statistics and the
trainer's own importance-weight/ESS diagnostics only, no Eq-EMD2/Sinkhorn/W2
metrics against a (nonexistent) reference set.
"""
import torch
import matplotlib.pyplot as plt
import wandb

from .lennardjones_energy import (
    LennardJonesPotential,
    distance_vectors,
    distances_from_vectors,
    compute_distances,
)
from .base_energy import BaseEnergy
from ..utils.eval_logweight import compute_log_weights_metrics


def morse_energy_torch(r, eps=1.0, rm=1.0, a=6.0):
    """V(r) = eps * (exp(-2a(r-rm)) - 2*exp(-a(r-rm))). Minimum -eps at r=rm,
    matching lennard_jones_energy_torch's own convention exactly."""
    x = a * (r - rm)
    return eps * (torch.exp(-2 * x) - 2 * torch.exp(-x))


class MorsePotential(LennardJonesPotential):
    """Same bgflowEnergy scaffolding (distance computation, mean-free harmonic
    oscillator confinement, energy clipping, two_event_dims handling) as
    LennardJonesPotential -- only the pairwise energy function itself differs
    -- so this subclasses it and overrides just `_energy`'s pairwise term via
    duplicating the parent's _energy but swapping the energy function. Kept
    as a thin subclass (not a full reimplementation) to guarantee identical
    geometry/oscillator/clipping behavior to the LJ baseline it's compared
    against.
    """

    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        a=6.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
        energy_clip=None,
    ):
        super().__init__(
            dim=dim,
            n_particles=n_particles,
            eps=eps,
            rm=rm,
            oscillator=oscillator,
            oscillator_scale=oscillator_scale,
            two_event_dims=two_event_dims,
            energy_factor=energy_factor,
            energy_clip=energy_clip,
        )
        self._a = a

    def _energy(self, x):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self._n_dims)
        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        morse_energies = morse_energy_torch(dists, self._eps, self._rm, self._a)
        if self._energy_clip is not None:
            # Straight-through clip -- see the identical fix (and its rationale)
            # in lennardjones_energy.py's _energy(): a plain torch.clip has zero
            # gradient beyond its bounds, which backprops through grad_E() and
            # silently kills the repulsive push-apart signal for near-overlaps.
            # Morse itself is bounded/smooth even at r=0 (unlike LJ's r^-12), so
            # no distance floor is needed here to keep the gradient finite.
            clipped = torch.clip(morse_energies, -self._energy_clip, self._energy_clip)
            morse_energies = morse_energies + (clipped - morse_energies).detach()
        morse_energies = morse_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(*batch_shape)
            morse_energies = morse_energies + osc_energies * self._oscillator_scale

        return morse_energies[:, None]


class MorseEnergyNoGT(BaseEnergy):
    """Ground-truth-free Morse cluster energy. See module docstring for why
    this follows LennardJonesEnergyNoGT's pattern rather than
    LennardJonesEnergy's (no valid reference sample set exists)."""

    def __init__(
        self,
        n_particles: int,
        spatial_dim: int = 3,
        eps: float = 1.0,
        rm: float = 1.0,
        a: float = 6.0,
        oscillator_scale: float = 1.0,
        energy_clip: float | None = None,
        device: str = "cpu",
    ):
        super().__init__(n_particles * spatial_dim)
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.device = device

        self.morse = MorsePotential(
            dim=n_particles * spatial_dim,
            n_particles=n_particles,
            eps=eps,
            rm=rm,
            a=a,
            oscillator=True,
            oscillator_scale=oscillator_scale,
            two_event_dims=False,
            energy_factor=1.0,
            energy_clip=energy_clip,
        )

    @property
    def is_mean_free(self) -> bool:
        return True

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view(samples.shape[0], -1)
        return -self.morse._log_prob(samples).squeeze(-1)

    def to(self, device):
        self.morse.to(device)

    def sample(self, size: tuple | int) -> torch.Tensor:
        raise NotImplementedError(
            "No ground-truth samples available for MorseEnergyNoGT -- a Morse-N "
            "cluster's equilibrium distribution differs from the corresponding "
            "Lennard-Jones system, so LJ's reference samples are not valid here."
        )

    @torch.no_grad()
    def visualize(self, samples, show=True, title=None):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = compute_distances(samples, self.n_particles, self.spatial_dim).detach().cpu()
        axs[0].hist(
            dist_samples.view(-1), bins=100, alpha=0.6, density=True,
            histtype="step", linewidth=4, label="generated data",
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend()

        energy_samples = self.eval(samples).detach().cpu()
        axs[1].hist(
            energy_samples, bins=100, density=True, alpha=0.6,
            histtype="step", linewidth=4, label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        wb_key = f"figures/vis_{title}" if title is not None else "figures/vis"
        wb = {wb_key: [wandb.Image(fig)]}
        plt.close(fig)
        return wb

    @torch.no_grad()
    def visualize_raw(self, samples, title="raw"):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)
        wb = {f"figures/vis_{title}": [wandb.Image(fig)]}
        plt.close(fig)
        return wb

    def log_details(self, results: dict, prefix: str = "", visualize: bool = True) -> dict:
        data_1 = results["data_1"]
        energy = self.eval(data_1)

        logs = {
            f"{prefix}loss/neg_avg_energy": energy.mean().detach().cpu(),
            f"{prefix}energy/std": energy.std(unbiased=False).detach().cpu(),
            f"{prefix}energy/min": energy.min().detach().cpu(),
            f"{prefix}energy/max": energy.max().detach().cpu(),
        }

        for name, value in results.items():
            if not name.startswith("log_RN"):
                continue
            base_name = name[len("log_"):] if name.startswith("log_") else name
            stabilize = not name.endswith("det")
            metrics = compute_log_weights_metrics(value, stabilize=stabilize, normalize=stabilize)
            logs.update({f"{prefix}{base_name}/{key}": val for key, val in metrics.items()})

        if visualize:
            logs.update(self.visualize(data_1, title=f"{prefix}data_1"))

        if prefix == "":
            if "lm" in results:
                logs["lag_multiplier/lambda"] = results["lm"]
            if "gamma" in results:
                logs["lag_multiplier/gamma"] = results["gamma"]
            if "gamma_h" in results:
                logs["lag_multiplier/gamma_h"] = results["gamma_h"]

        return logs

    def __repr__(self):
        return f"Morse-{self.n_particles}-D{self.spatial_dim}-noGT"
