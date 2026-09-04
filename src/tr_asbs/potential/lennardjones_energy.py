# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import numpy as np
import matplotlib.pyplot as plt
from .bgflow_energy import bgflowEnergy
from .base_energy import BaseSymtheticEnergy, BaseEnergy
from ..utils.eval_logweight import compute_log_weights_metrics
import wandb

def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]

def lennard_jones_energy_torch(r, eps=1.0, rm=1.0, softcore_alpha=0.0):
    """Standard LJ when softcore_alpha=0.0 (default -- exact, unchanged
    behavior for all existing configs). When > 0, replaces r with a "softened"
    effective distance r_sc = (softcore_alpha*rm^6 + r^6)^(1/6) before
    evaluating the LJ formula (standard soft-core LJ from the free-energy-
    perturbation literature, e.g. Beutler et al. 1994). As r -> 0, r_sc ->
    softcore_alpha^(1/6)*rm (finite), so the potential no longer diverges at
    exact overlap -- it's bounded by eps*(softcore_alpha^-2 - 2*softcore_alpha^-1).
    For r not too small (r_sc ~ r), this is numerically indistinguishable from
    standard LJ: at softcore_alpha=0.01, the well region (r in [1.0, 1.5]) is
    perturbed by <0.002 (negligible vs. the well depth of eps=1.0), while the
    near-overlap energy at r=0.3 drops from ~1.9e6 to ~8.5e3 -- a ~220x
    reduction in exactly the regime believed to drive the catastrophic
    Girsanov log-importance-weight variance seen in cold-start TR-SBS training
    (see morse_energy.py for the parallel Morse-potential experiment testing
    the same hypothesis via a different mechanism).
    """
    if softcore_alpha:
        r = (softcore_alpha * rm ** 6 + r ** 6) ** (1.0 / 6.0)
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj

# from https://github.com/jarridrb/DEM/blob/main/dem/energies/lennardjones_energy.py
class LennardJonesPotential(bgflowEnergy):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
        energy_clip=None,
        softcore_alpha=0.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        energy_clip : float | None
            If set, clip each pairwise LJ term to [-energy_clip, energy_clip] before summing,
            so a single near-overlapping pair can't blow the total energy up to astronomical
            values (the 1/r^12 repulsive core diverges as r -> 0). None disables clipping.
        softcore_alpha : float
            If > 0, use a soft-core LJ formula (see lennard_jones_energy_torch) that
            smoothly bounds the r -> 0 divergence instead of a hard post-hoc clip.
            0.0 (default) reproduces standard LJ exactly.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale
        self._energy_clip = energy_clip
        self._softcore_alpha = softcore_alpha

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

    def _energy(self, x):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self._n_dims)
        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )
        # Floor the distance away from exact 0 before it hits r^-12/r^-13: with
        # the straight-through energy clip below now letting the true gradient
        # flow (instead of silently zeroing it -- see that comment), an
        # extreme near-exact overlap (r ~1e-5 or less, e.g. from a cold-started
        # or still-undertrained policy) makes the raw gradient overflow float32
        # to +-inf, which then turns into NaN downstream (inf * 0 in the
        # norm-based grad clip). 1e-2 keeps r^-13 comfortably inside float32
        # range (~1e26 vs the ~3e38 limit) while still being far smaller than
        # the energy_clip threshold (~0.36 for energy_clip=1e4), so it only
        # ever bites in this already-pathological regime.
        dists = dists.clamp(min=1e-2)

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm, self._softcore_alpha)
        if self._energy_clip is not None:
            # Straight-through clip: the clipped VALUE is used forward (numerical
            # safety for eval/logging/downstream log-weight sums), but the
            # GRADIENT flows as if unclipped. A plain torch.clip has EXACTLY
            # ZERO gradient beyond the clip boundary -- since grad_E()
            # backprops through this same _energy() call, that silently
            # zeroed out the repulsive push-apart signal for exactly the
            # near-overlapping pairs that most need a corrective gradient
            # (verified numerically: for a pair at r=0.124, clipped grad_E
            # gives near-identical, not opposite, gradients for the two
            # particles -- i.e. no repulsive signal at all -- while the
            # unclipped gradient correctly points them apart, just at an
            # enormous, numerically risky magnitude). The existing norm-based
            # grad_clip_val (ASBSTerminalCost/BaseEnergy._clip_score) already
            # exists specifically to bound that raw magnitude while preserving
            # direction, so this straight-through pass lets it do that job
            # instead of the gradient being zero before it ever gets there.
            clipped = torch.clip(lj_energies, -self._energy_clip, self._energy_clip)
            lj_energies = lj_energies + (clipped - lj_energies).detach()
        lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(*batch_shape)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def _log_prob(self, x):
        return -self._energy(x)


class LennardJonesEnergy(BaseSymtheticEnergy):
    def __init__(
        self,
        n_particles,
        spatial_dim,
        ref_samples_path,
        alpha_max: float = 100.,
        device="cpu",
        energy_clip: float | None = None,
        softcore_alpha: float = 0.0,
        softcore_alpha_start: float | None = None,
        softcore_alpha_end: float | None = None,
        softcore_alpha_anneal_epochs: int = 0,
        softcore_alpha_anneal_power: float = 1.0,
    ):
        """softcore_alpha_start/_end/_anneal_epochs (all optional): anneal
        softcore_alpha from softcore_alpha_start down to softcore_alpha_end
        over the first softcore_alpha_anneal_epochs epochs of training, then
        hold at softcore_alpha_end. Motivation: start training with a very
        soft (safe, always-informative-gradient) potential while the policy
        is still cold-started and far from equilibrium, then gradually
        sharpen back toward the TRUE target potential (softcore_alpha_end,
        e.g. 0.0 for exact LJ) as the policy learns to avoid overlaps on its
        own. Requires trainer support for calling update_epoch() once per
        epoch (see train_gamma_asbs_iw.py) -- a no-op if
        softcore_alpha_anneal_epochs is 0 (the default), in which case
        softcore_alpha behaves exactly as before (a fixed constant).

        softcore_alpha_anneal_power (default 1.0 = linear): alpha(frac) =
        start + (end-start)*frac**power, frac = min(1, epoch/anneal_epochs).
        power < 1 (e.g. 0.5) makes alpha drop QUICKLY through the "easy" high-
        alpha regime and then SLOW DOWN as it approaches alpha_end -- i.e.
        spends proportionally more epochs finishing near the true potential
        rather than uniformly across the whole range. Motivated by an
        observed failure mode: a linear anneal (power=1) let the MLP variant
        make real progress while alpha was still >~0.25, but gamma collapsed
        and Eq-EMD2 got WORSE again in the final stretch as alpha rushed from
        0.25 to 0 too quickly for the network to adapt.
        """
        # Set the name and dim
        super().__init__(n_particles, spatial_dim, ref_samples_path)
        self.device = device
        self.alpha_max = alpha_max

        self.softcore_alpha_start = softcore_alpha_start if softcore_alpha_start is not None else softcore_alpha
        self.softcore_alpha_end = softcore_alpha_end if softcore_alpha_end is not None else softcore_alpha
        self.softcore_alpha_anneal_epochs = softcore_alpha_anneal_epochs
        self.softcore_alpha_anneal_power = softcore_alpha_anneal_power

        self.lennard_jones = LennardJonesPotential(
            dim=n_particles * spatial_dim,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            oscillator=True,
            oscillator_scale=1.0,
            two_event_dims=False,
            energy_factor=1.0,
            energy_clip=energy_clip,
            softcore_alpha=self.softcore_alpha_start,
        )

    def update_epoch(self, epoch: int) -> None:
        """Called once per epoch by the trainer (if present) to advance the
        softcore_alpha anneal schedule. No-op once softcore_alpha_anneal_epochs
        epochs have elapsed, or if annealing wasn't configured (start==end)."""
        if self.softcore_alpha_anneal_epochs <= 0:
            return
        frac = min(1.0, epoch / self.softcore_alpha_anneal_epochs)
        frac = frac ** self.softcore_alpha_anneal_power
        alpha = self.softcore_alpha_start + frac * (self.softcore_alpha_end - self.softcore_alpha_start)
        self.lennard_jones._softcore_alpha = alpha

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view(samples.shape[0], -1)
        return - self.lennard_jones._log_prob(samples).squeeze(-1)

    def to(self, device):
        self.lennard_jones.to(device)
    
    def visualize(self, samples, show=True, title=None):
        test_data_smaller = self.sample(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = compute_distances(samples, self.n_particles, self.spatial_dim).detach().cpu()
        dist_test = compute_distances(test_data_smaller, self.n_particles, self.spatial_dim).detach().cpu()

        if self.n_particles == 13:
            bins = 100
        elif self.n_particles == 55:
            bins = 50

        axs[0].hist(
            dist_samples.view(-1),
            bins=bins,
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
        # min_energy = min(energy_test.min(), energy_samples.min()).item()
        # max_energy = max(energy_test.max(), energy_samples.max()).item()
        if self.n_particles == 13:
            min_energy = -60
            max_energy = 0

        elif self.n_particles == 55:
            min_energy = -380
            max_energy = -180

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
            wb = {"figures/vis": [wandb.Image(fig)]}

        return wb
    
    def __repr__(self):
        return f"LJ-{self.n_particles}-D{self.spatial_dim}"


class LennardJonesEnergyNoGT(BaseEnergy):
    """
    Ground-truth-free Lennard-Jones cluster energy.

    Unlike `LennardJonesEnergy`, this does not require a `ref_samples_path` and
    provides no `sample()` / distributional metrics (Sinkhorn, Eq-EMD2, energy-W2)
    against ground truth. Intended for clusters (e.g. LJ-38) where no trustworthy
    i.i.d. equilibrium reference set is available -- e.g. because the landscape
    has competing structural funnels that plain MCMC cannot equilibrate across.
    Evaluation is limited to energy statistics and the importance-weight/ESS
    diagnostics already computed by the trainer (results["log_RN..."]), following
    the same pattern as `GinzburgLandauEnergy`.
    """

    def __init__(
        self,
        n_particles: int,
        spatial_dim: int = 3,
        eps: float = 1.0,
        rm: float = 1.0,
        oscillator_scale: float = 1.0,
        energy_clip: float | None = None,
        device: str = "cpu",
    ):
        super().__init__(n_particles * spatial_dim)
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.device = device

        self.lennard_jones = LennardJonesPotential(
            dim=n_particles * spatial_dim,
            n_particles=n_particles,
            eps=eps,
            rm=rm,
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
        return -self.lennard_jones._log_prob(samples).squeeze(-1)

    def to(self, device):
        self.lennard_jones.to(device)

    def sample(self, size: tuple | int) -> torch.Tensor:
        raise NotImplementedError(
            "No ground-truth samples available: this cluster's landscape has no "
            "trustworthy i.i.d. equilibrium reference set (see LennardJonesEnergyNoGT docstring)."
        )

    @torch.no_grad()
    def visualize(self, samples, show=True, title=None):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = compute_distances(samples, self.n_particles, self.spatial_dim).detach().cpu()
        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.6,
            density=True,
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend()

        energy_samples = self.eval(samples).detach().cpu()
        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.6,
            histtype="step",
            linewidth=4,
            label="generated data",
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
        return f"LJ-{self.n_particles}-D{self.spatial_dim}-noGT"


def tile(a, dim, n_tile):
    """
    Tiles a pytorch tensor along one an arbitrary dimension.

    Parameters
    ----------
    a : PyTorch tensor
        the tensor which is to be tiled
    dim : Integer
        dimension along the tensor is tiled
    n_tile : Integer
        number of tiles

    Returns
    -------
    b : PyTorch tensor
        the tensor with dimension `dim` tiled `n_tile` times
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    order_index = torch.LongTensor(order_index).to(a).long()
    return torch.index_select(a, dim, order_index)

def distance_vectors(x, remove_diagonal=True):
    r"""
    Computes the matrix :math:`r` of all distance vectors between
    given input points where

    .. math::
        r_{ij} = x_{i} - y_{j}

    as used in :footcite:`Khler2020EquivariantFE`

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    r : torch.Tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    Examples
    --------
    TODO

    References
    ----------
    .. footbibliography::

    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
    r = r - r.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r

def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.
    
    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.
    
    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()

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