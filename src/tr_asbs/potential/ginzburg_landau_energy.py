from typing import Dict, Optional, Sequence, Tuple

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from .base_energy import BaseEnergy
from ..utils.distribution_distance import emd2_1d_samples
from ..utils.eval_logweight import compute_log_weights_metrics


class GinzburgLandauPotential:
    """
    Ginzburg-Landau variant of the many-well target:
      E(x) = 0.5 * sum_{i=1}^d (x_i - x_{i+1})^2
           + 0.5 * sum_{i=1}^m (x_i^2 - delta)^2
           + 0.5 * sum_{i=m+1}^d x_i^2

    By default x_{d+1} = x_1.
    """

    def __init__(
        self,
        dim: int,
        m: int,
        delta: float,
        periodic_boundary: bool = True,
        coupling_strength: float = 1.0,
    ):
        assert 0 < m <= dim, "m must be in [1, dim]"
        self.dim = dim
        self.m = m
        self.delta = torch.as_tensor(delta, dtype=torch.get_default_dtype())
        self.periodic_boundary = periodic_boundary
        self.coupling_strength = coupling_strength

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d)
        returns: (..., 1)
        """
        x = x.view(*x.shape[:-1], self.dim)
        prefix = x[..., : self.m]
        suffix = x[..., self.m :]
        delta = self.delta.to(device=x.device, dtype=x.dtype)

        if self.periodic_boundary:
            shifted = torch.roll(x, shifts=-1, dims=-1)
            coupling = 0.5 * self.coupling_strength * ((x - shifted) ** 2).sum(dim=-1)
        elif self.dim > 1:
            coupling = 0.5 * self.coupling_strength * ((x[..., :-1] - x[..., 1:]) ** 2).sum(dim=-1)
        else:
            coupling = torch.zeros_like(prefix[..., 0])

        wells = 0.5 * ((prefix**2 - delta) ** 2).sum(dim=-1)
        harmonic = 0.5 * (suffix**2).sum(dim=-1) if self.dim > self.m else torch.zeros_like(wells)

        return (coupling + wells + harmonic).unsqueeze(-1)


class GinzburgLandauEnergy(BaseEnergy):
    """
    Energy-only Ginzburg-Landau target.

    This class intentionally does not provide log_Z or ground-truth sampling.
    Visualization shows sample histograms and energy histograms.
    """

    def __init__(
        self,
        dim: int = 8,
        m: int = 5,
        delta: float = 8.0,
        periodic_boundary: bool = True,
        coupling_strength: float = 1.0,
        plot_bound: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__(dim)
        self.m = m
        self.delta = delta
        self.device = torch.device(device)
        self.periodic_boundary = periodic_boundary
        self.coupling_strength = coupling_strength
        self._plot_bound = float(plot_bound) if plot_bound is not None else 2.0 * math.sqrt(delta)

        self.potential = GinzburgLandauPotential(
            dim=dim,
            m=m,
            delta=delta,
            periodic_boundary=periodic_boundary,
            coupling_strength=coupling_strength,
        )

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.potential.energy(x).squeeze(-1)

    def log_prob_2D(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        padded = torch.zeros(*x.shape[:-1], self.dim, device=x.device, dtype=x.dtype)
        n_dims = min(x.shape[-1], self.dim)
        padded[..., :n_dims] = x[..., :n_dims]
        return -self.eval(padded)

    @property
    def log_Z(self) -> torch.Tensor:
        raise NotImplementedError("log_Z is not implemented for GinzburgLandauEnergy.")

    def sample(self, size: tuple | int) -> torch.Tensor:
        raise NotImplementedError("Ground-truth sampling is not implemented for GinzburgLandauEnergy.")

    @torch.no_grad()
    def visualize(
        self,
        samples: torch.Tensor,
        dims: Optional[Sequence[int]] = (0, 1, 2, 3),
        title: Optional[str] = None,
        bins: int = 100,
        alpha: float = 0.6,
        figsize: Tuple[int, int] = (10, 8),
        use_wandb: bool = True,
    ):
        samples = samples.to(self.device)
        if samples.ndim != 2:
            samples = samples.reshape(-1, samples.shape[-1])

        sample_dim = samples.shape[1]
        dims = tuple(d for d in (dims or (0, 1, 2, 3)) if 0 <= d < min(sample_dim, self.dim))
        if not dims:
            raise ValueError(f"No valid dims to plot (samples have D={sample_dim}).")

        bound = self._plot_bound

        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        for ax, dim in zip(axes_flat[: min(3, len(dims))], dims[:3]):
            x = samples[:, dim].detach().cpu().numpy()

            ax.hist(x, bins=bins, density=True, alpha=alpha, label=f"samples (dim {dim})")
            ax.set_xlim(-bound, bound)
            ax.set_xlabel(f"x_{dim}")
            ax.set_ylabel("density")
            ax.set_title(f"1D histogram (dim {dim})")
            ax.legend(loc="best")

        used = min(3, len(dims))
        for ax in axes_flat[used:3]:
            ax.axis("off")

        energy = self.eval(samples).detach().cpu().numpy()
        ax_energy = axes_flat[3]
        ax_energy.hist(energy, bins=bins, density=True, alpha=0.7, label="Energy (input)")
        ax_energy.set_xlabel("energy")
        ax_energy.set_ylabel("density")
        ax_energy.set_title("Energy distribution")
        ax_energy.legend(loc="best")

        if title is None:
            fig.suptitle("Ginzburg-Landau samples")
            wb_key = "figures/vis_ginzburg_landau"
        else:
            fig.suptitle(title)
            wb_key = f"figures/vis_ginzburg_landau_{title}"

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if use_wandb:
            dim0 = samples[:, 0].detach().cpu().numpy()
            fig_dim0, ax_dim0 = plt.subplots(1, 1, figsize=(6, 4))
            ax_dim0.hist(dim0, bins=bins, density=True, alpha=alpha, label="samples (dim 0)")
            ax_dim0.set_xlim(-bound, bound)
            ax_dim0.set_xlabel("x_0")
            ax_dim0.set_ylabel("density")
            ax_dim0.set_title("1D histogram (dim 0)")
            ax_dim0.legend(loc="best")
            fig_dim0.tight_layout()

            wb = {
                wb_key: [wandb.Image(fig)],
                f"{wb_key}_dim0_hist": [wandb.Image(fig_dim0)],
            }
            plt.close(fig)
            plt.close(fig_dim0)
            return wb
        return fig

    @torch.no_grad()
    def _symmetry_violation_metrics(self, samples: torch.Tensor, bins: int = 100) -> Dict[str, torch.Tensor]:
        samples = samples.reshape(-1, samples.shape[-1]).detach()
        mean_vec = samples.mean(dim=0)
        odd3_vec = (samples**3).mean(dim=0)
        w2_per_dim = []
        for dim in range(samples.shape[1]):
            x_dim = samples[:, dim]
            emd2 = emd2_1d_samples(x_dim, -x_dim)
            w2_per_dim.append(torch.sqrt(torch.clamp(emd2, min=0.0)))
        w2_tensor = torch.stack(w2_per_dim)
        x0_std = samples[:, 0].std(unbiased=False).clamp_min(1e-12)

        return {
            "first_moment_norm": torch.linalg.vector_norm(mean_vec).detach().cpu(),
            "odd3_moment_norm": torch.linalg.vector_norm(odd3_vec).detach().cpu(),
            "w2_x0_mirror": w2_tensor[0].detach().cpu(),
            "w2_x0_mirror_norm": (w2_tensor[0] / x0_std).detach().cpu(),
            "w2_mirror_mean": w2_tensor.mean().detach().cpu(),
            "w2_mirror_sum": w2_tensor.sum().detach().cpu(),
            "w2_mirror_max": w2_tensor.max().detach().cpu(),
        }

    @torch.no_grad()
    def visualize_symmetry_violation(
        self,
        samples: torch.Tensor,
        title: Optional[str] = None,
        bins: int = 100,
        use_wandb: bool = True,
    ):
        samples = samples.reshape(-1, samples.shape[-1]).to(self.device)
        metrics = self._symmetry_violation_metrics(samples, bins=bins)
        bound = self._plot_bound
        samples_np = samples.detach().cpu().numpy()
        dim_ids = np.arange(samples_np.shape[1])
        w2_per_dim = []
        for dim in range(samples_np.shape[1]):
            emd2 = emd2_1d_samples(samples[:, dim], -samples[:, dim])
            w2_per_dim.append(float(torch.sqrt(torch.clamp(emd2, min=0.0)).detach().cpu()))
        w2_per_dim = np.asarray(w2_per_dim, dtype=np.float64)

        plt.close()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        ax_l, ax_r = axes[0, 0], axes[0, 1]

        x0 = samples_np[:, 0]
        ax_l.hist(x0, bins=bins, range=(-bound, bound), density=True, alpha=0.6, label="x_0")
        ax_l.hist(-x0, bins=bins, range=(-bound, bound), density=True, alpha=0.6, label="-x_0")
        ax_l.set_xlim(-bound, bound)
        ax_l.set_xlabel("value")
        ax_l.set_ylabel("density")
        ax_l.set_title("Dim 0 vs mirrored")
        ax_l.legend(loc="best")

        ax_r.bar(dim_ids, w2_per_dim)
        ax_r.set_xlabel("dim")
        ax_r.set_ylabel("W2(x_i, -x_i)")
        ax_r.set_title("Per-dim symmetry violation")

        fig.suptitle(
            f"symmetry: ||E[x]||={float(metrics['first_moment_norm']):.3e}, "
            f"||E[x^3]||={float(metrics['odd3_moment_norm']):.3e}, "
            f"W2mean={float(metrics['w2_mirror_mean']):.3e}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        if title is None:
            wb_key = "figures/vis_ginzburg_landau_symmetry"
        else:
            wb_key = f"figures/vis_ginzburg_landau_symmetry_{title}"

        if use_wandb:
            wb = {wb_key: [wandb.Image(fig)]}
            plt.close(fig)
            return wb
        return fig

    @torch.no_grad()
    def _log_weight_estimates(self, log_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_weights = log_weights.detach().reshape(-1)
        n_samples = log_weights.shape[0]
        log_z_iw = torch.logsumexp(log_weights, dim=0) - math.log(n_samples)
        log_z_lw = log_weights.mean(dim=0)

        stable_log_weights = log_weights - log_weights.max()
        weights = torch.exp(stable_log_weights)
        ess = (weights.sum() ** 2) / (n_samples * (weights**2).sum().clamp_min(1e-32))

        return {
            "logZ_iw_estimate": log_z_iw.detach().cpu(),
            "logZ_lw_estimate": log_z_lw.detach().cpu(),
            "ess": ess.detach().cpu(),
        }

    def log_details(self, results: dict, prefix: str = "", visualize: bool = True) -> Dict:
        data_1 = results["data_1"]
        energy = self.eval(data_1)
        sym = self._symmetry_violation_metrics(data_1)

        logs = {
            f"{prefix}loss/neg_avg_energy": energy.mean().detach().cpu(),
            f"{prefix}energy/std": energy.std(unbiased=False).detach().cpu(),
            f"{prefix}energy/min": energy.min().detach().cpu(),
            f"{prefix}energy/max": energy.max().detach().cpu(),
            f"{prefix}symmetry/first_moment_norm": sym["first_moment_norm"],
            f"{prefix}symmetry/odd3_moment_norm": sym["odd3_moment_norm"],
            f"{prefix}symmetry/w2_x0_mirror": sym["w2_x0_mirror"],
            f"{prefix}symmetry/w2_x0_mirror_norm": sym["w2_x0_mirror_norm"],
            f"{prefix}symmetry/w2_mirror_mean": sym["w2_mirror_mean"],
            f"{prefix}symmetry/w2_mirror_max": sym["w2_mirror_max"],
        }

        for name, value in results.items():
            if not name.startswith("log_RN"):
                continue
            base_name = name[len("log_"):] if name.startswith("log_") else name
            stabilize = not name.endswith("det")
            metrics = compute_log_weights_metrics(value, stabilize=stabilize, normalize=stabilize)
            logs.update({f"{prefix}{base_name}/{key}": val for key, val in metrics.items()})

            estimates = self._log_weight_estimates(value)
            logs.update({
                f"{prefix}{base_name}/logZ_iw_estimate": estimates["logZ_iw_estimate"],
                f"{prefix}{base_name}/logZ_lw_estimate": estimates["logZ_lw_estimate"],
                f"{prefix}{base_name}/ess_raw": estimates["ess"],
            })

            if name == "log_RN[h|u]":
                logs[f"{prefix}logZ/iw_estimate"] = estimates["logZ_iw_estimate"]
                logs[f"{prefix}logZ/lw_estimate"] = estimates["logZ_lw_estimate"]
                logs[f"{prefix}iw/ess"] = estimates["ess"]

        if visualize:
            logs.update(self.visualize(data_1, title=f"{prefix}data_1"))
            logs.update(self.visualize_symmetry_violation(data_1, title=f"{prefix}data_1"))

        if prefix == "":
            if "lm" in results:
                logs["lag_multiplier/lambda"] = results["lm"]
            if "gamma" in results:
                logs["lag_multiplier/gamma"] = results["gamma"]

        return logs

    def __repr__(self):
        return f"GL-d:{self.dim}-m:{self.m}-delta:{self.delta}"
