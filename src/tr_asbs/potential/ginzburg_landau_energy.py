from typing import Dict, Optional, Sequence, Tuple

import math

import matplotlib.pyplot as plt
import torch
import wandb

from .base_energy import BaseEnergy
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
    Visualization overlays a normalized 1D energy-slice density, not the true
    marginal density.
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
    def _energy_slice_density(
        self,
        dim: int,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        points = torch.zeros(xs.shape[0], self.dim, device=self.device, dtype=xs.dtype)
        points[:, dim] = xs
        energy = self.eval(points)
        density = torch.exp(-(energy - energy.min()))
        dx = (xs[1] - xs[0]).abs()
        return density / (density.sum() * dx).clamp_min(1e-12)

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
        xs = torch.linspace(-bound, bound, 1000, device=self.device, dtype=samples.dtype)
        xs_np = xs.detach().cpu().numpy()

        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        for ax, dim in zip(axes_flat[: min(3, len(dims))], dims[:3]):
            x = samples[:, dim].detach().cpu().numpy()
            slice_density = self._energy_slice_density(dim, xs).detach().cpu().numpy()

            ax.hist(x, bins=bins, density=True, alpha=alpha, label=f"samples (dim {dim})")
            ax.plot(xs_np, slice_density, linewidth=2.0, label="energy-slice density")
            ax.set_xlim(-bound, bound)
            ax.set_xlabel(f"x_{dim}")
            ax.set_ylabel("density")
            ax.set_title(f"1D histogram + slice density (dim {dim})")
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

        if prefix == "":
            if "lm" in results:
                logs["lag_multiplier/lambda"] = results["lm"]
            if "gamma" in results:
                logs["lag_multiplier/gamma"] = results["gamma"]

        return logs

    def __repr__(self):
        return f"GL-d:{self.dim}-m:{self.m}-delta:{self.delta}"
