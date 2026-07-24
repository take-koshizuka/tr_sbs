from typing import List, Optional, Tuple, Union, Sequence, Dict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import matplotlib.pyplot as plt
import wandb
from scipy.integrate import quad

from .base_energy import BaseEnergy
from ..utils.distribution_distance import emd2_1d_samples

class ManyWellPotential:
    """
    Potential for the 'Many-well' target:
      E_{MW}(x) = sum_{i=1..m} (x_i^2 - delta)^2 + 0.5 * sum_{i=m+1..d} x_i^2
    """

    def __init__(self, dim: int, m: int, delta: float):
        assert 0 < m <= dim, "m must be in [1, dim]"
        self.m = m
        # keep delta as a scalar tensor for broadcasting on device
        self.dim = dim
        self.delta = torch.as_tensor(delta, dtype=torch.get_default_dtype())

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d)
        returns: (..., 1) energy
        """
        x = x.view(*x.shape[:-1], self.dim)
        prefix = x[..., : self.m]
        suffix = x[..., self.m :]

        k = ((prefix ** 2 - self.delta) ** 2).sum(dim=-1)  # (...) 
        k2 = 0.5 * (suffix ** 2).sum(dim=-1) if self.dim > self.m else torch.zeros_like(k)

        e = k + k2
        return e.unsqueeze(-1)  # (..., 1)


class ManyWellEnergy(BaseEnergy):
    """
    PyTorch rewrite of JAX ManyWellEnergy with a structure similar to
    MultiDoubleWellEnergy / MultiDoubleWellPotential.

    - log_prob(x) = - potential.energy(x)
    - log_prob_2D: only first two dims used (others ignored)
    - visualize: 2D contour + scatter (marginal pair) and hist
    - sample: rejection sampling for double-well coords + Gaussian for the rest
    - log_Z: Monte-Carlo estimate as in the JAX code (factorized)
    """

    def __init__(
        self,
        dim: int = 5,
        m: int = 5,
        delta: float = 4.0,
        device: str = "cpu",
    ):
        super().__init__(dim)
        self.m = m
        self.delta = delta
        self.device = torch.device(device)
        plotting_bound = 2.0 * np.sqrt(delta)
        self._plot_bound = float(plotting_bound)

        self.potential = ManyWellPotential(dim=dim, m=m, delta=delta)

        # cache for 1D logZ (double well), computed in log_Z()
        self._logZ_1d_dw: Optional[torch.Tensor] = None
        self._logZ: Optional[torch.Tensor] = None
    # -------------------------
    # probabilities
    # -------------------------
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d)
        returns: (...) log_prob
        """
        x = x.to(self.device)
        return self.potential.energy(x).squeeze(-1)

    def log_prob_2D(self, x: torch.Tensor) -> torch.Tensor:
        """
        Only uses first two dimensions with the double-well term.
        """
        x = x.to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x2 = x[..., :2]
        delta = self.potential.delta
        k = ((x2 ** 2 - delta) ** 2).sum(dim=-1)
        return -k

    # -------------------------
    # partition function (factorized MC estimate)
    # -------------------------
    @property
    def log_Z(self) -> torch.Tensor:
        """
        Numerical integration (quadrature) approximation.
        log Z = m * log ∫ exp(-(x^2 - δ)^2) dx + (d-m) * log ∫ exp(-0.5 x^2) dx
        """
        # Define the limits for the double-well integral
        # Since the integrand exp(-(x^2-delta)^2) decays very fast, 
        # -inf to inf is usually safe and accurate with quad.
        limit = np.inf 

        if self._logZ is not None:
            return self._logZ
        
        # 1D double-well integral
        if self._logZ_1d_dw is None:
            delta_val = self.potential.delta.item()  # Convert torch scalar to python float
            
            def dw_integrand(x):
                return np.exp(-(x**2 - delta_val)**2)
            
            # quad returns (integral_value, absolute_error)
            integral_dw, _ = quad(dw_integrand, -limit, limit)
            self._logZ_1d_dw = torch.tensor(np.log(integral_dw + 1e-32), device=self.device)

        # 1D Gaussian integral
        # The analytical value is sqrt(2 * pi), but here is the numerical version:
        def gauss_integrand(x):
            return np.exp(-0.5 * x**2)
        
        integral_gauss, _ = quad(gauss_integrand, -limit, limit)
        logZ_gauss_1d = torch.tensor(np.log(integral_gauss + 1e-32), device=self.device)
        
        self._logZ = self._logZ_1d_dw * self.m + logZ_gauss_1d * (self.dim - self.m)
        return self._logZ
    
    def multi_well(x):
        return np.exp(-(x**4 - 4 * x**2))
    # -------------------------
    # sampling
    # -------------------------
    def _proposal_mixture_1d(self, shift: float, separation: float) -> Tuple[D.Categorical, List[D.Normal]]:
        """
        Simple symmetric 2-component Gaussian mixture for the 1D double-well.
        Means: shift ± sqrt(separation), scales: 1/sqrt(separation).
        """
        sep = float(separation)
        mean_left = shift - math.sqrt(sep)
        mean_right = shift + math.sqrt(sep)
        scale = 1.0 / math.sqrt(sep)

        cat = D.Categorical(probs=torch.tensor([0.5, 0.5], device=self.device))
        comps = [D.Normal(loc=torch.tensor(mean_left, device=self.device),
                          scale=torch.tensor(scale, device=self.device)),
                 D.Normal(loc=torch.tensor(mean_right, device=self.device),
                          scale=torch.tensor(scale, device=self.device))]
        return cat, comps

    def _proposal_sample_logprob(self, cat: D.Categorical, comps: List[D.Normal], shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the 2-comp mixture and compute its log_prob.
        Returns samples and proposal.log_prob(samples).
        """
        k = cat.sample(shape)  # (...,)
        # gather params per-sample
        means = torch.stack([comps[0].loc, comps[1].loc])
        scales = torch.stack([comps[0].scale, comps[1].scale])
        mu = means[k]
        sigma = scales[k]
        base = D.Normal(mu, sigma)
        samples = base.rsample()  # reparam not necessary but fine
        # mixture log_prob = log( 0.5 N0 + 0.5 N1 )
        logp0 = comps[0].log_prob(samples)
        logp1 = comps[1].log_prob(samples)
        logmix = torch.logsumexp(torch.stack([math.log(0.5) + logp0,
                                              math.log(0.5) + logp1], dim=0), dim=0)
        return samples, logmix

    def _doublewell_1d_log_pdf(self, xs: torch.Tensor, shift: float, separation: float) -> torch.Tensor:
        """
        Unnormalized target pdf for 1D double-well, already normalized by logZ_1d.
        Returns log f(x).
        """
        if self._logZ_1d_dw is None:
            _ = self.log_Z  # compute cache
        delta = torch.as_tensor(separation, device=self.device, dtype=xs.dtype)
        logf = -((xs - shift) ** 2 - delta) ** 2 - self._logZ_1d_dw
        return logf

    def _rejection_sampling_1d(
        self,
        n_samples: int,
        shift: float,
        separation: float,
        scaling: float = 6.0,
    ) -> torch.Tensor:
        """
        Rejection sampling for the 1D double-well using 2-comp Gaussian mixture proposal.
        Returns: (n_samples,)
        """
        cat, comps = self._proposal_mixture_1d(shift, separation)

        collected = []
        # oversample factor; a few rounds typically suffice
        batch = int(math.ceil(n_samples * scaling * 1.5))

        while sum(t.numel() for t in collected) < n_samples:
            xs, log_q = self._proposal_sample_logprob(cat, comps, torch.Size([batch]))
            # u ~ Uniform(0, 1); accept if u < f(x) / (c q(x))
            # equivalently: log u < log f(x) - log c - log q(x)
            log_u = torch.log(torch.rand_like(xs))
            log_f = self._doublewell_1d_log_pdf(xs, shift, separation)
            accept = log_u < (log_f - math.log(scaling) - log_q)
            accepted = xs[accept]
            if accepted.numel() > 0:
                collected.append(accepted)

        out = torch.cat(collected)[:n_samples]
        return out

    @torch.no_grad()
    def sample(self, sample_shape: Union[Tuple[int, ...], int], seed: Optional[int] = None) -> torch.Tensor:
        """
        sample_shape: e.g., (N,) or (B, N,)
        returns: (*sample_shape, d)
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, )
            
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            randn = lambda shape: torch.randn(shape, generator=g, device=self.device)
            rand = lambda shape: torch.rand(shape, generator=g, device=self.device)
        else:
            randn = lambda shape: torch.randn(shape, device=self.device)
            rand = lambda shape: torch.rand(shape, device=self.device)

        total = int(math.prod(sample_shape))
        # first m dims: double-well via rejection sampling (independent across dims)
        dw_list = []
        for _ in range(self.m):
            dw_i = self._rejection_sampling_1d(n_samples=total, shift=0.0, separation=float(self.potential.delta))
            dw_list.append(dw_i.unsqueeze(-1))
        dw = torch.cat(dw_list, dim=-1) if dw_list else torch.empty(total, 0, device=self.device)

        # remaining dims: standard Gaussian
        n_gauss = self.dim - self.m
        gauss = randn((total, n_gauss)) if n_gauss > 0 else torch.empty(total, 0, device=self.device)

        samples = torch.cat([dw, gauss], dim=-1).reshape(*sample_shape, self.dim)
        return samples

    # -------------------------
    # visualization
    # -------------------------
    @torch.no_grad()
    def visualize(
        self,
        samples: torch.Tensor,
        dims: Optional[Sequence[int]] = (0, 1, 2, 3),
        title: Optional[str] = None,
        bins: int = 100,
        alpha: float = 0.6,
        figsize: tuple = (10, 8),
        use_wandb: bool = True,
    ):
        device = self.device
        samples = samples.to(device)

        # ensure 1D double-well normalizer is computed (self._logZ_1d_dw)
        if self._logZ_1d_dw is None:
            _ = self.log_Z

        # plotting range
        bound = self._plot_bound

        # grid for theoretical 1D marginal density of the double-well
        xs = torch.linspace(-bound, bound, 1000, device=device, dtype=torch.get_default_dtype())
        delta = self.potential.delta  # scalar tensor
        # log p_1d(x) = -((x^2 - delta)^2) - logZ_1d
        logp_1d = -((xs**2 - delta)**2) - self._logZ_1d_dw
        p_1d = torch.exp(logp_1d).detach().cpu().numpy()
        xs_np = xs.detach().cpu().numpy()

        # 可視化できる次元をサンプル形状に合わせてクリップ
        D = samples.shape[1]
        dims = tuple(d for d in (dims or (0, 1, 2, 3)) if 0 <= d < D)
        if len(dims) == 0:
            raise ValueError(f"No valid dims to plot (samples have D={D}).")

        # 常に 2x2 グリッドを作り、4枚目をエネルギー図に使う
        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        # 1〜3枚目: 各次元のヒスト＋理論密度（dims から最大3つまで）
        for ax, d in zip(axes_flat[:min(3, len(dims))], dims[:3]):
            x = samples[:, d].detach().cpu().numpy()
            ax.hist(x, bins=bins, density=True, alpha=alpha, label=f"samples (dim {d})")
            ax.plot(xs_np, p_1d, linewidth=2.0, label="theoretical density")
            ax.set_xlim(-bound, bound)
            ax.set_xlabel(f"x_{d}")
            ax.set_ylabel("density")
            ax.set_title(f"1D hist + density (dim {d})")
            ax.legend(loc="best")

        # 4枚目: エネルギー分布（入力 vs Ground-truth）
        ax_energy = axes_flat[3]
        # Ground-truth samples: 同数で比較（計算量を落としたいときはサブサンプリングでもOK）
        N = samples.shape[0]
        gt = self.sample((N,))
        E_in = self.potential.energy(samples).squeeze(-1).detach().cpu().numpy()
        E_gt = self.potential.energy(gt).squeeze(-1).detach().cpu().numpy()

        ax_energy.hist(E_in, bins=bins, density=True, alpha=0.6, label="Energy (input)")
        ax_energy.hist(E_gt, bins=bins, density=True, alpha=0.6, label="Energy (ground-truth)")
        ax_energy.set_xlabel("energy")
        ax_energy.set_ylabel("density")
        ax_energy.set_title("Energy distribution: input vs ground-truth")
        ax_energy.legend(loc="best")

        # 余ったサブプロット（1〜3枚に満たない場合）を非表示
        used = min(3, len(dims))
        for ax in axes_flat[used:3]:
            ax.axis("off")

        if title is None:
            fig.suptitle("1D hist + density (first 3 dims) + Energy histogram")
            wb_key = "figures/vis_tiled"
        else:
            fig.suptitle(title)
            wb_key = f"figures/vis_tiled_{title}"

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if use_wandb:
            wb = {wb_key: [wandb.Image(fig)]}
            return wb
        else:
            return fig
    
    def score(self, samples: torch.Tensor) -> torch.Tensor:
        grad_fxn = torch.func.grad(self.log_prob)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples)

    def __repr__(self):
        return f'MW-d:{self.dim}-m:{self.m}-delta:{self.delta}'
    
    @torch.no_grad()
    def metric_emd_per_dim(
        self,
        samples: torch.Tensor,
        gt_samples: Optional[torch.Tensor] = None,
        seed: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        samples: (N, D)  ※他の形状なら先頭次元をNとして整形します
        gt_samples: None の場合は self.sample((N,), seed=seed) を使う
        returns dict:
          - emd_per_dim: (D,)
          - emd_mean: scalar
          - emd_max: scalar
          - emd_argmax: scalar long
        """
        samples = samples.to(self.device)
        if samples.ndim != 2:
            samples = samples.reshape(-1, samples.shape[-1])

        N, D = samples.shape

        if gt_samples is None:
            gt_samples = self.sample((N,), seed=seed)
        else:
            gt_samples = gt_samples.to(self.device)
            if gt_samples.ndim != 2:
                gt_samples = gt_samples.reshape(-1, gt_samples.shape[-1])

        # 次元を合わせる（安全策）
        D2 = min(D, gt_samples.shape[1])
        samples = samples[:, :D2]
        gt_samples = gt_samples[:, :D2]

        emds = []
        for i in range(D2):
            emd_i = emd2_1d_samples(samples[:, i], gt_samples[:, i])
            emds.append(emd_i.squeeze())
        emd_per_dim = torch.stack(emds)  # (D2,)

        emd_mean = emd_per_dim.mean()
        emd_sum  = emd_per_dim.sum()
        emd_max = emd_per_dim.max()
        emd_argmax = torch.argmax(emd_per_dim)

        return {
            "emd_mean": emd_mean.detach().cpu(),
            "emd_sum":  emd_sum.detach().cpu(),
            "emd_max": emd_max.detach().cpu(),
            "emd_argmax": emd_argmax.detach().cpu(),
        }

    @torch.no_grad()
    def plot_1d_hist_with_theory(
        self,
        samples: torch.Tensor,
        dim: int,
        gt_samples: Optional[torch.Tensor] = None,
        seed: Optional[int] = 0,
        bins: int = 80,
        bound: Optional[float] = None,
        alpha: float = 0.6,
        n_grid: int = 1000,
        title: Optional[str] = None,
        use_wandb: bool = True,
    ):
        """
        指定dimについて:
          - samples ヒスト
          - gt_samples ヒスト（未指定なら生成）
          - 理論 1D 周辺密度（energyから導出）を線で重ねる
        """
        device = self.device
        samples = samples.to(device)
        if samples.ndim != 2:
            samples = samples.reshape(-1, samples.shape[-1])

        N, D = samples.shape
        if not (0 <= dim < D):
            raise ValueError(f"dim={dim} is out of range for D={D}")

        if gt_samples is None:
            gt_samples = self.sample((N,), seed=seed)
        else:
            gt_samples = gt_samples.to(device)
            if gt_samples.ndim != 2:
                gt_samples = gt_samples.reshape(-1, gt_samples.shape[-1])

        D2 = min(D, gt_samples.shape[1])
        if dim >= D2:
            raise ValueError(f"dim={dim} is out of range after matching gt_samples (D2={D2})")

        x_s = samples[:, dim].detach().cpu().numpy()
        x_g = gt_samples[:, dim].detach().cpu().numpy()

        if bound is None:
            bound = self._plot_bound if hasattr(self, "_plot_bound") else 3.0

        # --- theoretical 1D marginal from energy ---
        xs = torch.linspace(-bound, bound, n_grid, device=device, dtype=torch.get_default_dtype())

        if dim < self.m:
            # p(x) ∝ exp(- (x^2 - δ)^2 ), normalize by 1D Z
            # 既に self.log_Z があるなら、その中の _logZ_1d_dw を使える
            if self._logZ_1d_dw is None:
                _ = self.log_Z  # compute and cache _logZ_1d_dw

            delta = self.potential.delta.to(device=device, dtype=xs.dtype)
            logp = -((xs**2 - delta)**2) - self._logZ_1d_dw
            p_theory = torch.exp(logp)
        else:
            # Standard normal exactly normalized
            logp = -0.5 * xs**2 - 0.5 * math.log(2 * math.pi)
            p_theory = torch.exp(logp)

        xs_np = xs.detach().cpu().numpy()
        p_np = p_theory.detach().cpu().numpy()

        # --- plot ---
        plt.close()
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()

        ax.hist(x_s, bins=bins, density=True, alpha=alpha, label=f"samples (dim {dim})")
        ax.hist(x_g, bins=bins, density=True, alpha=alpha, label=f"gt samples (dim {dim})")
        ax.plot(xs_np, p_np, linewidth=2.0, label="theoretical 1D marginal")

        ax.set_xlim(-bound, bound)
        ax.set_xlabel(f"x_{dim}")
        ax.set_ylabel("density")

        if title is None:
            ax.set_title(f"1D histogram + theoretical density (dim {dim})")
            wb_key = f"figures/vis_1d"
        else:
            ax.set_title(title)
            wb_key = f"figures/{title}_vis_1d"

        ax.legend(loc="best")
        fig.tight_layout()

        if use_wandb:
            wb = {wb_key: [wandb.Image(fig)]}
            return wb
        else:
            return fig
        
    @torch.no_grad()
    def _compute_empirical_pdf_torch(
        self,
        samples_1d: torch.Tensor,
        x_grid: torch.Tensor,
        dx: float,
        x_range: Tuple[float, float],
        n_bins: int,
    ) -> torch.Tensor:
        xmin, xmax = x_range
        # torch.histc: bins 個の counts (float) を返す
        counts = torch.histc(samples_1d, bins=n_bins, min=xmin, max=xmax)
        total = counts.sum()

        if total > 0:
            pdf = counts / (total * dx)
        else:
            pdf = torch.ones_like(counts) / (n_bins * dx)  # uniform fallback
        return pdf

    @torch.no_grad()
    def _compute_true_pdf_1d_torch(
        self,
        dim: int,
        x_grid: torch.Tensor,
        dx: float,
        epsilon: float,
    ) -> torch.Tensor:
        """
        ManyWell の 1D 周辺分布（理論）を grid 上で評価して離散正規化する
        - dim < m: p(x) ∝ exp(-(x^2 - δ)^2)
        - dim >= m: p(x) = N(0,1)
        """
        device = x_grid.device
        dtype = x_grid.dtype

        if dim < self.m:
            if self._logZ_1d_dw is None:
                _ = self.log_Z  # cache _logZ_1d_dw
            delta = self.potential.delta.to(device=device, dtype=dtype)
            logp = -((x_grid**2 - delta)**2) - self._logZ_1d_dw.to(device=device, dtype=dtype)
            p = torch.exp(logp)
        else:
            # exact standard normal pdf
            logp = -0.5 * x_grid**2 - 0.5 * math.log(2.0 * math.pi)
            p = torch.exp(logp)

        # discretized renormalization so that sum p_i * dx = 1
        Z = (p.sum() * dx).clamp_min(epsilon)
        return p / Z

    @torch.no_grad()
    def _tvd(self, p: torch.Tensor, q: torch.Tensor, dx: float) -> torch.Tensor:
        return 0.5 * (p.sub(q).abs().sum() * dx)

    @torch.no_grad()
    def _kl(self, p: torch.Tensor, q: torch.Tensor, dx: float, epsilon: float) -> torch.Tensor:
        p_safe = (p + epsilon)
        q_safe = (q + epsilon)
        p_safe = p_safe / (p_safe.sum() * dx)
        q_safe = q_safe / (q_safe.sum() * dx)
        kl = (p_safe * (p_safe / q_safe).log()).sum() * dx
        return torch.clamp(kl, min=0.0)

    @torch.no_grad()
    def _js(self, p: torch.Tensor, q: torch.Tensor, dx: float, epsilon: float) -> torch.Tensor:
        p_safe = p + epsilon
        q_safe = q + epsilon
        p_safe = p_safe / (p_safe.sum() * dx)
        q_safe = q_safe / (q_safe.sum() * dx)
        m = 0.5 * (p_safe + q_safe)
        m = m / (m.sum() * dx)

        kl_pm = (p_safe * (p_safe / m).log()).sum() * dx
        kl_qm = (q_safe * (q_safe / m).log()).sum() * dx
        js = 0.5 * (kl_pm + kl_qm)
        return torch.clamp(js, min=0.0, max=math.log(2.0))

    @torch.no_grad()
    def _w1(self, p: torch.Tensor, q: torch.Tensor, dx: float) -> torch.Tensor:
        cdf_p = torch.cumsum(p, dim=0) * dx
        cdf_q = torch.cumsum(q, dim=0) * dx
        return (cdf_p.sub(cdf_q).abs().sum() * dx)

    @torch.no_grad()
    def metric_distribution_per_dim(
        self,
        samples: torch.Tensor,
        x_range: Tuple[float, float] = (-6.0, 6.0),
        n_bins: int = 100,
        epsilon: float = 1e-10,
    ) -> Dict[str, torch.Tensor]:
        """
        JAX の evaluate_single_dimension 相当を PyTorch で各次元に適用
        returns:
          - tvd_per_dim, js_per_dim, w1_per_dim, kl_per_dim, rkl_per_dim: (D,)
          - tvd_mean/std, ...: scalar
        """
        samples = samples.to(self.device)
        if samples.ndim != 2:
            samples = samples.reshape(-1, samples.shape[-1])
        N, D = samples.shape

        xmin, xmax = x_range
        x_grid = torch.linspace(xmin, xmax, n_bins, device=self.device, dtype=torch.get_default_dtype())
        dx = float(x_grid[1] - x_grid[0])

        tvd_list, js_list, w1_list, kl_list, rkl_list = [], [], [], [], []

        for d in range(D):
            emp_pdf = self._compute_empirical_pdf_torch(samples[:, d], x_grid, dx, x_range, n_bins)
            true_pdf = self._compute_true_pdf_1d_torch(d, x_grid, dx, epsilon)

            tvd = self._tvd(emp_pdf, true_pdf, dx)
            js  = self._js(emp_pdf, true_pdf, dx, epsilon)
            w1  = self._w1(emp_pdf, true_pdf, dx)
            kl  = self._kl(emp_pdf, true_pdf, dx, epsilon)
            rkl = self._kl(true_pdf, emp_pdf, dx, epsilon)

            tvd_list.append(tvd)
            js_list.append(js)
            w1_list.append(w1)
            kl_list.append(kl)
            rkl_list.append(rkl)

        tvd_per_dim = torch.stack(tvd_list)
        js_per_dim  = torch.stack(js_list)
        w1_per_dim  = torch.stack(w1_list)
        kl_per_dim  = torch.stack(kl_list)
        rkl_per_dim = torch.stack(rkl_list)

        out = {
            "tvd_per_dim": tvd_per_dim.detach().cpu(),
            "js_per_dim":  js_per_dim.detach().cpu(),
            "w1_per_dim":  w1_per_dim.detach().cpu(),
            "kl_per_dim":  kl_per_dim.detach().cpu(),
            "rkl_per_dim": rkl_per_dim.detach().cpu(),
            "tvd_mean": tvd_per_dim.mean().detach().cpu(),
            "tvd_std":  tvd_per_dim.std(unbiased=False).detach().cpu(),
            "js_mean":  js_per_dim.mean().detach().cpu(),
            "js_std":   js_per_dim.std(unbiased=False).detach().cpu(),
            "w1_mean":  w1_per_dim.mean().detach().cpu(),
            "w1_std":   w1_per_dim.std(unbiased=False).detach().cpu(),
            "kl_mean":  kl_per_dim.mean().detach().cpu(),
            "kl_std":   kl_per_dim.std(unbiased=False).detach().cpu(),
            "rkl_mean": rkl_per_dim.mean().detach().cpu(),
            "rkl_std":  rkl_per_dim.std(unbiased=False).detach().cpu(),
        }
        return out
        
    def log_Z_error(self, log_weights: torch.Tensor) -> torch.Tensor:
        """
        log_weights: [N]
        """
        logZ = torch.logsumexp(log_weights, dim=0) - math.log(log_weights.shape[0])
        logZ_lw = log_weights.mean(dim=0)
        true_logZ = float(self.log_Z)
        abs_errors = torch.abs(logZ - true_logZ)
        abs_errors_lw = torch.abs(logZ_lw - true_logZ)
        return {'logZ_error': abs_errors, 'logZ_error_lw': abs_errors_lw}
    
    def log_details(self, results: dict, prefix: str="", visualize=True) -> dict:
        logs = super().log_details(results, prefix=prefix, visualize=visualize)

        if "log_RN[h|u]" in results:
            logs.update(self.log_Z_error(results["log_RN[h|u]"]))

        pred_samples = results["data_1"]

        # --- EMD ---
        emd_val = self.metric_emd_per_dim(pred_samples)
        logs.update({
            f"{prefix}loss/emd_mean": emd_val["emd_mean"],
            f"{prefix}loss/emd_sum":  emd_val["emd_sum"],
            f"{prefix}loss/emd_max":  emd_val["emd_max"],
        })

        # --- Distribution metrics (evaluate_single_dimension 相当) ---
        dist = self.metric_distribution_per_dim(
            pred_samples,
            x_range=(-6.0, 6.0),
            n_bins=100,
            epsilon=1e-10,
        )
        logs.update({
            f"{prefix}loss/tvd_mean": dist["tvd_mean"],
            f"{prefix}loss/tvd_std":  dist["tvd_std"],
            f"{prefix}loss/js_mean":  dist["js_mean"],
            f"{prefix}loss/js_std":   dist["js_std"],
            f"{prefix}loss/w1_mean":  dist["w1_mean"],
            f"{prefix}loss/w1_std":   dist["w1_std"],
            f"{prefix}loss/kl_mean":  dist["kl_mean"],
            f"{prefix}loss/kl_std":   dist["kl_std"],
            f"{prefix}loss/rkl_mean": dist["rkl_mean"],
            f"{prefix}loss/rkl_std":  dist["rkl_std"],
        })

        # 可視化: 既存の worst-dim は EMD でやっているのでそのまま
        emd_argmax = emd_val["emd_argmax"].item()
        if visualize:
            logs_fig = self.plot_1d_hist_with_theory(
                samples=pred_samples,
                dim=emd_argmax,
                title=f"{prefix}hist_worst_dim",
                use_wandb=True,
            )
            logs.update(logs_fig)

        return logs

# -------------------------
# simple test
# -------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    model = ManyWellEnergy(dim=25, m=25, delta=4.0, device="cpu", can_sample=True)

    print("Estimating log Z ...")
    print("log_Z =", float(model.log_Z))

    print("Sampling ...")
    samples = model.sample((2000,), seed=0)  # (N, d)

    print("Visualizing ...")
    wb = model.visualize(samples)
    print("W&B payload keys:", list(wb.keys()))
