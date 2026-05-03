import math
from typing import Optional

import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import ot as pot
import numpy as np
from tr_asbs.potential.base_energy import BaseEnergy
from tr_asbs.potential.plot_density import plot_marginal_pair, plot_density
import torch.nn.functional as F


class MixtureTarget(BaseEnergy, torch.nn.Module):
    """
    Base class for mixture-based targets, adding specific evaluation metrics
    like Mode TVD, Sliced TVD, and Sinkhorn distance.
    """

    def __init__(self, dim: int, num_components: int, log_Z: Optional[float] = None, can_sample: bool = True):
        super().__init__(dim)
        self.num_components = num_components

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Identity transform."""
        return x

    def score(self, samples: torch.Tensor, temp=1.) -> torch.Tensor:
        """Computes the score (gradient of log_prob)."""
        grad_fxn = torch.func.grad(self.log_prob)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples) / temp

    def setup_train_set(self, size=100):
        return self.sample(size)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_prob(self, x: torch.Tensor, temp=1.) -> torch.Tensor:
        x = x.to(self.device)
        return self.distribution.log_prob(x) / temp

    @abstractmethod
    def log_prob_per_component(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of x under each component, including the mixture weight.

        Args:
            x: [Batch, Dim]
        Returns:
            log_probs: [Batch, NumComponents] where value [i, k] is log(pi_k * p_k(x_i))
        """
        pass

    @property
    @abstractmethod
    def component_weights(self) -> torch.Tensor:
        """Returns the mixture weights (probabilities) of the components. Shape: [NumComponents]"""
        pass

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [D] または [N, D]
        returns: スカラー or [N]
        """
        batched = (x.dim() == 2)
        if not batched:
            x = x.unsqueeze(0)
        x = x.to(self.device, self.dtype)
        logp = self.distribution.log_prob(x)
        if not batched:
            logp = logp.squeeze(0)
        return -logp

    def compute_mode_tvd(self, samples: torch.Tensor) -> float:
        """
        Computes the Total Variation Distance (TVD) between the empirical mode distribution
        of the samples and the true mixture weights.
        """
        with torch.no_grad():
            # Assign each sample to the component with the highest posterior probability
            # log_prob_per_component returns log(pi_k) + log(p(x|k))
            comp_log_probs = self.log_prob_per_component(samples)
            assignments = torch.argmax(comp_log_probs, dim=1)

            # Calculate empirical frequencies
            counts = torch.bincount(assignments, minlength=self.num_components).float()
            empirical_weights = counts / samples.shape[0]

            # True weights
            true_weights = self.component_weights.to(samples.device)

            # TVD = 0.5 * sum(|p_i - q_i|)
            tvd = 0.5 * torch.sum(torch.abs(empirical_weights.to(samples.device) - true_weights))
            return tvd.item()

    def compute_sliced_tvd(self, samples: torch.Tensor, num_projections: int = 100,
                           num_target_samples: int = 2000) -> float:
        """
        Computes Sliced TVD. Projects data onto random 1D lines, computes histograms,
        calculates TVD between histograms, and averages over projections.
        """
        if not self.can_sample:
            raise ValueError("Target must be sampleable to compute Sliced TVD.")

        with torch.no_grad():
            target_samples = self.sample(num_target_samples).to(samples.device)

            # Generate random projection vectors (normalized)
            # Shape: [num_projections, dim]
            projections = torch.randn(num_projections, self.dim, device=samples.device)
            projections = projections / torch.norm(projections, dim=1, keepdim=True)

            # Project samples: [N, D] @ [D, P]^T -> [N, P]
            proj_model = samples @ projections.T
            proj_target = target_samples @ projections.T

            tvds = []
            bins = 50

            for i in range(num_projections):
                # Determine common bounds for the histogram
                min_val = min(proj_model[:, i].min().item(), proj_target[:, i].min().item())
                max_val = max(proj_model[:, i].max().item(), proj_target[:, i].max().item())

                # Compute histograms
                hist_model = torch.histc(proj_model[:, i], bins=bins, min=min_val, max=max_val)
                hist_target = torch.histc(proj_target[:, i], bins=bins, min=min_val, max=max_val)

                # Normalize to get probabilities
                prob_model = hist_model / (hist_model.sum() + 1e-8)
                prob_target = hist_target / (hist_target.sum() + 1e-8)

                # Compute TVD for this slice
                slice_tvd = 0.5 * torch.sum(torch.abs(prob_model - prob_target))
                tvds.append(slice_tvd)

            return torch.tensor(tvds).mean().item()

    def compute_sinkhorn(self, samples: torch.Tensor, num_target_samples: int = 2000, reg: float = 0.1) -> float:
        """
        Computes the Sinkhorn distance (entropic regularization of Wasserstein)
        between the generated samples and true samples.
        """
        if pot is None:
            raise ImportError("Python Optimal Transport (ot) library is not installed.")

        if not self.can_sample:
            raise ValueError("Target must be sampleable to compute Sinkhorn distance.")

        with torch.no_grad():
            target_samples = self.sample(num_target_samples).to(samples.device)

            # Uniform weights for samples
            a = torch.ones(samples.shape[0], device=samples.device) / samples.shape[0]
            b = torch.ones(target_samples.shape[0], device=samples.device) / target_samples.shape[0]

            # Cost matrix (Squared Euclidean distance)
            M = torch.cdist(samples, target_samples, p=2) ** 2

            # Normalize cost matrix to help stability
            M = M / M.max()

            # Compute Sinkhorn loss
            loss = pot.sinkhorn2(a, b, M, reg)
            return loss.item()

    def visualize(
            self,
            trainer,
            show=True,
    ) -> dict:
        """
        Visualizes 1D and 2D marginals of the provided samples against the target density.
        """
        import wandb

        integration_outcome = trainer.sde_integrate(
            current_conditioning_outcome=None,
            compute_target_scores_and_weights=False,
        )

        model_samples = integration_outcome.x_T
        model_samples = model_samples.detach().cpu()
        num_samples = model_samples.shape[0]
        dim = self.dim

        # Limit dimensions for visualization if too high (e.g., only first 4)
        vis_dim = min(dim, 4)

        fig, axes = plt.subplots(vis_dim, vis_dim, figsize=(3 * vis_dim, 3 * vis_dim))

        # Handle 1D case separately to avoid indexing issues
        if vis_dim == 1:
            axes = np.array([[axes]])
        elif vis_dim > 1 and len(axes.shape) == 1:
            # Handle case where subplots might return 1D array
            axes = axes.reshape(vis_dim, vis_dim)

        # Plot bounds (can be inferred or use defaults)
        bounds = (-5, 5)  # Default, could be made an attribute

        for i in range(vis_dim):
            for j in range(vis_dim):
                ax = axes[i, j]

                if i == j:
                    # 1D Marginal (Diagonal)
                    ax.hist(model_samples[:, i].numpy(), bins=50, density=True, alpha=0.6, color='red', label='Model')

                    # If we can sample, plot target histogram too
                    if self.can_sample:
                        target_samples = self.sample(num_samples).cpu()
                        ax.hist(target_samples[:, i].numpy(), bins=50, density=True, alpha=0.3, color='blue',
                                label='Target')

                    ax.set_title(f"Dim {i}")
                    if i == 0 and j == 0:
                        ax.legend(fontsize='x-small')

                else:
                    # 2D Marginal (Off-diagonal)
                    # Use utility from plot_density.py
                    plot_marginal_pair(
                        model_samples, ax, label='Model', color='red', marginal_dims=(j, i), bounds=bounds
                    )

                    # Overlay target contours if feasible (requires integration or projection of log_prob)
                    # For simplicity, we plot samples or just use the log_prob if it were a 2D problem.
                    # As a generic solution, we often just scatter plot target samples.
                    if self.can_sample:
                        target_samples = self.sample(num_samples).cpu()
                        # Clip for cleanliness
                        target_samples = torch.clamp(target_samples, bounds[0], bounds[1])
                        ax.scatter(target_samples[:, j], target_samples[:, i], c='blue', alpha=0.1, marker='.',
                                   label='Target')

        plt.tight_layout()

        wb = {"figures/vis_marginals": fig}
        plt.close(fig)  # free memory after creating image
        if show:
            plt.show()

        return wb


class RingGMM(MixtureTarget):
    def __init__(self, device, dim: int, radius: float = 4.0, scale: float = 1.0,
                 log_Z: float = 0., can_sample: bool = True, dtype: torch.dtype = torch.float32,
):
        # We specifically want 9 components
        num_components = 9
        super().__init__(dim, num_components, log_Z, can_sample)

        self.device = device
        self.name = 'ring_gmm'
        self.dtype = dtype
        self.n_mixes = num_components
        # 1. Different mixture weights for all 9 Gaussians
        # We create linearly increasing weights [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # which will be normalized in the component_weights property.
        weights = torch.arange(1, num_components + 1, dtype=torch.float32)
        self.register_buffer("cat_probs", weights)

        # 2. Arrange 9 Gaussians on a ring
        # Calculate the angles for 9 equally spaced points along a circle
        theta = torch.linspace(0, 2 * math.pi, num_components + 1)[:-1]

        locs = torch.zeros((num_components, dim))

        # To ensure the projection looks like a ring in higher dimensions,
        # we repeat the circle's (cos, sin) coordinates across pairs of dimensions.
        for d in range(0, dim, 2):
            locs[:, d] = radius * torch.cos(theta)
            if d + 1 < dim:
                locs[:, d + 1] = radius * torch.sin(theta)

        self.register_buffer("locs", locs)

        # 3. Scales (standard deviations)
        # All Gaussians share the same scale parameter here, but you can modify this.
        scales_tensor = torch.ones((num_components, dim)) * scale
        self.register_buffer("scales", scales_tensor)

        # Plotting bounds (radius + some padding based on scale)
        self._plot_bounds = (-1.5 *radius - 3 * scale, 1.5 *radius + 3 * scale)
        self.is_many_body = False

        self.to(self.device)

    @property
    def component_weights(self) -> torch.Tensor:
        """
        Returns normalized component weights so they sum to 1.
        """
        return self.cat_probs / self.cat_probs.sum()

    @torch.no_grad()
    def pi_hat(self, samples: torch.Tensor, soft: bool = False) -> torch.Tensor:
        x = samples.to(self.device, self.dtype)
        K = self.n_mixes

        expanded = x.unsqueeze(-2)  # [N, 1, D]
        logps = self.distribution.component_distribution.log_prob(expanded)  # [N, K]

        if soft:
            log_pi_true = torch.log(self.cat_probs + 1e-10)  # [K]
            log_joint = log_pi_true + logps
            responsibilities = torch.softmax(log_joint, dim=1)  # [N, K]
            pi_hat = responsibilities.mean(dim=0)  # [K]
            return pi_hat
        else:
            idx = torch.argmax(logps, dim=1)  # [N]
            pi_hat = F.one_hot(idx, num_classes=K).to(self.dtype).mean(dim=0)  # [K]

        return pi_hat

    @torch.no_grad()
    def entropy(self, samples: torch.Tensor) -> torch.Tensor:
        """
        ハード割当て後の経験的モード分布 q のエントロピー（底 K で正規化）
        H(q) = -sum_k q_k log_K q_k
        """
        assert samples is not None, "samples を渡してください"
        x = samples.to(self.device, self.dtype)
        # 各成分での log_prob（[N, K]）
        # Independent の log_prob は [N, K] を返すには x を [N, 1, D] に拡張して components にブロードキャスト
        expanded = x.unsqueeze(-2)                    # [N, 1, D]
        logps = self.distribution.component_distribution.log_prob(expanded)  # [N, K]
        idx = torch.argmax(logps, dim=1)             # [N]
        K = self.n_mixes
        q = F.one_hot(idx, num_classes=K).to(self.dtype).mean(dim=0)  # [K]

        # 0*log 0 = 0 になるように clamp
        q_safe = q.clamp_min(torch.finfo(self.dtype).tiny)
        entropy = -(q * (q_safe.log() / math.log(K))).sum()
        return entropy

    @torch.no_grad()
    def log_Z_error(self, log_weights: torch.Tensor) -> torch.Tensor:
        """
        log_weights: [N]
        """
        logZ = torch.logsumexp(log_weights, dim=0) - math.log(log_weights.shape[0])
        logZ_lw = log_weights.mean(dim=0)
        true_logZ = 0.0
        abs_errors = torch.abs(logZ - true_logZ)
        abs_errors_lw = torch.abs(logZ_lw - true_logZ)
        return {'logZ_error': abs_errors, 'logZ_error_lw': abs_errors_lw}

    @torch.no_grad()
    def tvd(self, samples: torch.Tensor, soft: bool=False) -> torch.Tensor:
        pi_hat = self.pi_hat(samples, soft=soft)
        mixture_probs = self.cat_probs.to(self.device, self.dtype)  # [K]
        return 0.5 * torch.abs(pi_hat - mixture_probs).sum()

    def log_details(self, results: dict, prefix: str = "", visualize=True) -> dict:
        # まずスーパークラスのログ関数を呼ぶ
        logs = super().log_details(results, prefix=prefix, visualize=visualize)
        # サンプルを用意して経験的分布を得る
        # samples = self.sample(sample_shape=(1000,))
        pred_samples = results["data_1"]
        # TVD を計算
        tvd_val_soft = self.tvd(pred_samples, soft=True)
        tvd_val_hard = self.tvd(pred_samples, soft=False)
        mode_tvd = self.compute_mode_tvd(pred_samples)
        if "log_RN[h|u]" in results:
            logs.update(self.log_Z_error(results["log_RN[h|u]"]))

        # dict に追加
        logs[f"loss/{prefix}tvd_soft"] = tvd_val_soft.item()
        logs[f"loss/{prefix}tvd"] = tvd_val_hard.item()
        logs[f"loss/{prefix}mode_tvd"] = mode_tvd
        logs.update(self.visualize_2d_and_weights(pred_samples, show=True))
        return logs

    def log_prob_per_component(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of x under each component, including the mixture weight.
        """
        x = x.to(self.device)

        log_weights = torch.log(self.component_weights)  # [K]

        comp_dist = torch.distributions.Independent(
            torch.distributions.Normal(self.locs, self.scales),
            reinterpreted_batch_ndims=1
        )

        comp_log_probs = comp_dist.log_prob(x.unsqueeze(1))

        return log_weights + comp_log_probs

    @property
    def distribution(self):
        """
        Constructs the full PyTorch MixtureSameFamily distribution.
        """
        mix = torch.distributions.Categorical(self.component_weights.to(self.device))
        com = torch.distributions.Independent(
            torch.distributions.Normal(self.locs.to(self.device), self.scales.to(self.device)),
            reinterpreted_batch_ndims=1
        )
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    # ---------------------------------------------------------
    # Visualization and Sampling utilities inherited/adapted
    # ---------------------------------------------------------

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the explicit categorical distribution, then add Gaussian noise.
        """
        with torch.no_grad():
            indices = torch.distributions.Categorical(self.component_weights).sample((num_samples,))
            indices = indices.to(self.locs.device)

            locs_sampled = self.locs[indices]
            scales_sampled = self.scales[indices]

            eps = torch.randn_like(locs_sampled)

            return locs_sampled + eps * scales_sampled

    def get_distribution_2D(self, dim1: int, dim2: int):
        locs = self.locs[:, (dim1, dim2)]
        scales = self.scales[:, (dim1, dim2)]

        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.Independent(
            torch.distributions.Normal(locs, scales),
            reinterpreted_batch_ndims=1
        )

        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)


    def visualize_2d_and_weights(self, model_samples: torch.Tensor, show=True):
        """
        Visualizes the 2D projection (dim 0 and 1) with contours and compares
        true mixture weights against empirical sample weights.
        """
        # import matplotlib.pyplot as plt
        # import numpy as np
        import wandb

        model_samples_cpu = model_samples.detach().cpu()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=50)

        # ---------------------------------------------------------------------
        # Subfigure 1: 2D Contours (Dim 0 & 1) and Model Samples
        # ---------------------------------------------------------------------
        ax1 = axes[0]
        projected_target_density = self.get_distribution_2D(0, 1)

        # Helper to compute log_prob for the contour plot grid
        def log_prob_2d(x):
            return projected_target_density.log_prob(x.to(self.device)).cpu()

        # Plot target contours
        plot_density(ax1, log_prob_2d, plot_bounds=self._plot_bounds)

        # Scatter the generated samples
        ax1.scatter(model_samples_cpu[:, 0], model_samples_cpu[:, 1],
                    c='r', alpha=0.5, marker='x', label='Model Samples')

        ax1.set_title("Dimensions 0 and 1: Contours vs Samples")
        ax1.set_xlabel("Dimension 0")
        ax1.set_ylabel("Dimension 1")
        ax1.legend()

        # ---------------------------------------------------------------------
        # Subfigure 2: True vs. Empirical Mixture Weights
        # ---------------------------------------------------------------------
        ax2 = axes[1]

        with torch.no_grad():
            # Assign samples to components based on highest posterior probability
            comp_log_probs = self.log_prob_per_component(model_samples.to(self.device))
            assignments = torch.argmax(comp_log_probs, dim=1)

            # Calculate empirical frequencies
            counts = torch.bincount(assignments, minlength=self.num_components).float()
            empirical_weights = (counts / model_samples.shape[0]).cpu().numpy()

            # True weights
            true_weights = self.component_weights.cpu().numpy()

        indices = np.arange(self.num_components)
        width = 0.35

        # Grouped bar chart
        ax2.bar(indices - width / 2, true_weights, width, label='True Weights', color='tab:blue', alpha=0.8)
        ax2.bar(indices + width / 2, empirical_weights, width, label='Empirical Weights', color='tab:orange', alpha=0.8)

        ax2.set_title("Mixture Weights: True vs Empirical")
        ax2.set_xlabel("Component Index")
        ax2.set_ylabel("Weight / Frequency")
        ax2.set_xticks(indices)
        ax2.legend()

        plt.tight_layout()

        wb = {"figures/2d_and_weights": [wandb.Image(fig)]}

        if show:
            plt.show()
        else:
            plt.close(fig)

        return wb

    def visualize(self, model_samples: torch.Tensor, eta_star: float = 0., model_log_prob_fn=None, show=True) -> dict:
        """
        Visualizes 1D and 2D marginals of the provided samples against the target density.
        """

        return self.visualize_2d_and_weights(model_samples, show=show)


