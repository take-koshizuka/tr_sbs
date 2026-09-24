# Standard library imports
import math
from typing import Optional, Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import wandb
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
)
# Local application imports
from .base_energy import BaseEnergy
from .plot_density import (
    plot_contours_2D,
    plot_density,
    plot_marginal_pair,
)

class GMM(BaseEnergy):
    def __init__(
        self,
        dim: int = 2,
        num_components: int = 40,
        loc_scaling: float = 40.0,
        scale_scaling: float = 1.0,
        max_weight_ratio: float = 3.0,
        seed: int = 0,
        sample_bounds=None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        uniform_mixing_weights: bool = False,
        softplus: bool = False,
    ) -> None:
        super().__init__(dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self.seed = seed
        self.n_mixes = num_components

        # RNG
        seed = 0
        g = torch.Generator(device="cpu").manual_seed(seed)

        # === 位置パラメータ ===
        # mean ~ Uniform([-1, 1]) * loc_scaling
        mean = (torch.rand((num_components, dim), generator=g, device="cpu") * 2.0 - 1.0)
        mean = mean * loc_scaling

        # === 混合重み ===
        if uniform_mixing_weights:
            # 一様分布は logits=0 ベクトルで表現
            cat_logits = torch.zeros(num_components)
        else:
            # 確率比制約付き logits を生成
            cat_logits = self._generate_logits_with_prob_ratio_constraint(
                rng=g,
                num_categories=num_components,
                max_prob_ratio=max_weight_ratio,
            ).to(self.device, self.dtype)

        cat = Categorical(logits=cat_logits)
        self.cat_probs = cat.probs.to(self.device, self.dtype)
        
        # === スケール（対角成分のみ） ===
        scale_diag = torch.ones((num_components, dim)) * scale_scaling
        if softplus:
            scale_diag = F.softplus(scale_diag)

        # === 保存 ===
        self.locs: torch.Tensor = mean.to(self.device, self.dtype)
        self.scale_trils: torch.Tensor = torch.diag_embed(scale_diag).to(self.device, self.dtype)
        self.scale_trils_2D: torch.Tensor = torch.diag_embed(scale_diag[:, (0, 1)]).to(self.device, self.dtype)

        # === 分布定義 ===
        comp = MultivariateNormal(self.locs, scale_tril=self.scale_trils, validate_args=False)
        self.distribution = MixtureSameFamily(cat, comp, validate_args=False)

        # 2D可視化用（最初の2次元）
        locs2 = self.locs[:, (0, 1)]
        comp2 = MultivariateNormal(locs2, scale_tril=self.scale_trils_2D, validate_args=False)
        self.distribution_2D = MixtureSameFamily(cat, comp2, validate_args=False)

        self._plot_bound = min(loc_scaling * 3.5, 50)
        self.sample_bounds = sample_bounds

    # === JAXスタイルの logit 制約付き生成関数 ===
    @staticmethod
    def _generate_logits_with_prob_ratio_constraint(
        rng: torch.Generator,
        num_categories: int,
        max_prob_ratio: float = 3.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        JAX の generate_logits_with_prob_ratio_constraint に対応。
        logits の最大/最小比が log(max_prob_ratio) になるようにスケーリング。
        """
        logit_range = torch.log(torch.tensor(max_prob_ratio, device=device, dtype=dtype))
        raw_logits = torch.randn((num_categories,), generator=rng, device=device, dtype=dtype)
        sorted_logits, _ = torch.sort(raw_logits)
        minv, maxv = torch.min(sorted_logits), torch.max(sorted_logits)
        eps = torch.tensor(1e-8, device=device, dtype=dtype)
        logits = (sorted_logits - torch.mean(sorted_logits)) / (maxv - minv + eps)
        logits = logits * logit_range
        return logits
    
    # 勾配用のlog_probはno_gradを外す
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

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return the analytic score of the Gaussian mixture.

        Computing the score through autograd retains a large graph at every
        PIS/DIS integration step.  The closed form is both exact and much more
        memory efficient: component scores are weighted by posterior mixture
        responsibilities.
        """
        if x.dim() not in (1, 2):
            raise ValueError(
                f"Expected x with shape [D] or [B, D], got {tuple(x.shape)}"
            )

        input_device = x.device
        input_dtype = x.dtype
        batched = x.dim() == 2
        x_work = x.to(self.device, self.dtype)
        if not batched:
            x_work = x_work.unsqueeze(0)

        components = self.distribution.component_distribution
        component_log_probs = components.log_prob(x_work.unsqueeze(-2))
        log_mixture_probs = self.distribution.mixture_distribution.logits
        responsibilities = torch.softmax(
            component_log_probs + log_mixture_probs,
            dim=-1,
        )

        centered = components.loc.unsqueeze(0) - x_work.unsqueeze(-2)
        component_scores = torch.cholesky_solve(
            centered.unsqueeze(-1),
            components.scale_tril.unsqueeze(0),
            upper=False,
        ).squeeze(-1)
        score = (responsibilities.unsqueeze(-1) * component_scores).sum(dim=-2)

        if not batched:
            score = score.squeeze(0)
        return score.to(device=input_device, dtype=input_dtype)
    
    def eval_2D(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D投影分布の log_prob 計算
        x: [N, D]
        returns: [N]
        """
        batched = (x.dim() == 2)
        if not batched:
            x = x.unsqueeze(0)
        x = x.to(self.device, self.dtype)
        logp = self.distribution_2D.log_prob(x[:, (0, 1)])
        if not batched:
            logp = logp.squeeze(0)
        return -logp
    
    @torch.no_grad()
    def sample(self, sample_shape: tuple = (), generator: torch.Generator | None = None) -> torch.Tensor:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return self.distribution.sample(sample_shape=sample_shape).to(self.device, self.dtype)

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
    def pi_hat(self, samples: torch.Tensor, soft: bool=False) -> torch.Tensor:
        x = samples.to(self.device, self.dtype)
        K = self.n_mixes

        expanded = x.unsqueeze(-2)  # [N, 1, D]
        logps = self.distribution.component_distribution.log_prob(expanded)  # [N, K]
        
        if soft:
            log_pi_true = torch.log(self.cat_probs + 1e-10) # [K]
            log_joint = log_pi_true + logps
            responsibilities = torch.softmax(log_joint, dim=1) # [N, K]
            pi_hat = responsibilities.mean(dim=0) # [K]
            return pi_hat
        else:
            idx = torch.argmax(logps, dim=1)  # [N]
            pi_hat = F.one_hot(idx, num_classes=K).to(self.dtype).mean(dim=0)  # [K]
            
        return pi_hat
    
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

    @torch.no_grad()
    def visualize(self, samples: torch.Tensor, title=None) -> dict:
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot()

        if samples is not None:
            try:
                plot_marginal_pair(samples[:, :2].cpu().numpy(), ax, bounds=(-self._plot_bound, self._plot_bound))
            except NameError:
                ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)

        if self.dim == 2:
            try:
                plot_contours_2D(lambda x: -self.eval(x), ax, bound=self._plot_bound, levels=50)
            except NameError:
                ax.set_xlim(-self._plot_bound, self._plot_bound)
                ax.set_ylim(-self._plot_bound, self._plot_bound)
        else:
            try:
                plot_contours_2D(lambda x: -self.eval_2D(x), ax, bound=self._plot_bound, levels=50)
            except NameError:
                ax.set_xlim(-self._plot_bound, self._plot_bound)
                ax.set_ylim(-self._plot_bound, self._plot_bound)

        # ここでは wandb を使わない簡易版
        if title is not None:
            wb = {f"figures/vis_{title}": [wandb.Image(fig)]}
        else:
            wb = {"figures/vis": [wandb.Image(fig)]}

        plt.close(fig)
        return wb
    
    def log_details(self, results: dict, prefix: str="", visualize=True) -> dict:
        # まずスーパークラスのログ関数を呼ぶ
        logs = super().log_details(results, prefix=prefix, visualize=visualize)
        # サンプルを用意して経験的分布を得る
        # samples = self.sample(sample_shape=(1000,))
        pred_samples = results["data_1"]
        # TVD を計算
        tvd_val_soft = self.tvd(pred_samples, soft=True)
        tvd_val_hard = self.tvd(pred_samples, soft=False)
        
        if "log_RN[h|u]" in results:
            logs.update(self.log_Z_error(results["log_RN[h|u]"]))
            
        # dict に追加
        logs[f"loss/{prefix}tvd_soft"] = tvd_val_soft.item()
        logs[f"loss/{prefix}tvd"] = tvd_val_hard.item()
        return logs

    def __repr__(self):
        return f'GMM_k:{self.n_mixes}_d:{self.dim}'
