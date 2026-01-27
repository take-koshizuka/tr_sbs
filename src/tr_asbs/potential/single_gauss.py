from typing import Tuple, Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
from torch.distributions import MultivariateNormal

import seaborn as sns
import wandb # Assuming wandb is used for logging, as in the original code.

from numpy.typing import ArrayLike

from .base_energy import BaseEnergy
from .plot_density import plot_density, plot_contours_2D, plot_marginal_pair

def gaussian_axis_limits(mu, Sigma, p=0.997, pad=0.05):
    """
    mu: (2,) tensor, Sigma: (2,2) tensor
    p: 見せたい確率質量（例 0.95, 0.99, 0.997）
    pad: 余白率（5% など）
    """
    device, dtype = Sigma.device, Sigma.dtype
    c = Chi2(df=torch.tensor(2.0, device=device, dtype=dtype)).icdf(
            torch.tensor(p, device=device, dtype=dtype))
    sx = torch.sqrt(c * Sigma[0,0])
    sy = torch.sqrt(c * Sigma[1,1])
    mx, my = mu[0], mu[1]
    xlim = (mx - sx, mx + sx)
    ylim = (my - sy, my + sy)
    # padding
    wx = (xlim[1] - xlim[0]) * pad
    wy = (ylim[1] - ylim[0]) * pad
    return (xlim[0]-wx, xlim[1]+wx), (ylim[0]-wy, ylim[1]+wy)


def _as_torch(x, device, dtype):
    """任意入力をdevice/dtypeを統一したtorch.Tensorへ（コピー最小化）"""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

@torch.no_grad()
def spd_sqrt_and_invsqrt(S: torch.Tensor, eps: float = 0.0):
    """
    返り値: S^{1/2}, S^{-1/2}（固有値分解）
    S: (d,d) SPD（必要なら微小jitterを自分で足すか eps>0 を指定）
    """
    S = _sym(S)
    if eps > 0.0:
        S = S + eps * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    w, U = torch.linalg.eigh(S)  # w:昇順, U:直交
    if torch.any(w <= 0):
        raise ValueError("Sigma must be SPD (positive definite).")
    sw = torch.sqrt(w)
    invsw = 1.0 / sw
    S12   = (U * sw)    @ U.transpose(-1, -2)
    S_12  = (U * invsw) @ U.transpose(-1, -2)
    return _sym(S12), _sym(S_12)

@torch.no_grad()
def C_sigma_torch(S0: torch.Tensor, S1: torch.Tensor, sigma_star: torch.Tensor) -> torch.Tensor:
    """
    Static Gaussian-OT matrix C_σ（Mallasto などで使われる形）。
    すべてtorch版。
    """
    S0_12, S0_m12 = spd_sqrt_and_invsqrt(S0)
    d = S0.shape[0]
    I = torch.eye(d, device=S0.device, dtype=S0.dtype)
    # M は対称正値（数値誤差を抑えるため sym を挟む）
    M = _sym(4.0 * (S0_12 @ S1 @ S0_12) + (sigma_star**4) * I)
    w, U = torch.linalg.eigh(M)
    # 0未満はクリップ（理論上≥0だが数値誤差対策）
    sqrtw = torch.sqrt(torch.clamp(w, min=0.0))
    D = (U * sqrtw) @ U.transpose(-1, -2)
    C = 0.5 * (S0_12 @ D @ S0_m12 - (sigma_star**2) * I)
    return _sym(C)

class GaussianSB:
    def __init__(self, m0, S0, m1, S1, g_callable, kappa, device=None):
        """
        m0, S0, m1, S1 : torch.Tensor（任意device可）
        device : 明示しない場合は m0.device に合わせる
        """
        # デバイス/型を m0 に合わせて統一
        if not isinstance(m0, torch.Tensor):
            raise TypeError("m0 must be a torch.Tensor")
        
        self.cpu_device = torch.device("cpu")
        self.device = m0.device if device is None else torch.device(device)
        self.dtype  = m0.dtype

        self.m0 = _as_torch(m0, self.device, self.dtype).reshape(-1)
        self.m1 = _as_torch(m1, self.device, self.dtype).reshape(-1)
        self.S0 = _sym(_as_torch(S0, self.device, self.dtype))
        self.S1 = _sym(_as_torch(S1, self.device, self.dtype))

        d = self.S0.shape[0]
        assert self.S0.shape == (d, d) and self.S1.shape == (d, d)
        assert self.m0.shape[0] == d and self.m1.shape[0] == d

        # κ, r, rdot（g_callableはtorch対応想定）
        self.g = g_callable
        self.kappa = kappa
        self.kappa1 = kappa(1.0)
        self.r = lambda t: self.kappa(t) / self.kappa1
        self.rdot = lambda t: (self.g(t)**2) / self.kappa1
        
        # σ* と C_{σ*}
        self.sigma_star = torch.sqrt(self.kappa1)
        self.C = C_sigma_torch(self.S0, self.S1, self.sigma_star)
        self.I = torch.eye(d, device=self.device, dtype=self.dtype)
        
    @torch.no_grad()
    def log_prob(self, t, x, jitter: float = 0.0,
                *, force_cpu: bool = True, return_to_x_device: bool = True):
        """
        時刻 t における N(mean(t), cov(t)) の log p_t(x) を返す（内部計算はCPUに固定可能）。
        x: (..., D)   t: スカラー or (...,)
        force_cpu: True なら全演算をCPUで実行
        return_to_x_device: True なら結果だけ x と同じ device に戻す
        """
        # 入出力メタ
        x_in = _as_torch(x, device=None, dtype=None)  # 触らずにメタだけ知る
        x_dev, x_dtype = x_in.device, x_in.dtype

        comp_device = torch.device("cpu") if force_cpu else self.device
        comp_dtype  = x_dtype

        # 入力を計算デバイスへ
        t = torch.as_tensor(t, device=comp_device, dtype=comp_dtype)
        x = _as_torch(x, comp_device, comp_dtype)

        # 左パディング helper
        def _leftpad_to_ndim(tensor, ndim_target):
            pad = ndim_target - tensor.dim()
            if pad > 0:
                tensor = tensor.view((1,) * pad + tuple(tensor.shape))
            return tensor

        # CPU（任意デバイス）上で mean/cov を再計算
        mu_t  = self._mean_on(t, device=comp_device, dtype=comp_dtype)     # (..., D)
        Sig_t = self._cov_on (t, device=comp_device, dtype=comp_dtype)     # (..., D, D)

        mu_t  = _leftpad_to_ndim(mu_t,  x.dim())           # (..., D)
        Sig_t = _leftpad_to_ndim(Sig_t, x.dim() + 1)       # (..., D, D)

        if jitter and jitter > 0.0:
            I = self.I.to(device=comp_device, dtype=comp_dtype)
            Sig_t = Sig_t + jitter * I

        dist = torch.distributions.MultivariateNormal(loc=mu_t, covariance_matrix=Sig_t)
        logp = dist.log_prob(x)  # x を特徴次元として扱う

        if return_to_x_device and logp.device != x_dev:
            logp = logp.to(device=x_dev, dtype=x_dtype)
        return logp

        
    @torch.no_grad()
    def sample(self, n_samples: int, t: float,
            *, force_cpu: bool = True, return_to_device: torch.device | None = None):
        """
        t: スカラー or (B,)
        戻り値: (n_samples, d) or (B, n_samples, d)
        force_cpu: True ならサンプリングをCPUで実行
        return_to_device: 返却先デバイス（Noneなら self.device に戻す）
        """
        comp_device = torch.device("cpu") if force_cpu else self.device
        comp_dtype  = self.dtype
        out_device  = self.device if return_to_device is None else torch.device(return_to_device)

        t = torch.as_tensor(t, device=comp_device, dtype=comp_dtype)

        # 左パディング helper
        def _leftpad_to_ndim(tensor, ndim_target):
            pad = ndim_target - tensor.dim()
            if pad > 0:
                tensor = tensor.view((1,) * pad + tuple(tensor.shape))
            return tensor

        # mean/cov をCPU（任意デバイス）上で再計算
        mu_t  = self._mean_on(t, device=comp_device, dtype=comp_dtype)     # (..., d)
        Sig_t = self._cov_on (t, device=comp_device, dtype=comp_dtype)     # (..., d, d)

        if t.dim() == 0:
            samples = torch.distributions.MultivariateNormal(
                loc=mu_t, covariance_matrix=Sig_t
            ).sample((n_samples,))                                         # (n_samples, d)
        else:
            B = t.shape[0]
            # expand はビューなのでメモリ効率良くブロードキャストされる
            samples = torch.distributions.MultivariateNormal(
                loc=_leftpad_to_ndim(mu_t, 2).unsqueeze(1).expand(B, n_samples, -1),
                covariance_matrix=_leftpad_to_ndim(Sig_t, 3).unsqueeze(1).expand(B, n_samples, -1, -1)
            ).sample()                                                     # (B, n_samples, d)

        if samples.device != out_device:
            samples = samples.to(device=out_device, dtype=self.dtype)
        return samples

    # ---- time-marginal mean & covariance ----
    @torch.no_grad()
    def _mean_on(self, t, device, dtype):
        # t は任意形状 tensor / scalar
        t  = torch.as_tensor(t, device=device, dtype=dtype)
        rt = (self.kappa(t) / self.kappa1).to(device=device, dtype=dtype)  # (...,)
        m0 = self.m0.to(device=device, dtype=dtype)
        m1 = self.m1.to(device=device, dtype=dtype)
        return (1.0 - rt)[..., None] * m0 + rt[..., None] * m1             # (..., D)

    @torch.no_grad()
    def _cov_on(self, t, device, dtype):
        t      = torch.as_tensor(t, device=device, dtype=dtype)
        rt     = (self.kappa(t) / self.kappa1).to(device=device, dtype=dtype)[..., None, None]  # (...,1,1)
        omt    = (1.0 - self.kappa(t) / self.kappa1).to(device=device, dtype=dtype)[..., None, None]
        kappat = self.kappa(t).to(device=device, dtype=dtype)[..., None, None]

        S0 = self.S0.to(device=device, dtype=dtype)
        S1 = self.S1.to(device=device, dtype=dtype)
        C  = self.C.to(device=device,  dtype=dtype)
        I  = self.I.to(device=device,  dtype=dtype)

        Sigma_t = (omt**2) * S0 + (rt**2) * S1 \
                + (rt * omt) * (C + C.transpose(-1, -2)) \
                + kappat * (1.0 - rt.squeeze(-1).squeeze(-1))[..., None, None] * I
        # 対称化
        return 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))

    @torch.no_grad()
    def drift(self, t, x, *, force_cpu: bool = False, return_to_x_device: bool = True):
        """
        x: (B, *, D)
        t: スカラー or (...,) tensor
        force_cpu: True のとき、全計算を CPU 上で行う
        return_to_x_device: True なら結果だけ x の device に戻す
        """
        # 元の入力のメタ情報
        x_in = torch.as_tensor(x)
        x_dev, x_dtype = x_in.device, x_in.dtype

        # 計算デバイスと dtype を決める
        comp_device = self.cpu_device if force_cpu else self.device
        comp_dtype  = x_dtype

        # x / t を計算デバイスへ（計算は CPU で）
        x = torch.as_tensor(x, device=comp_device, dtype=comp_dtype)
        t = torch.as_tensor(t, device=comp_device, dtype=comp_dtype)

        # ---- 左パディング helper ----
        def _leftpad_to_ndim(tensor, ndim_target):
            pad = ndim_target - tensor.dim()
            if pad > 0:
                tensor = tensor.view((1,) * pad + tuple(tensor.shape))
            return tensor

        # ---- mean, cov（CPU上で再計算）----
        mu_t  = self._mean_on(t, device=comp_device, dtype=comp_dtype)              # (..., D)
        Sig_t = self._cov_on (t, device=comp_device, dtype=comp_dtype)              # (..., D, D)

        mu_t  = _leftpad_to_ndim(mu_t,  x.dim())                                    # (B,*,D) と演算可能
        Sig_t = _leftpad_to_ndim(Sig_t, x.dim() + 1)                                 # (B,*,D,D)

        # ---- r, rdot, g（CPU上で）----
        rt   = (self.kappa(t) / self.kappa1).to(device=comp_device, dtype=comp_dtype)
        rdot = (self.g(t)**2 / self.kappa1).to(device=comp_device, dtype=comp_dtype)
        gt   = self.g(t).to(device=comp_device, dtype=comp_dtype)

        rt   = _leftpad_to_ndim(rt,   x.dim()).unsqueeze(-1).unsqueeze(-1)          # (...,1,1)
        rdot = _leftpad_to_ndim(rdot, x.dim()).unsqueeze(-1).unsqueeze(-1)          # (...,1,1)
        gt   = _leftpad_to_ndim(gt,   x.dim()).unsqueeze(-1).unsqueeze(-1)          # (...,1,1)

        # ---- P, Q, S（CPU上で）----
        S0 = self.S0.to(device=comp_device, dtype=comp_dtype)
        S1 = self.S1.to(device=comp_device, dtype=comp_dtype)
        C  = self.C .to(device=comp_device, dtype=comp_dtype)
        I  = self.I .to(device=comp_device, dtype=comp_dtype)

        P = rdot * ( rt * S1 + (1.0 - rt) * C )
        Q = rdot * ( (1.0 - rt) * S0 + rt * C )
        S = P - Q.transpose(-1, -2) - (gt * gt) * rt * I                             # (B,*,D,D)

        # ---- v = Σ_t^{-1} (x - μ_t) ----
        dx = x - mu_t                                                                # (B,*,D)
        # ここは solve（逆行列は作らない）
        v = torch.linalg.solve(Sig_t, dx.unsqueeze(-1)).squeeze(-1)                  # (B,*,D)

        # ---- μ̇ と drift ----
        delta_m = (self.m1 - self.m0).to(device=comp_device, dtype=comp_dtype)       # (D,)
        mu_dot  = _leftpad_to_ndim((self.g(t)**2 / self.kappa1), x.dim())[..., None] * delta_m

        drift = (S.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1) + mu_dot         # (B,*,D)

        # 結果だけ元デバイスへ戻す（必要なら）
        if return_to_x_device and drift.device != x_dev:
            drift = drift.to(device=x_dev, dtype=x_dtype)

        # 形状は x と同じに
        return drift.reshape_as(x_in)

import torch
from torch.distributions import MultivariateNormal
from .base_energy import BaseEnergy
# 必要なimportは元のファイルに合わせてください（plot関連など）

"""
def covariance_with_difficulty_torch(
    dim: int, 
    difficulty: float, 
    generator: torch.Generator, 
    device: torch.device, 
    dtype: torch.dtype
) -> torch.Tensor:
    # 1. Eigenvalue spectrum (conditioning)
    max_condition = 1e3
    condition_number = 2.0 + difficulty * (max_condition - 2.0)
    
    # jnp.linspace -> torch.linspace
    eigs = torch.linspace(1.0, condition_number, dim, device=device, dtype=dtype)
    D = torch.diag(eigs)

    # 2. Random orthogonal rotation
    # jax.random.normal -> torch.randn
    A = torch.randn(dim, dim, generator=generator, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    Sigma_prime = Q @ D @ Q.T

    # 3. Toeplitz correlation via broadcasting
    rho = 0.1 + 0.8 * difficulty
    idx = torch.arange(dim, device=device)
    # idx[:, None] - idx[None, :] corresponds to unsqueeze in PyTorch
    dist_matrix = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
    # Pythonのべき乗演算子 ** はTorchテンソルに対しても機能します
    Corr = rho ** dist_matrix.to(dtype)

    # 4. Combine
    Sigma = Sigma_prime * Corr  # Element-wise multiplication
    # Symmetrize and add jitter
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * torch.eye(dim, device=device, dtype=dtype)

    return Sigma
"""

def covariance_with_difficulty_torch(
    dim: int,
    difficulty: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate a covariance matrix whose
    - maximum eigenvalue is fixed to 1
    - condition number is controlled by difficulty

    condition_number = lambda_max / lambda_min
    """
    # 1. Condition number control
    max_condition = 1e3
    condition_number = 2.0 + difficulty * (max_condition - 2.0)

    # 2. Eigenvalue spectrum
    # max eigenvalue = 1
    # min eigenvalue = 1 / condition_number
    eigs = torch.linspace(
        1.0 / condition_number,
        1.0,
        dim,
        device=device,
        dtype=dtype,
    )
    D = torch.diag(eigs)

    # 3. Random orthogonal rotation
    A = torch.randn(dim, dim, generator=generator, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    Sigma_prime = Q @ D @ Q.T

    # 4. Toeplitz-type correlation
    rho = 0.1 + 0.8 * difficulty
    idx = torch.arange(dim, device=device)
    dist_matrix = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
    Corr = rho ** dist_matrix.to(dtype)

    # 5. Combine scale and correlation
    Sigma = Sigma_prime * Corr

    # 6. Symmetrize and add jitter for numerical stability
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * torch.eye(
        dim, device=device, dtype=dtype
    )

    return Sigma

class SingleGaussian(BaseEnergy):
    def __init__(
        self,
        dim: int = 2,
        seed: int = 0,
        difficulty: float = 0.0,  # Added: 難易度パラメータ
        use_simple_cov: bool = True, # Added: 最初のコードのactiveな設定(0.5*I)を使うかどうか
        sample_bounds=None,
        can_sample: bool = True,
        log_Z: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        source = None,
        base_sde = None,
    ) -> None:
        super().__init__(dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed

        # PyTorchの乱数生成器
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # --- 移植箇所: 平均値の設定 ---
        # 元のコード: self.mu1 = jnp.ones([dim]) * 3
        mean = torch.ones((dim,), device=self.device, dtype=self.dtype) * 3.0
        
        # --- 移植箇所: 共分散行列の設定 ---
        if use_simple_cov:
            # 元のコードでactiveだった設定: self.Sigma1 = 0.5 * jnp.eye(dim)
            Sigma = 0.5 * torch.eye(dim, device=self.device, dtype=self.dtype)
        else:
            # 元のコードでコメントアウトされていた設定 (covariance_with_difficulty)
            Sigma = covariance_with_difficulty_torch(
                dim=dim, 
                difficulty=difficulty, 
                generator=g, 
                device=self.device, 
                dtype=self.dtype
            )

        # Cholesky分解（MultivariateNormalの入力用）
        # 数値安定性のためのガード
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2)) + 1e-8 * torch.eye(dim, device=self.device, dtype=self.dtype)
        scale_tril = torch.linalg.cholesky(Sigma)

        # 分布の定義
        self.distribution = MultivariateNormal(loc=mean, scale_tril=scale_tril)
        
        # Plot範囲の設定 (平均が3なので少し広めに)
        self._plot_bound = 10.0

        # --- 以下は元の実装と同様 ---
        if (source is not None) and (base_sde is not None):
            self.GaussianSB = GaussianSB(
                source.mean, source.cov, 
                mean, Sigma, 
                base_sde.diffusion, 
                kappa=base_sde.diffsquare_integral,
                device=self.device # GaussianSBのdevice引数を利用
            )
        
    def eval(self, x: torch.Tensor, normalize=True) -> torch.Tensor:
        batched = (x.dim() == 2)
        if not batched:
            x = x.unsqueeze(0)
        x = x.to(self.device, self.dtype)
        logp = self.distribution.log_prob(x)
        if not batched:
            logp = logp.squeeze(0)
        return -logp

    @torch.no_grad()
    def sample(self, sample_shape: tuple = (), generator: torch.Generator | None = None) -> torch.Tensor:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return self.distribution.sample(sample_shape=sample_shape).to(self.device, self.dtype)

    # ... (visualize, log_on_epoch_end などは既存のまま) ...

    @torch.no_grad()
    def visualize(self, samples: torch.Tensor | None = None, axes=None, show: bool = False, title=None, prefix: str = "") -> dict:
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()

        if samples is not None:
            try:
                plot_marginal_pair(samples[:, :2].cpu().numpy(), ax, bounds=(-self._plot_bound, self._plot_bound))
            except NameError:
                ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)

        if self.dim == 2:
            try:
                plot_contours_2D(self.eval, ax, bound=self._plot_bound, levels=50)
            except NameError:
                ax.set_xlim(-self._plot_bound, self._plot_bound)
                ax.set_ylim(-self._plot_bound, self._plot_bound)

        if show:
            plt.show()

        # ここでは wandb を使わない簡易版（wandb 未導入でも落ちないように）
        try:
            im = wandb.Image(fig)  # type: ignore
            key = f"figures/vis_{title}" if title is not None else "figures/vis"
            wb = {key: [im]}
        except Exception:
            wb = {}
            
        plt.close(fig)
        return wb
    
    @torch.no_grad()
    def visualize_gt(self, ts) -> dict:
        for i, t in enumerate(ts):
            samples = self.GaussianSB.sample(1000, t)  # (1000, dim)
            title = f"GT_Traj"
            log = { "generation_iter": i }
            log.update(self.visualize(samples=samples, show=False, title=title))
            wandb.log(log)

    def log_details(self, results: dict, prefix: str="", visualize=True) -> dict:
        """
        GMMのlog_detailsに倣った実装。
        W2距離の計算は除外し、Drift MSEと可視化のみを行います。
        """
        # 1. スーパークラスのログ関数を呼ぶ
        logs = super().log_details(results, prefix=prefix, visualize=visualize)

        # 2. サンプルの取得 (生成されたデータ)
        # "data_1" を優先し、なければ "state" を探します
        pred_samples = results.get("data_1", results.get("state", None))

        # --- Metric: Drift MSE ---
        # results に予測関数が含まれている場合のみ計算
        predict_drift_fn = results.get('predict_drift_fn', None)
        
        if hasattr(self, 'GaussianSB') and predict_drift_fn is not None:
            # 時間 t をサンプリング (評価用)
            n_eval = 1000
            ts = torch.rand((n_eval,), device=self.device, dtype=self.dtype) * (1.0 - 1e-6)
            
            # 【修正箇所】
            # 各 t に対して 1サンプルだけ生成し、形状を (n_eval, dim) にします。
            # n_samples=n_eval とすると (n_eval, n_eval, dim) になってしまいます。
            x_t = self.GaussianSB.sample(n_samples=1, t=ts).squeeze(1) 
            
            # x_t shape: (1000, D)
            # ts shape:  (1000,)
            
            # Drift計算: 正解 (Ground Truth) vs 予測 (Model)
            drift_gt = self.GaussianSB.drift(ts, x_t)
            drift_pred = predict_drift_fn(ts, x_t)
            
            # MSE
            mse = torch.mean(torch.sum((drift_pred - drift_gt) ** 2, dim=-1))
            logs[f"loss/{prefix}drift_mse"] = mse.item()

        # 3. Log Z Error の計算 (もし log_RN[h|u] があれば)
        if "log_RN[h|u]" in results:
            if hasattr(self, 'log_Z_error'):
                logs.update(self.log_Z_error(results["log_RN[h|u]"]))

        return logs
    
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
    
    def __repr__(self):
        return f'SingleGaussian_k:1_d:{self.dim}'

    
    
