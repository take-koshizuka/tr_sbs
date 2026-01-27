from typing import Callable, Optional, Tuple
from matplotlib.axes import Axes
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_density(
        ax: Axes,
        log_prob: Callable[[torch.Tensor], torch.Tensor],
        title: Optional[str] = None,
        plot_bounds: Tuple[float, float] = (-5., 5.),
        levels=100,
        resolution=100,
        log: bool = True,
) -> None:
    min_bound, max_bound = plot_bounds

    with torch.no_grad():
        xline = torch.linspace(min_bound, max_bound, resolution)
        yline = torch.linspace(min_bound, max_bound, resolution)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.hstack([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)])

        if torch.cuda.is_available():
            xyinput = xyinput.to(torch.device("cuda"))

        zgrid = log_prob(xyinput)
        zgrid = torch.reshape(zgrid, (resolution, resolution))

        if not log:
            zgrid = torch.exp(zgrid)

        # ax.contourf(xgrid, ygrid, zgrid.cpu().detach(), levels=levels)
        ax.contour(xgrid, ygrid, zgrid.cpu().detach(), levels=levels)

        if title is not None:
            ax.set_title(title)

# ---------- JAX版を踏襲した PyTorch 等高線プロッタ ----------
def plot_contours_2D(
    log_prob_func: Callable[[torch.Tensor], torch.Tensor],
    ax: Optional[plt.Axes] = None,
    bound: float = 3.0,
    levels: int = 20,
    n_points: int = 100,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    2次元の log確率関数の等高線を描画（JAX版をPyTorch移植）。
    - log_prob_func: [N,2] -> [N] の torch 関数（GMM40.log_prob をそのまま渡せる）
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # グリッド生成（JAX版は itertools.product、こちらは meshgrid で等価な点集合を作る）
    x1 = np.linspace(-bound, bound, n_points)
    x2 = np.linspace(-bound, bound, n_points)
    X1, X2 = np.meshgrid(x1, x2)                           # [n,n]
    pts = np.stack([X1.reshape(-1), X2.reshape(-1)], 1)    # [n*n, 2]

    with torch.no_grad():
        xy = torch.from_numpy(pts).to(device=device, dtype=dtype)  # [n*n,2]
        z  = log_prob_func(xy)                                     # [n*n]
        # 下限クリップ（JAX版と同じ）
        z  = torch.clamp(z, min=-1000.0)
        Z  = z.reshape(n_points, n_points).cpu().numpy()

    ax.contour(X1, X2, Z, levels=levels)
    # ax.contourf(X1, X2, np.exp(Z), levels=20, cmap='viridis')  # 元コードと同様に例だけコメント

def plot_marginal_pair(
    samples,  # torch.Tensor or np.ndarray
    ax: Axes,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5, 5),
    alpha: float = 0.5,
):
    with torch.no_grad():
        if isinstance(samples, np.ndarray):
            s = torch.from_numpy(samples)
        else:
            s = samples
        s = s.detach().to("cpu")
        s = torch.clamp(s, bounds[0], bounds[1])
        x = s[:, marginal_dims[0]].numpy()
        y = s[:, marginal_dims[1]].numpy()
        ax.plot(x, y, "o", alpha=alpha)
