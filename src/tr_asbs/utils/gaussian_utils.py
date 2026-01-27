import torch

def gaussian_log_prob(samples, mean, std, dim: int):
    """
    Computes the log probability of samples under a multivariate Gaussian
    with diagonal covariance and batch-dependent std.

    Args:
        samples: Tensor of shape (..., D)
        mean:    Tensor of shape (..., D)
        std:     Tensor of shape (..., D)

    Returns:
        log_probs: Tensor of shape (...)
    """
    if isinstance(std, float) or isinstance(std, int):
        std = torch.full_like(mean, float(std))
    elif isinstance(std, torch.Tensor) and std.dim() == 0:
        std = std.expand_as(mean)
    
    # 次元 D
    D = samples.shape[-1]

    var = std ** 2
    diff = samples - mean
    squared_mahalanobis = torch.sum((diff ** 2) / var, dim=-1)

    log_det = torch.sum(torch.log(std), dim=-1)  # 各次元で和を取る
    # log det Σ = sum_i log σ_i, ここでは diagonal covariance を仮定

    log_probs = -0.5 * (
        squared_mahalanobis
        + D * torch.log(torch.tensor(2 * torch.pi, device=samples.device, dtype=samples.dtype))
        + 2 * log_det
    )

    return log_probs

def gaussian_score_diag(x, mu, var):
    """
    return ∇_x log N(x | mu, diag(var))
    Args:
        x:   (..., D)
        mu:  (..., D) or (D)
        var: (..., D) or (D) or scalar or (B,)
    Returns:
        score: (..., D)
    """
    # スカラーの場合: expand_as(x)
    if var.dim() == 0:
        var = var.expand_as(x)
    # バッチごとに1値 (B,)
    elif var.shape == x.shape[:-1]:   # (B,)
        var = var.unsqueeze(-1).expand_as(x)
    # muとxに合わせる
    elif var.shape != x.shape:
        var = torch.broadcast_to(var, x.shape)
    if mu.shape != x.shape:
        mu = torch.broadcast_to(mu, x.shape)

    return -(x - mu) / var.clamp_min(1e-12)