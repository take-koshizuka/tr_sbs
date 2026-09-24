import torch
import numpy as np
import math

def stabilize_log_weights(log_weights):
    """
    Stabilize log weights by subtracting the maximum log weight.

    Args:
        log_weights: Tensor of shape (n_samples,)
    Returns:
        stabilized_log_weights: Tensor of shape (n_samples,)
    """
    max_log_weight = torch.max(log_weights)
    stabilized_log_weights = log_weights - max_log_weight
    return stabilized_log_weights

def compute_reverse_ess(log_weights):
    # Compute the importance weights in a numerically stable way
    is_weights = torch.exp(log_weights)

    # Compute the sums needed for ESS
    sum_is_weights = torch.sum(is_weights)
    sum_is_weights_squared = torch.sum(is_weights ** 2)

    # Calculate the effective sample size (ESS)
    ess = (sum_is_weights ** 2) / (len(is_weights) * sum_is_weights_squared)
    return ess

def compute_raw_path_ess(log_weights):
    """Normalized ESS of untempered path weights, robust to constant shifts."""
    log_weights = log_weights.reshape(-1)
    if log_weights.numel() == 0 or not torch.isfinite(log_weights).all():
        return 0.0
    return compute_reverse_ess(stabilize_log_weights(log_weights)).item()

def normalize_log_weights(log_weights, dim=0):
    """
    Scale log weights for numerical stability.

    Args:
        log_weights: Tensor of shape (n_samples,)
    Returns:

        Scaled log weights: Tensor of shape (n_samples,)
    """
    
    logZ = torch.logsumexp(log_weights, dim=dim) - math.log(log_weights.shape[0])
    scaled_log_weights = log_weights - logZ
    return scaled_log_weights

def compute_log_weights_metrics(log_weights, stabilize=True, normalize=True):
    """
    Compute metrics from log weights.

    Args:
        log_weights: Tensor of shape (n_samples,)
    Returns:
        A dictionary containing:
            - 'max_log_weight': Maximum log weight
    """
    if stabilize:
        log_weights = stabilize_log_weights(log_weights)
    if normalize:
        log_weights = normalize_log_weights(log_weights)    
    
    ln_z_iw = torch.logsumexp(log_weights, dim=0) - np.log(log_weights.shape[0])
    mean = torch.mean(log_weights)
    log_var = torch.var(log_weights, unbiased=True)
    var = torch.var(torch.exp(log_weights), unbiased=True)
    ess = compute_reverse_ess(log_weights)
    metrics = {
        'log_mean_weight': ln_z_iw.item(),
        'mean': mean.item(),
        'log_var': log_var.item(),
        'var': var.item(),
        'ess': ess.item(),
    }
    return metrics
