import math
from functools import partial
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack

import ot as pot
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm

def compute_distribution_distances(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute distances between two distributions without a time axis.

    Args:
        pred: [batch, dims] tensor of predictions
        true: [batch, dims] tensor of ground truth
    Returns:
        metrics: dictionary mapping metric name -> scalar value
    """
    if pred.ndim != 2 or true.ndim != 2:
        raise ValueError(f"Inputs must be [batch, dims], got {pred.shape}, {true.shape}")
    if pred.shape[1] != true.shape[1]:
        raise ValueError(f"Dimension mismatch: {pred.shape} vs {true.shape}")

    metrics = {}

    # Wasserstein distances
    metrics["W_1"] = compute_wasserstein1_exact(pred, true)
    metrics["W_2"] = compute_wasserstein2_exact(pred, true)
    metrics["SD"] = compute_sinkhorn_divergence_jax(pred, true, epsilon=1e-3)
    metrics["Sliced_W_1"] = sliced_wasserstein(pred, true, p=1, n_projections=100)
    metrics["Sliced_W_2"] = sliced_wasserstein(pred, true, p=2, n_projections=100)
    
    # MMD metrics
    metrics["Linear_MMD"] = linear_mmd2(pred, true).item()
    metrics["Poly_MMD"]   = poly_mmd2(pred, true, d=2, alpha=1.0, c=2.0).item()
    metrics["RBF_MMD"]    = mix_rbf_mmd2(pred, true, sigma_list=[0.01, 0.1, 1, 10, 100]).item()

    # Mean-based distances
    mean_dists = compute_mse(torch.mean(pred, dim=0), torch.mean(true, dim=0))
    metrics["Mean_MSE"], metrics["Mean_L2"], metrics["Mean_L1"] = mean_dists

    # Median-based distances
    median_dists = compute_mse(torch.median(pred, dim=0)[0], torch.median(true, dim=0)[0])
    metrics["Median_MSE"], metrics["Median_L2"], metrics["Median_L1"] = median_dists
            
    return metrics

################################################################################
# MMD
################################################################################
# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=1e-8))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY**2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0)
        )
    )
    return mmd2, var_est


################################################################################
# Mean/Median based distances
################################################################################
def compute_mse(pred, true):
    """Computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


################################################################################
# Equivariant Earth Mover's Distance
################################################################################
def compute_eot(pred, true, n_particles, spatial_dim):
    npart = n_particles
    ndim  = spatial_dim
    metrics = {}
    try:
        metrics["Eq-EMD2"] = eot(
            pred.reshape(-1, npart, ndim).cpu(),
            true.reshape(-1, npart, ndim).cpu(),
        )
    except Exception as e:
        raise ValueError(
            "Eq-EMD2 reshape failed. "
            "Ensure dims == n_particles * n_spatial_dim."
        ) from e
    return metrics

def eot(x0, x1):
    M = []
    for i in tqdm(range(len(x0))):
        reordered = []
        for j in range(len(x1)):
            x1_reordered = _assignment_reorder(x0[i], x1[j])
            reordered.append(x1_reordered)
        reordered = torch.stack(reordered)
        R, t = torch.vmap(_find_rigid_alignment)(x0[i][None].repeat(len(x1), 1, 1), reordered)
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    M = torch.stack(M).squeeze()
    return pot.emd2(M=M, a=torch.ones(len(x0)) / len(x0), b=torch.ones(len(x1)) / len(x1))

def _find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()

def _assignment_reorder(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    x1 = x1[col_ind]
    return x1


#################################################################################
# Sliced Wasserstein
#################################################################################
def sliced_wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    n_projections: int = 50,
    power: int = 2,
    projections: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    log: bool = False,
    **kwargs,
) -> float:
    """
    Compute the Sliced Wasserstein distance (POT: ot.sliced.sliced_wasserstein_distance)
    in the same interface/style as the provided EMD function.
    The return value is always a float (the distance).

    Args:
        x0, x1: Sample tensors. Shape is (n_samples, n_features), or flattened if higher-dimensional.
        n_projections: Number of 1D projections.
        power: The p in p-Wasserstein (1 or 2 recommended). Passed directly to POT.
        projections: Explicit projection directions of shape (n_projections, n_features). If omitted, random.
        seed: Random seed.
        a, b: Weights for each sample (length matches the number of samples in x0 and x1). Defaults to uniform.
        log: Even if True, this function returns only the float distance (POT’s log is discarded).

    Returns:
        float: The Sliced Wasserstein distance.
    """
    assert power in (1, 2), "power must be 1 or 2"

    # Flatten to (n, d)
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    # Ensure weights (uniform by default)
    if a is None:
        a_np = pot.unif(x0.shape[0])
    else:
        a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    if b is None:
        b_np = pot.unif(x1.shape[0])
    else:
        b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b

    # Projections (optional)
    proj_np = None
    if projections is not None:
        proj_np = projections.detach().cpu().numpy() if isinstance(projections, torch.Tensor) else projections

    # To numpy (float64)
    Xs = x0.detach().cpu().numpy()
    Xt = x1.detach().cpu().numpy()

    # Compute Sliced Wasserstein with POT
    if log:
        val, _ = pot.sliced.sliced_wasserstein_distance(
            Xs, Xt, a=a_np, b=b_np,
            n_projections=n_projections,
            p=power,
            projections=proj_np,
            seed=seed,
            log=True
        )
    else:
        val = pot.sliced.sliced_wasserstein_distance(
            Xs, Xt, a=a_np, b=b_np,
            n_projections=n_projections,
            p=power,
            projections=proj_np,
            seed=seed,
            log=False
        )

    # POT already returns the p-Wasserstein distance (W1 for p=1, W2 for p=2)
    # Always cast to float
    return float(val)

################################################################################
# Wasserstein distance
################################################################################
import math
import torch
import numpy as np
import ot as pot  # POT: Python Optimal Transport


def _prepare_ot_inputs(
    x0: torch.Tensor,
    x1: torch.Tensor,
    power: int = 2,
):
    """
    Prepare uniform weights and a cost matrix for OT computation.

    power=1 → use Euclidean distance
    power=2 → use squared Euclidean distance
    """
    assert power in (1, 2)

    # Flatten in case tensors have additional dimensions
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    # Uniform weights (sum to 1)
    a = pot.unif(x0.shape[0])
    b = pot.unif(x1.shape[0])

    # Compute pairwise distances
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M ** 2

    # POT expects numpy arrays
    M_np = M.detach().cpu().numpy()
    return a, b, M_np


def compute_wasserstein1_exact(
    x0: torch.Tensor,
    x1: torch.Tensor,
    num_iter_max: int = int(1e7),
) -> float:
    """
    Compute the exact Wasserstein-1 distance using POT's emd2.
    """
    a, b, M = _prepare_ot_inputs(x0, x1, power=1)
    cost = pot.emd2(a, b, M, numItermax=num_iter_max)
    return float(cost)


def compute_wasserstein2_exact(
    x0: torch.Tensor,
    x1: torch.Tensor,
    num_iter_max: int = int(1e7),
) -> float:
    """
    Compute the exact Wasserstein-2 distance using POT's emd2.

    The optimal transport cost is computed on squared distances, then
    the square root is taken to obtain the W2 distance.
    """
    a, b, M = _prepare_ot_inputs(x0, x1, power=2)
    cost_sq = pot.emd2(a, b, M, numItermax=num_iter_max)
    return float(math.sqrt(cost_sq))

#################################################################################
# Sinkhorn Divergence with JAX OTT
#################################################################################
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence

def torch_to_jax(x: torch.Tensor):
    x_cpu = x.detach().contiguous().cpu()
    try:
        return jax.dlpack.from_dlpack(to_dlpack(x_cpu))
    except TypeError:
        return jnp.asarray(x_cpu.numpy())

def _get_divergence(out):
    if hasattr(out, "divergence"):
        return out.divergence
    if isinstance(out, tuple):
        return out[0]
    return out

def compute_sinkhorn_divergence_jax(
    x0: torch.Tensor,
    x1: torch.Tensor,
    epsilon: float = 1e-3,
) -> float:
    """
    Sinkhorn divergence using JAX OTT library.
    Returns a float (the divergence).

    Args:
        x0, x1: Sample tensors of shape (n_samples, n_features) or flattened if higher.
        epsilon: Regularization parameter.

    Returns:
        float: Sinkhorn divergence.
    """
    # Convert to JAX
    x0_jax = torch_to_jax(x0)
    x1_jax = torch_to_jax(x1)

    geom = pointcloud.PointCloud(x0_jax, x1_jax, epsilon=epsilon)
    out = sinkhorn_divergence.sinkhorn_divergence(geom, x=geom.x, y=geom.y)
    sd = _get_divergence(out)
    return float(sd)