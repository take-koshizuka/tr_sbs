# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_mean_batch(positions: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean position per graph in a batch.

    Args:
        positions (torch.Tensor): Tensor of shape (N, D) where N is the number of nodes
            and D is the dimensionality of the positions.
        batch_index (torch.Tensor): Tensor of shape (N,) indicating the graph index for
            each node.

    Returns:
        torch.Tensor: Tensor of shape (K, D) where K is the number of graphs and D is
            the dimensionality of the positions.
    """
    num_graphs, dim = batch_index[-1] + 1, positions.shape[1]
    means = torch.zeros(num_graphs, dim, dtype=positions.dtype, device=positions.device)
    means.index_reduce_(0, batch_index, positions, reduce="mean", include_self=False)
    return means


def subtract_mean_batch(
    positions: torch.Tensor, batch_index: torch.Tensor
) -> torch.Tensor:
    """
    Subtract the mean position per graph from the positions.

    Args:
        positions (torch.Tensor): Tensor of shape (N, D) where N is the number of nodes
            and D is the dimensionality of the positions.
        batch_index (torch.Tensor): Tensor of shape (N,) indicating the graph index for
            each node.

    Returns:
        torch.Tensor: Tensor of shape (N, D) with mean positions subtracted.
    """
    means = get_mean_batch(positions, batch_index)
    return positions - means[batch_index]


def is_mean_free_batch(
    positions: torch.Tensor, batch_index: torch.Tensor, atol: float = 1e-5
) -> bool:
    """
    Check if the positions are mean-free per graph in a batch.

    Args:
        positions (torch.Tensor): Tensor of shape (N, D) where N is the number of nodes
            and D is the dimensionality of the positions.
        batch_index (torch.Tensor): Tensor of shape (N,) indicating the graph index for
            each node.
        atol (float): Absolute tolerance for checking mean-free condition.

    Returns:
        bool: True if positions are mean-free, False otherwise.
    """
    means = get_mean_batch(positions, batch_index)
    return torch.allclose(means, torch.zeros_like(means), atol=atol)


def is_mean_free_data(data, atol: float = 1e-5) -> bool:
    """
    Check if the positions in an AtomicData object are mean-free per graph.

    Args:
        data (AtomicData): AtomicData object containing positions and batch indices.
        atol (float): Absolute tolerance for checking mean-free condition.

    Returns:
        bool: True if positions are mean-free, False otherwise.
    """
    return is_mean_free_batch(data.pos, data.batch, atol=atol)


def subtract_com_vector(samples, n_particles, n_dimensions):
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples
