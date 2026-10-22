"""One Gaussian kernel used by sampling, path densities, and conditional KL."""

from dataclasses import dataclass
import math

import torch

from .reference import positions, coefficient
from .sde import _is_graph_data
from tr_asbs.utils.geometry import subtract_mean_batch


def with_positions(template, value):
    if _is_graph_data(template):
        result = template.clone()
        result.pos = value
        return result
    return value


def project(data, value):
    if _is_graph_data(data):
        return subtract_mean_batch(value, data.batch)
    if value.ndim != 3:
        raise ValueError(
            "Mean-free flat kernels require (batch, particles, spatial_dim)"
        )
    return value - value.mean(-2, keepdim=True)


@dataclass
class GaussianTransition:
    mean: object
    variance: torch.Tensor
    mean_free: bool = False

    def __post_init__(self):
        self.variance = torch.as_tensor(self.variance).to(positions(self.mean))
        if not bool(torch.all(torch.isfinite(self.variance) & (self.variance > 0))):
            raise ValueError("Gaussian transition variance must be finite and positive")
        if self.mean_free:
            self.mean = with_positions(
                self.mean, project(self.mean, positions(self.mean))
            )

    def _sum(self, value):
        if _is_graph_data(self.mean):
            out = value.new_zeros(self.mean.num_graphs)
            return out.index_add(0, self.mean.batch, value.sum(-1))
        return value.reshape(value.shape[0], -1).sum(-1)

    def _dimensions(self):
        pos = positions(self.mean)
        if _is_graph_data(self.mean):
            counts = torch.bincount(self.mean.batch, minlength=self.mean.num_graphs).to(
                pos
            )
            return (counts - int(self.mean_free)) * pos.shape[-1]
        dim = pos[0].numel() - (pos.shape[-1] if self.mean_free else 0)
        return pos.new_full((pos.shape[0],), dim)

    def _variance_per_sample(self):
        return self.variance.reshape(-1).expand(self._dimensions().shape)

    def sample(self, noise=None):
        pos = positions(self.mean)
        noise = torch.randn_like(pos) if noise is None else noise.to(pos)
        if noise.shape != pos.shape:
            raise ValueError("Noise shape must match the state")
        if self.mean_free:
            noise = project(self.mean, noise)
        return with_positions(
            self.mean, pos + coefficient(self.variance.sqrt(), self.mean) * noise
        )

    def log_prob(self, value):
        delta = positions(value) - positions(self.mean)
        valid = None
        if self.mean_free:
            projected = project(self.mean, delta)
            # Covariance is singular in ambient coordinates: evaluate its density
            # on the COM-free subspace, never silently discard a COM displacement.
            valid = self._sum((delta - projected).square()) < 1e-10
            delta = projected
        quad = self._sum(delta.square() / coefficient(self.variance, self.mean))
        result = -0.5 * (
            quad
            + self._dimensions() * torch.log(2 * math.pi * self._variance_per_sample())
        )
        return torch.where(valid, result, -torch.inf) if valid is not None else result

    def entropy(self):
        return (
            0.5
            * self._dimensions()
            * torch.log(2 * math.pi * math.e * self._variance_per_sample())
        )

    def kl(self, other):
        if self.mean_free != other.mean_free:
            raise ValueError("KL requires the same covariance support")
        delta = positions(self.mean) - positions(other.mean)
        mean_term = self._sum(delta.square() / coefficient(other.variance, self.mean))
        ratio = self._variance_per_sample() / other._variance_per_sample()
        return 0.5 * (mean_term + self._dimensions() * (ratio - 1 - torch.log(ratio)))
