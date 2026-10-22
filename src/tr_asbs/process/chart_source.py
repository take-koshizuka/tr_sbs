"""A common physical source represented in each frozen ALDP chart."""

import math
from pathlib import Path

import torch
from torch import nn

from tr_asbs.potential.internal_coordinate_chart import InternalCoordinateChart


def load_calibration_bundle(path):
    # These are campaign-owned tensors/primitive metadata, never pickled modules.
    bundle = torch.load(Path(path), map_location='cpu', weights_only=True)
    if bundle.get('schema') != 'aldp_chart_calibration_v1':
        raise ValueError('Unsupported ALDP calibration bundle')
    return bundle


class ChartMappedGaussianPrior(nn.Module):
    """Map a fixed base-chart Gaussian to another chart with its exact density.

    Gaussian reference *kernels* remain in the new y coordinates. The source
    itself is generally not Gaussian there; claiming stationary VP covariance
    for this source is deliberately unsupported.
    """

    def __init__(self, calibration_bundle_path, chart_key, dim=60, scale=1., device='cpu'):
        super().__init__()
        bundle = load_calibration_bundle(calibration_bundle_path)
        self.base_chart = InternalCoordinateChart.from_state(bundle['base_chart']).to(device)
        self.chart = InternalCoordinateChart.from_state(
            bundle['base_chart'], bundle['chart_offsets'][chart_key]).to(device)
        self.dim, self.scale = int(dim), float(scale)
        if self.dim != len(self.chart.means) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid chart source dimensions or scale')

    @property
    def cov(self):
        if torch.any(self.chart.offsets != self.base_chart.offsets):
            raise ValueError('Shifted-chart source is not a stationary Gaussian for VP')
        return self.scale**2 * torch.eye(self.dim, device=self.chart.means.device)

    def sample(self, size):
        if isinstance(size, int):
            size = (size,)
        y_base = torch.randn(*size, self.dim, device=self.chart.means.device,
                             dtype=self.chart.means.dtype) * self.scale
        z, _ = self.base_chart(y_base)
        return self.chart.inverse(self.base_chart.canonicalize(z), canonical=True)

    def log_prob(self, y):
        z, logjac = self.chart(y)
        y_base = self.base_chart.inverse(self.chart.canonicalize(z), canonical=True)
        _, base_logjac = self.base_chart(y_base)
        gaussian = -.5 * (y_base / self.scale).square().sum(-1)
        gaussian -= self.dim * math.log(self.scale * math.sqrt(2 * math.pi))
        return gaussian + logjac - base_logjac

    def score(self, y):
        with torch.enable_grad():
            value = y.detach().requires_grad_(True)
            return torch.autograd.grad(self.log_prob(value).sum(), value)[0]

    def replicate_sample(self, n_samples, duplicate_size):
        return self.sample(n_samples).repeat_interleave(duplicate_size, dim=0)
