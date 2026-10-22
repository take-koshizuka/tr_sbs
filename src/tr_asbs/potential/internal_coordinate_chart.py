"""Smooth R^60 chart for ALDP's standardized internal coordinates.

Bond lengths stay positive, bond angles in (0, pi), and each dihedral occupies
one fundamental interval centered on the frozen transform's circular mean.
The derivative is one near the calibration mean; extra Jacobians are explicit.
Gaussian reference kernels act on unconstrained y, without post-step wrapping.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


class InternalCoordinateChart(nn.Module):
    def __init__(self, complete_transform, dihedral_offsets=None):
        super().__init__()
        ic = complete_transform.ic_transform
        means = torch.zeros(60, device=complete_transform.mean_b1.device)
        scales = torch.ones_like(means)
        kinds = torch.zeros(60, device=means.device, dtype=torch.long)
        # kind: 0=bond, 1=angle, 2=dihedral.
        means[:3] = torch.stack(
            (
                complete_transform.mean_b1,
                complete_transform.mean_b2,
                complete_transform.mean_angle,
            )
        )
        scales[:3] = torch.stack(
            (
                complete_transform.std_b1,
                complete_transform.std_b2,
                complete_transform.std_angle,
            )
        )
        kinds[2] = 1
        for kind, indices, mean, std in (
            (0, ic.bond_indices, ic.mean_bonds, ic.std_bonds),
            (1, ic.angle_indices, ic.mean_angles, ic.std_angles),
            (2, ic.dih_indices, ic.mean_dih, ic.std_dih),
        ):
            for j, original in enumerate(indices):
                index = (
                    int(torch.nonzero(complete_transform.permute[9:] == original)[0])
                    + 3
                )
                means[index], scales[index], kinds[index] = mean[j], std[j], kind
        self._initialize(means, scales, kinds)
        if dihedral_offsets is not None:
            self.set_dihedral_offsets(dihedral_offsets)

    @classmethod
    def from_state(cls, state, dihedral_offsets=None):
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result._initialize(state['means'].clone(), state['scales'].clone(),
                           state['kinds'].clone())
        if dihedral_offsets is not None:
            result.set_dihedral_offsets(dihedral_offsets)
        else:
            result.offsets.copy_(state.get('offsets', torch.zeros_like(result.means)))
        return result

    def set_dihedral_offsets(self, offsets):
        offsets = torch.as_tensor(offsets, dtype=self.means.dtype, device=self.means.device)
        if offsets.shape != self.means.shape or not torch.isfinite(offsets).all():
            raise ValueError('Chart offsets must be a finite native-dimensional vector')
        if torch.any(offsets[self.kinds != 2] != 0):
            raise ValueError('Only dihedral chart centers may be shifted')
        self.offsets.copy_(offsets)

    def _initialize(self, means, scales, kinds):
        if not torch.all(torch.isfinite(scales) & (scales > 0)):
            raise ValueError("IC chart calibration scales must be finite and positive")
        if not torch.all(means[kinds == 0] > 0):
            raise ValueError("Bond calibration means must be positive")
        angle_means = means[kinds == 1]
        if not torch.all((angle_means > 0) & (angle_means < math.pi)):
            raise ValueError("Angle calibration means must be inside (0, pi)")
        self.register_buffer("means", means)
        self.register_buffer("scales", scales)
        self.register_buffer("kinds", kinds)
        self.register_buffer("offsets", torch.zeros_like(means))

    def canonicalize(self, z):
        """Choose FAB's representatives; this is a coordinate change, not a path step."""
        result = z.clone()
        indices = self.kinds == 2
        delta = z[..., indices] * self.scales[indices]
        result[..., indices] = (torch.remainder(delta + math.pi, 2 * math.pi)
                               - math.pi) / self.scales[indices]
        return result

    def forward(self, y):
        if y.shape[-1] != self.means.numel():
            raise ValueError("IC chart dimension mismatch")
        z, log_jac = torch.empty_like(y), torch.empty_like(y)
        for kind in (0, 1, 2):
            indices = self.kinds == kind
            v, mean, std = y[..., indices], self.means[indices], self.scales[indices]
            if kind == 0:
                offset = inverse_softplus(mean / std)
                z[..., indices] = F.softplus(v + offset) - mean / std
                log_jac[..., indices] = F.logsigmoid(v + offset)
            elif kind == 1:
                p = mean / math.pi
                scale = std / (math.pi * p * (1 - p))
                a = v * scale + torch.logit(p)
                z[..., indices] = (math.pi * torch.sigmoid(a) - mean) / std
                log_jac[..., indices] = (
                    torch.log(math.pi * scale / std)
                    + F.logsigmoid(a)
                    + F.logsigmoid(-a)
                )
            else:
                a = v * std / math.pi
                z[..., indices] = (self.offsets[indices] + math.pi * torch.tanh(a)) / std
                log_jac[..., indices] = (
                    2 * math.log(2) + F.logsigmoid(2 * a) + F.logsigmoid(-2 * a)
                )
        return z, log_jac.sum(-1)

    def inverse(self, z, canonical=False):
        y = torch.empty_like(z)
        for kind in (0, 1, 2):
            indices = self.kinds == kind
            v, mean, std = z[..., indices], self.means[indices], self.scales[indices]
            if kind == 0:
                ratio = v + mean / std
                if not torch.all(ratio > 0):
                    raise ValueError("IC inverse chart received a nonpositive bond")
                y[..., indices] = inverse_softplus(ratio) - inverse_softplus(mean / std)
            elif kind == 1:
                p = mean / math.pi
                prob = (v * std + mean) / math.pi
                if not torch.all((prob > 0) & (prob < 1)):
                    raise ValueError(
                        "IC inverse chart received an angle outside (0, pi)"
                    )
                scale = std / (math.pi * p * (1 - p))
                y[..., indices] = (torch.logit(prob) - torch.logit(p)) / scale
            else:
                delta = v * std - self.offsets[indices]
                if canonical:
                    delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
                ratio = delta / math.pi
                if not torch.all(ratio.abs() < 1):
                    raise ValueError(
                        "IC inverse chart received a dihedral outside its fundamental interval"
                    )
                y[..., indices] = torch.atanh(ratio) * math.pi / std
        return y
