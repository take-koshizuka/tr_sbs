"""Scalar noise functions independent of the integration grid (horizon T=1)."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class NoiseSchedule(nn.Module, ABC):
    @abstractmethod
    def diffusion(self, t):
        """g(t)."""

    @abstractmethod
    def variance(self, t):
        """V(t) = integral from 0 to t of g(s)^2 ds."""

    def interval_variance(self, s, t):
        return self.variance(t) - self.variance(s)

    def inverse_variance(self, v):
        lo, hi = torch.zeros_like(v), torch.ones_like(v)
        for _ in range(56):
            mid = (lo + hi) / 2
            left = self.variance(mid) < v
            lo, hi = torch.where(left, mid, lo), torch.where(left, hi, mid)
        return (lo + hi) / 2


class AnalyticNoiseSchedule(NoiseSchedule):
    """Constant, linear-amplitude, or decreasing geometric g, with exact V.

    Geometric matches VESDE when g_max=sigma_max*sqrt(2*log(r)) and
    g_min=sigma_min*sqrt(2*log(r)). Equal endpoints are a constant schedule.
    """

    def __init__(self, kind="constant", g_min=1.0, g_max=1.0, total_variance=None):
        super().__init__()
        if kind not in ("constant", "linear", "geometric"):
            raise ValueError(f"Unknown noise schedule: {kind}")
        if min(g_min, g_max) <= 0:
            raise ValueError("Noise amplitudes must be positive")
        self.kind = kind
        self.register_buffer("g_min", torch.tensor(g_min, dtype=torch.double))
        self.register_buffer("g_max", torch.tensor(g_max, dtype=torch.double))
        scale = 1.0
        if total_variance is not None:
            if total_variance <= 0:
                raise ValueError("total_variance must be positive")
            scale = (total_variance / self._variance(self.g_min.new_tensor(1))).sqrt()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.double))

    def _time(self, t):
        return torch.as_tensor(t, device=self.g_min.device, dtype=torch.double)

    def _variance(self, t):
        if self.kind == "constant":
            return t * self.g_max.square()
        if self.kind == "linear":
            d = self.g_min - self.g_max
            return (
                self.g_max.square() * t
                + self.g_max * d * t.square()
                + d.square() * t**3 / 3
            )
        rate = torch.log(self.g_max / self.g_min)
        if abs(rate.item()) < 1e-12:
            return self.g_max.square() * t
        return self.g_max.square() * (-torch.expm1(-2 * rate * t)) / (2 * rate)

    def diffusion(self, t):
        time = self._time(t)
        if self.kind == "constant":
            g = self.g_max + time * 0
        elif self.kind == "linear":
            g = self.g_max + (self.g_min - self.g_max) * time
        else:
            g = self.g_max * torch.exp(-torch.log(self.g_max / self.g_min) * time)
        return (self.scale * g).to(t) if isinstance(t, torch.Tensor) else self.scale * g

    def variance(self, t):
        return self.scale.square() * self._variance(self._time(t))


class LegacyNoiseSchedule(NoiseSchedule):
    """Expose VE/EDM/discrete PIS schedules without changing their coefficients.

    PIS num_steps describes noise knots, independently of solver steps. Optional
    normalization multiplies the amplitude; it does not change the time shape.
    """

    def __init__(self, process, total_variance=None):
        super().__init__()
        self.process = process
        scale = 1.0
        if total_variance is not None:
            if total_variance <= 0:
                raise ValueError("total_variance must be positive")
            scale = (total_variance / process.total_variance).sqrt()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.double))

    def diffusion(self, t):
        return self.process.diffusion(t) * self.scale.to(t)

    def variance(self, t):
        return self.process.diffsquare_integral(t) * self.scale.square()
