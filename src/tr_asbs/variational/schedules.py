# Portions adapted from variational_sampling_methods commit 32de621.
# Copyright (c) 2023 Denis Blessing. Licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for attribution.

from __future__ import annotations

import math

import torch
from torch import nn


def _resolve_power(power: float, pow: float | None) -> float:
    if pow is None:
        return float(power)
    if power != 2.0 and not math.isclose(float(power), float(pow)):
        raise ValueError("Specify only one schedule exponent via power or pow")
    return float(pow)


def cosine_diffusion_schedule(
    num_steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    s: float = 0.008,
    power: float = 2.0,
    *,
    pow: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute the PIS/DIS cosine schedule values for steps ``1, ..., T``.

    This preserves the factor of ``0.5`` used by the reference PIS
    implementation.  Consequently, ``sigma_max`` names its configuration
    parameter rather than the exact value attained by the returned schedule.
    """

    power = _resolve_power(power, pow)
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if sigma_min <= 0.0:
        raise ValueError(f"sigma_min must be positive, got {sigma_min}")
    if sigma_max < sigma_min:
        raise ValueError(
            "sigma_max must be greater than or equal to sigma_min, "
            f"got sigma_min={sigma_min} and sigma_max={sigma_max}"
        )
    if s < 0.0:
        raise ValueError(f"s must be non-negative, got {s}")
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating point, got {dtype}")

    steps = torch.arange(
        1,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )
    t = steps / float(num_steps)
    offset = 1.0 + float(s)
    angle = 0.5 * math.pi * (offset - t) / offset
    return 0.5 * (float(sigma_max) - float(sigma_min)) * torch.cos(angle).pow(
        float(power)
    ) + 0.5 * float(sigma_min)


class DiffusionSchedule(nn.Module):
    """A validated, precomputed diffusion schedule.

    ``values[i]`` is the coefficient for the one-indexed step ``i + 1``.
    PIS consumes these values in reverse order.
    """

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(
                "Schedule values must be a non-empty rank-one tensor, "
                f"got shape {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError("Schedule values must have a floating-point dtype")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Schedule values must all be finite")
        if not bool((values > 0).all()):
            raise ValueError("Schedule values must all be positive")

        self.register_buffer("values", values.detach().clone())

    @property
    def num_steps(self) -> int:
        return int(self.values.numel())

    @property
    def simulation_values(self) -> torch.Tensor:
        """Values in the order used for prior-to-target simulation."""

        return self.values.flip(0)

    def integrated_variance(self, dt: float | None = None) -> torch.Tensor:
        """Return ``sum_t sigma_t**2 * dt``."""

        step_size = 1.0 / self.num_steps if dt is None else float(dt)
        if step_size <= 0.0:
            raise ValueError(f"dt must be positive, got {step_size}")
        return self.values.square().sum() * step_size

    def forward(self, step: torch.Tensor | int) -> torch.Tensor:
        """Look up one-indexed schedule steps."""

        index = torch.as_tensor(step, device=self.values.device)
        if index.is_floating_point():
            if not bool((index == index.round()).all()):
                raise ValueError("Schedule steps must be integer valued")
        index = index.to(dtype=torch.long) - 1
        if not bool(((0 <= index) & (index < self.num_steps)).all()):
            raise IndexError(f"Schedule steps must be in [1, {self.num_steps}]")
        return self.values[index]


class CosineDiffusionSchedule(DiffusionSchedule):
    """Precomputed cosine schedule shared by PIS and DIS."""

    def __init__(
        self,
        num_steps: int,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        s: float = 0.008,
        power: float = 2.0,
        *,
        pow: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        power = _resolve_power(power, pow)
        values = cosine_diffusion_schedule(
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s=s,
            power=power,
            device=device,
            dtype=dtype,
        )
        super().__init__(values)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.s = float(s)
        self.power = float(power)
        self.pow = self.power


# Noise-schedule terminology is retained as a compatibility aid for the JAX
# reference implementation and its configuration names.
CosineNoiseSchedule = CosineDiffusionSchedule
cosine_noise_schedule = cosine_diffusion_schedule
