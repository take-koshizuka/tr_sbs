# Portions adapted from variational_sampling_methods commit 32de621.
# Copyright (c) 2023 Denis Blessing. Licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for attribution.
#
# Path Integral Sampler: https://arxiv.org/abs/2111.15141

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from .schedules import CosineDiffusionSchedule, DiffusionSchedule
from .types import RolloutResult, TargetDensity, VariationalMethod


class PISMethod(VariationalMethod):
    """Batched PyTorch implementation of Path Integral Sampling.

    Samples are vectorized across the batch.  The only Python loop is over the
    time steps in the faithful ``T, ..., 1`` simulation order.
    """

    def __init__(
        self,
        control: nn.Module,
        target: TargetDensity,
        num_steps: int = 128,
        max_diffusion: float | None = None,
        *,
        sigma_min: float = 0.01,
        s: float = 0.008,
        power: float = 2.0,
        pow: float | None = None,
        schedule: DiffusionSchedule | torch.Tensor | None = None,
        noise_clip: float = 4.0,
    ) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if noise_clip <= 0.0:
            raise ValueError(f"noise_clip must be positive, got {noise_clip}")
        if not hasattr(target, "dim"):
            raise TypeError("target must expose a dim attribute")
        if not hasattr(target, "log_prob") or not hasattr(target, "score"):
            raise TypeError("target must implement log_prob(x) and score(x)")

        self.dim = int(target.dim)
        if self.dim <= 0:
            raise ValueError(f"target.dim must be positive, got {self.dim}")
        control_dim = getattr(control, "dim", self.dim)
        if int(control_dim) != self.dim:
            raise ValueError(
                f"control dim {control_dim} does not match target dim {self.dim}"
            )

        if schedule is None:
            if max_diffusion is None:
                raise ValueError(
                    "max_diffusion is required when schedule is not provided"
                )
            noise_schedule: DiffusionSchedule = CosineDiffusionSchedule(
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=max_diffusion,
                s=s,
                power=power,
                pow=pow,
            )
        elif isinstance(schedule, torch.Tensor):
            noise_schedule = DiffusionSchedule(schedule)
        elif isinstance(schedule, DiffusionSchedule):
            noise_schedule = schedule
        else:
            raise TypeError("schedule must be a DiffusionSchedule, Tensor, or None")
        if noise_schedule.num_steps != num_steps:
            raise ValueError(
                "Schedule length must equal num_steps, "
                f"got {noise_schedule.num_steps} and {num_steps}"
            )

        self.control = control
        self.target = target
        self.num_steps = int(num_steps)
        self.noise_clip = float(noise_clip)
        self.dt = 1.0 / self.num_steps
        self.sqrt_dt = math.sqrt(self.dt)
        self.noise_schedule = noise_schedule
        self.register_buffer(
            "reference_variance",
            noise_schedule.integrated_variance(self.dt).detach().clone(),
        )

    @property
    def schedule_values(self) -> torch.Tensor:
        """Diffusion coefficients indexed by ascending step ``1, ..., T``."""

        return self.noise_schedule.values

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.control.parameters(), None)
        if parameter is not None:
            return parameter.device, parameter.dtype
        buffer = next(self.control.buffers(), None)
        if buffer is not None:
            return buffer.device, buffer.dtype
        return self.schedule_values.device, self.schedule_values.dtype

    def _prepare_noise(
        self,
        noise: torch.Tensor | None,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor | None:
        if noise is None:
            return None
        expected_shape = (self.num_steps, batch_size, self.dim)
        if noise.shape != expected_shape:
            raise ValueError(
                "noise must have shape [num_steps, batch_size, dim] in "
                f"simulation order T..1; expected {expected_shape}, "
                f"got {tuple(noise.shape)}"
            )
        if generator is not None:
            raise ValueError("generator cannot be combined with fixed noise")
        return noise.to(device=device, dtype=dtype)

    def _target_score(self, x: torch.Tensor, step: int) -> torch.Tensor:
        if not bool(getattr(self.control, "uses_score_hint", True)):
            return torch.zeros_like(x)
        # grad((1 - step / T) * log pi(x)) from the reference PIS code.
        # Detaching the score input prevents analytic implementations from
        # retaining the path graph.  Enabling grad keeps autograd fallbacks
        # usable when rollout itself is evaluated under torch.no_grad().
        with torch.enable_grad():
            score = self.target.score(x.detach())
        if score.shape != x.shape:
            raise ValueError(
                "target.score(x) must have the same shape as x, "
                f"got {tuple(score.shape)} and {tuple(x.shape)}"
            )
        coefficient = 1.0 - step / self.num_steps
        return (score * coefficient).detach()

    def _target_log_prob(
        self,
        x: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # Do not detach: this terminal term supplies the pathwise training
        # gradient that is deliberately absent from the score hint.
        log_prob = self.target.log_prob(x)
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.as_tensor(
                log_prob,
                device=x.device,
                dtype=x.dtype,
            )
        if log_prob.numel() != batch_size:
            raise ValueError(
                "target.log_prob(x) must return one value per sample, "
                f"got shape {tuple(log_prob.shape)} for batch size {batch_size}"
            )
        return log_prob.reshape(batch_size).to(device=x.device, dtype=x.dtype)

    def _reference_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        variance = self.reference_variance.to(device=x.device, dtype=x.dtype)
        log_normalizer = self.dim * (math.log(2.0 * math.pi) + torch.log(variance))
        return -0.5 * (x.square().sum(dim=-1) / variance + log_normalizer)

    def rollout(
        self,
        batch_size: int,
        *,
        noise: torch.Tensor | None = None,
        return_trajectory: bool = False,
        generator: torch.Generator | None = None,
    ) -> RolloutResult:
        """Simulate a prior-to-target PIS batch.

        Args:
            batch_size: Number of fully vectorized sample paths.
            noise: Optional raw standard-normal draws with shape
                ``[T, B, D]`` in simulation order (steps ``T`` through ``1``).
                Values are clipped internally just like generated noise.
            return_trajectory: Store states after each step when true.
            generator: Optional generator used for random draws.
        """

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        device, dtype = self._model_device_dtype()
        if not dtype.is_floating_point:
            raise TypeError(f"control dtype must be floating point, got {dtype}")
        fixed_noise = self._prepare_noise(
            noise,
            batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        x = torch.zeros(batch_size, self.dim, device=device, dtype=dtype)
        running_cost = torch.zeros(batch_size, device=device, dtype=dtype)
        stochastic_cost = torch.zeros(
            batch_size,
            device=device,
            dtype=dtype,
        )
        trajectory: list[torch.Tensor] | None = [] if return_trajectory else None
        simulation_sigmas = self.noise_schedule.simulation_values.to(
            device=device,
            dtype=dtype,
        )

        for index, step in enumerate(range(self.num_steps, 0, -1)):
            sigma = simulation_sigmas[index]
            score_hint = self._target_score(x, step)
            # Keep the compiled control's input signature tensor-only and
            # shape-stable across steps; the value remains the raw step.
            step_tensor = x.new_tensor(float(step))
            control = self.control(step_tensor, x, score_hint)
            if control.shape != x.shape:
                raise ValueError(
                    "control(step, x, score_hint) must have shape [B, D], "
                    f"got {tuple(control.shape)} and expected {tuple(x.shape)}"
                )

            if fixed_noise is None:
                step_noise = torch.randn(
                    x.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            else:
                step_noise = fixed_noise[index]
            step_noise = step_noise.clamp(
                min=-self.noise_clip,
                max=self.noise_clip,
            )

            x = x + sigma * control * self.dt + sigma * step_noise * self.sqrt_dt
            running_cost = running_cost + 0.5 * control.square().sum(dim=-1) * self.dt
            stochastic_cost = (
                stochastic_cost + (control * step_noise).sum(dim=-1) * self.sqrt_dt
            )
            if trajectory is not None:
                trajectory.append(x)

        terminal_cost = self._reference_log_prob(x) - self._target_log_prob(
            x, batch_size
        )
        stored_trajectory = (
            torch.stack(cast(list[torch.Tensor], trajectory), dim=0)
            if trajectory is not None
            else None
        )
        return RolloutResult(
            samples=x,
            running_cost=running_cost,
            stochastic_cost=stochastic_cost,
            terminal_cost=terminal_cost,
            trajectory=stored_trajectory,
        )
