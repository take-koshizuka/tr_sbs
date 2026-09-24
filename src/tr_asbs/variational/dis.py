# Portions adapted from variational_sampling_methods commit 32de621.
# Copyright (c) 2023 Denis Blessing. Licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for attribution.
#
# Time-Reversed Diffusion Sampler: https://openreview.net/forum?id=oYIjw37pTP

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from .schedules import CosineDiffusionSchedule, DiffusionSchedule
from .types import RolloutResult, TargetDensity, VariationalMethod


class DISMethod(VariationalMethod):
    """Batched PyTorch implementation of the Time-Reversed Diffusion Sampler.

    DIS starts from an isotropic Gaussian and controls the time reversal of a
    variance-preserving Ornstein-Uhlenbeck diffusion. Samples are vectorized
    across the batch; the only Python loop is over the time steps in the
    faithful ``T, ..., 1`` simulation order.
    """

    def __init__(
        self,
        control: nn.Module,
        target: TargetDensity,
        num_steps: int = 128,
        max_diffusion: float | None = None,
        *,
        init_std: float = 1.0,
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
        if not math.isfinite(init_std) or init_std <= 0.0:
            raise ValueError(f"init_std must be finite and positive, got {init_std}")
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
        self.init_std = float(init_std)
        self.noise_clip = float(noise_clip)
        self.dt = 1.0 / self.num_steps
        self.sqrt_dt = math.sqrt(self.dt)
        self.noise_schedule = noise_schedule

    @property
    def schedule_values(self) -> torch.Tensor:
        """OU rate coefficients indexed by ascending step ``1, ..., T``."""

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
        initial_noise: torch.Tensor | None,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if generator is not None and (noise is not None or initial_noise is not None):
            raise ValueError("generator cannot be combined with fixed noise")

        prepared_noise = None
        if noise is not None:
            expected_shape = (self.num_steps, batch_size, self.dim)
            if noise.shape != expected_shape:
                raise ValueError(
                    "noise must have shape [num_steps, batch_size, dim] in "
                    f"simulation order T..1; expected {expected_shape}, "
                    f"got {tuple(noise.shape)}"
                )
            prepared_noise = noise.to(device=device, dtype=dtype)

        prepared_initial_noise = None
        if initial_noise is not None:
            expected_shape = (batch_size, self.dim)
            if initial_noise.shape != expected_shape:
                raise ValueError(
                    "initial_noise must have shape [batch_size, dim]; "
                    f"expected {expected_shape}, got {tuple(initial_noise.shape)}"
                )
            prepared_initial_noise = initial_noise.to(device=device, dtype=dtype)

        return prepared_noise, prepared_initial_noise

    def _score_hint(
        self,
        x: torch.Tensor,
        step: int,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(getattr(self.control, "uses_score_hint", True)):
            return torch.zeros_like(x)
        # sigma_t * grad((1 - step / T) log pi(x)
        #                + (step / T) log N(x; 0, init_std**2 I))
        # from the reference DIS implementation.
        with torch.enable_grad():
            target_score = self.target.score(x.detach())
        if target_score.shape != x.shape:
            raise ValueError(
                "target.score(x) must have the same shape as x, "
                f"got {tuple(target_score.shape)} and {tuple(x.shape)}"
            )
        target_coefficient = 1.0 - step / self.num_steps
        initial_coefficient = step / self.num_steps
        initial_score = -x.detach() / self.init_std**2
        return (
            sigma
            * (target_coefficient * target_score + initial_coefficient * initial_score)
        ).detach()

    def _target_log_prob(
        self,
        x: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # The terminal target term intentionally remains pathwise so it
        # supplies training gradients through the simulated trajectory.
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

    def _initial_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        variance = self.init_std**2
        log_normalizer = self.dim * math.log(2.0 * math.pi * variance)
        return -0.5 * (x.square().sum(dim=-1) / variance + log_normalizer)

    def rollout(
        self,
        batch_size: int,
        *,
        noise: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        return_trajectory: bool = False,
        generator: torch.Generator | None = None,
    ) -> RolloutResult:
        """Simulate a prior-to-target DIS batch.

        Args:
            batch_size: Number of fully vectorized sample paths.
            noise: Optional raw standard-normal integration draws with shape
                ``[T, B, D]`` in simulation order (steps ``T`` through ``1``).
                Values are clipped internally.
            initial_noise: Optional standard-normal draws with shape ``[B, D]``
                used to sample the initial Gaussian. Values are clipped
                internally.
            return_trajectory: Store states after each integration step when
                true.
            generator: Optional generator used for all random draws.
        """

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        device, dtype = self._model_device_dtype()
        if not dtype.is_floating_point:
            raise TypeError(f"control dtype must be floating point, got {dtype}")
        fixed_noise, fixed_initial_noise = self._prepare_noise(
            noise,
            initial_noise,
            batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        if fixed_initial_noise is None:
            initial_draw = torch.randn(
                (batch_size, self.dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            initial_draw = fixed_initial_noise
        initial_draw = initial_draw.clamp(
            min=-self.noise_clip,
            max=self.noise_clip,
        )
        initial_x = self.init_std * initial_draw
        x = initial_x

        running_cost = torch.zeros(batch_size, device=device, dtype=dtype)
        stochastic_cost = torch.zeros(batch_size, device=device, dtype=dtype)
        trajectory: list[torch.Tensor] | None = [] if return_trajectory else None
        simulation_betas = self.noise_schedule.simulation_values.to(
            device=device,
            dtype=dtype,
        )

        for index, step in enumerate(range(self.num_steps, 0, -1)):
            beta = simulation_betas[index]
            sigma = torch.sqrt(2.0 * beta) * self.init_std
            score_hint = self._score_hint(x, step, sigma)
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

            x = (
                x
                + (sigma * control + beta * x) * self.dt
                + sigma * step_noise * self.sqrt_dt
            )
            running_cost = (
                running_cost
                + (-self.dim * beta + 0.5 * control.square().sum(dim=-1)) * self.dt
            )
            stochastic_cost = (
                stochastic_cost + (control * step_noise).sum(dim=-1) * self.sqrt_dt
            )
            if trajectory is not None:
                trajectory.append(x)

        terminal_cost = self._initial_log_prob(initial_x) - self._target_log_prob(
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
