# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Licensed under LICENSE in the root directory.
"""Gaussian-kernel integrators. Pure transitions also drive all path densities."""

import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from tqdm import tqdm

from .sde import ControlledFBSDE, ZeroDriftSDE
from .reference import LinearGaussianReference, positions, coefficient
from .transition import GaussianTransition, project, with_positions


def _get_device(data):
    return positions(data).device


def get_timesteps(t0, t1, dt=None, steps=None, rescale_t=None):
    if (steps is None) == (dt is None):
        raise ValueError("Exactly one of dt and steps should be defined.")
    start, end = float(t0), float(t1)
    if end <= start or (dt is not None and float(dt) <= 0):
        raise ValueError("Require t1 > t0 and positive dt")
    intervals = (
        int(steps) if steps is not None else math.ceil((end - start) / float(dt))
    )
    if intervals < 1:
        raise ValueError("At least one interval is required")
    fraction = torch.linspace(0, 1, intervals + 1)
    if rescale_t == "quad":
        fraction = fraction.sqrt()
    elif rescale_t == "cosine":
        phase = (
            (torch.arange(intervals, dtype=torch.float) / intervals + 0.008)
            / 1.008
            * torch.pi
            / 2
        )
        weights = phase.cos().pow(4)
        fraction = torch.cat((weights.new_zeros(1), weights.cumsum(0) / weights.sum()))
    elif rescale_t is not None:
        raise ValueError("Unknown timestep rescaling method")
    result = start + (end - start) * fraction
    result[0], result[-1] = start, end
    return result


class BaseIntegrator(nn.Module, ABC):
    def __init__(self, sde, timesteps=None, steps=128, grid="uniform_t"):
        super().__init__()
        self.sde = sde
        self.grid = grid
        if grid not in ("uniform_t", "uniform_tau"):
            raise ValueError("grid must be uniform_t or uniform_tau")
        if timesteps is None:
            timesteps = get_timesteps(0, 1, steps=steps)
        if grid == "uniform_tau":
            base = self.reference
            values = (
                torch.linspace(
                    0,
                    1,
                    len(timesteps),
                    dtype=torch.double,
                    device=base.total_variance.device,
                )
                * base.total_variance
            )
            timesteps = base.inverse_variance(values).float()
            timesteps[0], timesteps[-1] = 0, 1
        if not bool(torch.all(timesteps[1:] > timesteps[:-1])):
            raise ValueError("Timesteps must be strictly increasing")
        self.register_buffer("timesteps", timesteps)

    @property
    def reference(self):
        return getattr(self.sde, "base_sde", self.sde)

    def kernel_metadata(self):
        return {
            "schema": 2,
            "solver": type(self).__name__,
            "grid": self.grid,
            "reference": type(self.reference).__name__,
            "reference_parameters": {
                k: v.detach().cpu().tolist()
                for k, v in self.reference.state_dict().items()
            },
            "timesteps": self.timesteps.detach().cpu().tolist(),
            "mean_free": self.sde.is_mean_free,
            "control_units": "score",
        }

    @abstractmethod
    def transition(self, data, time, dt, backward=False):
        """Differentiable Gaussian transition; no random draws or in-place edits."""

    @torch.no_grad()
    def step(self, data, time, dt):
        kernel = self.transition(data, time, dt)
        return kernel.mean, kernel.sample()

    @torch.no_grad()
    def step_backward(self, data, time, dt):
        kernel = self.transition(data, time, dt, backward=True)
        return kernel.mean, kernel.sample()

    def _apply_centering(self, data):
        return with_positions(data, project(data, positions(data)))

    def _rollout(
        self,
        data,
        timesteps,
        center_every_step,
        zero_last_step_noise,
        return_trajectory,
        progress_bar,
        backward,
    ):
        if center_every_step is None:
            center_every_step = self.sde.is_mean_free
        if center_every_step and not self.sde.is_mean_free:
            raise ValueError("Centering changes full-space Gaussian support")
        times = (self.timesteps if timesteps is None else timesteps).to(
            _get_device(data)
        )
        if backward:
            times = times.flip(0)
        data = data.clone()
        trajectory = [data]
        iterator = range(len(times) - 1)
        if progress_bar:
            iterator = tqdm(
                iterator, desc="sdeint_bwd" if backward else "sdeint_fwd", leave=False
            )
        for i in iterator:
            dt = abs(times[i + 1] - times[i])
            kernel = self.transition(data, times[i], dt, backward)
            data = (
                kernel.mean
                if zero_last_step_noise and i == len(times) - 2
                else kernel.sample()
            )
            if center_every_step:
                data = self._apply_centering(data)
            if return_trajectory:
                trajectory.append(data)
        return trajectory if return_trajectory else data

    @torch.no_grad()
    def run(
        self,
        initial_data,
        timesteps=None,
        center_every_step=None,
        zero_last_step_noise=False,
        return_trajectory=True,
        progress_bar=True,
    ):
        return self._rollout(
            initial_data,
            timesteps,
            center_every_step,
            zero_last_step_noise,
            return_trajectory,
            progress_bar,
            False,
        )

    @torch.no_grad()
    def run_backward(
        self,
        final_data,
        timesteps=None,
        zero_last_step_noise=False,
        center_every_step=None,
        return_trajectory=True,
        progress_bar=True,
    ):
        return self._rollout(
            final_data,
            timesteps,
            center_every_step,
            zero_last_step_noise,
            return_trajectory,
            progress_bar,
            True,
        )

    @torch.no_grad()
    def run_with_logweights(
        self,
        integrator,
        prior_log_prob_fn,
        target_log_prob_fn,
        initial_data,
        timesteps=None,
        center_every_step=None,
        zero_last_step_noise=False,
        return_trajectory=False,
        progress_bar=False,
    ):
        if zero_last_step_noise:
            raise ValueError("Deterministic final steps have no Gaussian path density")
        if center_every_step is None:
            center_every_step = self.sde.is_mean_free
        if center_every_step and not self.sde.is_mean_free:
            raise ValueError(
                "Centering changes full-space Gaussian support; set center_every_step=False"
            )
        times = (self.timesteps if timesteps is None else timesteps).to(
            _get_device(initial_data)
        )
        data = initial_data
        log_ref_h = prior_log_prob_fn(data)
        log_ref_u = torch.zeros_like(log_ref_h)
        kl_ref_u = torch.zeros_like(log_ref_h)
        trajectory = [data]
        iterator = range(len(times) - 1)
        if progress_bar:
            iterator = tqdm(iterator, desc="sdeint", leave=False)
        for i in iterator:
            t, tn = times[i], times[i + 1]
            dt = tn - t
            proposal = self.transition(data, t, dt)
            model = (
                proposal if self is integrator else integrator.transition(data, t, dt)
            )
            nxt = proposal.sample()
            reverse = integrator.transition(nxt, tn, dt, backward=True)
            log_p = proposal.log_prob(nxt)
            log_ref_h = log_ref_h + log_p - reverse.log_prob(data)
            log_ref_u = log_ref_u + log_p - model.log_prob(nxt)
            kl_ref_u = kl_ref_u + proposal.kl(model)
            data = nxt
            if return_trajectory:
                trajectory.append(data)
        terminal_log_prob = target_log_prob_fn(data)
        return dict(
            data_1=data,
            trajectory=trajectory if return_trajectory else [initial_data, data],
            log_Ph_Pref=terminal_log_prob - log_ref_h,
            target_log_prob=terminal_log_prob,
            log_Pu_Pref=-log_ref_u,
            log_Pref_Pu_det=kl_ref_u,
            kernel_metadata=self.kernel_metadata(),
        )


class EulerMaruyamaIntegrator(BaseIntegrator):
    """Physical-time Euler with its own endpoint g^2 dt covariance."""

    def transition(self, data, time, dt, backward=False):
        drift = (
            self.sde.bwd_drift(time, data) if backward else self.sde.drift(time, data)
        )
        mean = with_positions(data, positions(data) + coefficient(dt, data) * drift)
        variance = self.sde.diffusion(time).square() * dt
        return GaussianTransition(mean, variance, self.sde.is_mean_free)


class ExponentialIntegrator(BaseIntegrator):
    """Legacy exact-noise Euler: endpoint drift dt, integrated Brownian noise.

    This historical name is retained for configs/checkpoints. It is not an
    exponential OU solver and does not integrate the control in diffusion time.
    """

    def transition(self, data, time, dt, backward=False):
        if not isinstance(self.reference, ZeroDriftSDE):
            raise ValueError(
                "Legacy ExponentialIntegrator requires a zero-drift reference"
            )
        if backward:
            drift = (
                self.sde.bwd_drift(time, data, interval_dt=dt)
                if isinstance(self.sde, ControlledFBSDE)
                else self.sde.bwd_drift(time, data)
            )
            variance = self.reference.diffsquare_integral(
                time
            ) - self.reference.diffsquare_integral(time - dt)
        else:
            drift = self.sde.drift(time, data)
            variance = self.reference.diffsquare_integral(
                time + dt
            ) - self.reference.diffsquare_integral(time)
        mean = with_positions(data, positions(data) + coefficient(dt, data) * drift)
        return GaussianTransition(mean, variance, self.sde.is_mean_free)


class DiffusionTimeEulerIntegrator(BaseIntegrator):
    """Zero-drift reference, mean x + Delta V * score-unit control."""

    def transition(self, data, time, dt, backward=False):
        if not isinstance(self.reference, ZeroDriftSDE):
            raise ValueError("Diffusion-time Euler requires a zero-drift reference")
        s, t = (time - dt, time) if backward else (time, time + dt)
        variance = self.reference.diffsquare_integral(
            t
        ) - self.reference.diffsquare_integral(s)
        name = "h_controller" if backward else "u_controller"
        model = getattr(self.sde, name, None)
        control = (
            torch.zeros_like(positions(data)) if model is None else model(time, data)
        )
        mean = with_positions(
            data, positions(data) + coefficient(variance, data) * control
        )
        return GaussianTransition(mean, variance, self.sde.is_mean_free)


class LinearExponentialIntegrator(BaseIntegrator):
    """Exact scalar linear propagation with frozen control or midpoint control.

    Midpoint is a deterministic two-stage control evaluation with the same exact
    reference noise covariance. No second-order weak-accuracy claim is made.
    """

    def __init__(self, *args, control_evaluation="left", **kwargs):
        super().__init__(*args, **kwargs)
        if control_evaluation not in ("left", "midpoint"):
            raise ValueError("control_evaluation must be left or midpoint")
        self.control_evaluation = control_evaluation

    def kernel_metadata(self):
        return dict(
            super().kernel_metadata(), control_evaluation=self.control_evaluation
        )

    def transition(self, data, time, dt, backward=False):
        if not isinstance(self.reference, LinearGaussianReference):
            raise ValueError(
                "Linear exponential solver requires LinearGaussianReference"
            )
        s, t = (time - dt, time) if backward else (time, time + dt)
        phi, k, q = self.reference.transition_coefficients(s, t, backward)
        name = "h_controller" if backward else "u_controller"
        model = getattr(self.sde, name, None)
        control = (
            torch.zeros_like(positions(data)) if model is None else model(time, data)
        )
        if self.control_evaluation == "midpoint" and model is not None:
            mid = (s + t) / 2
            hs, ht = (mid, t) if backward else (s, mid)
            hp, hk, _ = self.reference.transition_coefficients(hs, ht, backward)
            predictor = with_positions(
                data,
                coefficient(hp, data) * positions(data)
                + coefficient(hk, data) * control,
            )
            control = model(mid, predictor)
        mean = with_positions(
            data,
            coefficient(phi, data) * positions(data) + coefficient(k, data) * control,
        )
        return GaussianTransition(mean, q, self.sde.is_mean_free)


# Public Hydra paths retained; graph batches now share the very same kernels.
class AtomicIntegratorMixin:
    pass


class AtomicEulerMaruyamaIntegrator(AtomicIntegratorMixin, EulerMaruyamaIntegrator):
    pass


class AtomicExponentialIntegrator(AtomicIntegratorMixin, ExponentialIntegrator):
    pass
