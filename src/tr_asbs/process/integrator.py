# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Sequence, Dict, Any, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from tr_asbs.process.sde import BaseSDE, FwdBwdSDE, DataType
from tr_asbs.utils.geometry import subtract_com_vector
from tr_asbs.utils.gaussian_utils import gaussian_log_prob


def get_timesteps(
    t0: torch.Tensor | float,
    t1: torch.Tensor | float,
    dt: torch.Tensor | float | None = None,
    steps: int | None = None,
    rescale_t: str | None = None,
) -> torch.Tensor:
    if not (steps is None) ^ (dt is None):
        raise ValueError("Exactly one of dt and steps should be defined.")
    if steps is None:
        steps = int(math.ceil((t1 - t0) / dt))
    else:
        steps = steps + 1

    if rescale_t is None:
        return torch.linspace(t0, t1, steps=steps, dtype=torch.float)
    elif rescale_t == "quad":
        return torch.sqrt(
            torch.linspace(t0, t1.square(), steps=steps, dtype=torch.float)
        ).clip(max=t1)
    elif rescale_t == "cosine":
        s = 0.008
        pre_phase = torch.linspace(t0, t1, steps=steps, dtype=torch.float) / t1
        phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5
        dts = torch.cos(phase) ** 4
        dts = (dts / dts.sum()) * t1
        dts_out = torch.concat(
            (torch.tensor([t0], dtype=torch.float), torch.cumsum(dts, -1))
        )
        return dts_out
    raise ValueError("Unknown timestep rescaling method.")


class BaseIntegrator(nn.Module, ABC):
    """Base class for SDE integrators."""

    def __init__(self, sde: BaseSDE, timesteps: torch.Tensor | None = None):
        super().__init__()
        self.sde = sde
        if timesteps is not None:
            self.register_buffer("timesteps", timesteps)

    @abstractmethod
    @torch.no_grad()
    def step(
        self,
        data: DataType,
        time: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[DataType, DataType]:
        """Return (deterministic_next, stochastic_next) for one forward step."""
        pass

    
    def _get_drift_diffusion(
        self, data: DataType, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to extract drift (mean) and diffusion (sigma) for log-prob calculations.
        """
        # This default implementation assumes self.sde has drift/diffusion methods.
        # Override this if your architecture is different.
        if hasattr(self.sde, "drift") and hasattr(self.sde, "diffusion"):
            return self.sde.drift(t, data), self.sde.diffusion(t, data)
        raise NotImplementedError(
            "run_with_logweights requires _get_drift_diffusion or sde.drift/diffusion implementation."
        )
        
    @torch.no_grad()
    def run(
        self,
        initial_data: DataType,
        timesteps: torch.Tensor | None = None,
        center_every_step: bool = True,
        zero_last_step_noise: bool = False,
        return_trajectory: bool = True,
        progress_bar: bool = True,
    ) -> Union[DataType, list[DataType]]:
        """Integrate the SDE forward over a sequence of timesteps."""
        if timesteps is None:
            if not hasattr(self, "timesteps"):
                raise ValueError("No timesteps provided and no default timesteps set.")
            timesteps = self.timesteps

        timesteps = timesteps.to(initial_data.device)
        data = initial_data.clone() if hasattr(initial_data, "clone") else initial_data

        trajectory = [data] if return_trajectory else []

        iterator = range(len(timesteps) - 1)
        if progress_bar:
            iterator = tqdm(iterator, desc="sdeint_fwd", leave=False)

        for i in iterator:
            dt = timesteps[i + 1] - timesteps[i]
            deterministic_next, stochastic_next = self.step(data, time=timesteps[i], dt=dt)
            data = deterministic_next if (zero_last_step_noise and i == len(timesteps) - 2) else stochastic_next
            
            # Center of Mass correction (optional)
            if center_every_step:
                data = self._apply_centering(data)

            if return_trajectory:
                trajectory.append(data) # Warning: careful with referencing mutable objects in list

        return trajectory if return_trajectory else data

    # ---------- Backward Run ----------
    
    # ---------- Utilities ----------
    def _apply_centering(self, data: DataType):
        """Helper to apply CoM centering."""
        return subtract_com_vector(data, data.shape[-2], data.shape[-1])
        
    @abstractmethod
    @torch.no_grad()
    def step_backward(
        self,
        data: DataType,
        time: torch.Tensor,
        dt: torch.Tensor,  # distinct from forward dt, usually positive magnitude
    ) -> Tuple[DataType, DataType]:
        """
        Return (deterministic_prev, stochastic_prev) for one reverse step.
        """
        pass
    
    @torch.no_grad()
    def run_backward(
        self,
        final_data: DataType,
        timesteps: torch.Tensor | None = None,
        zero_last_step_noise: bool = False,
        center_every_step: bool = True,
        return_trajectory: bool = True,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Integrate the SDE backward (reverse time)."""
        if timesteps is None:
            if not hasattr(self, "timesteps"):
                raise ValueError("No timesteps provided and no default timesteps set.")
            timesteps = self.timesteps
        
        timesteps = timesteps.to(final_data.device)
        assert timesteps[0] < timesteps[-1], "Timesteps should be in ascending order."
        # Reverse the timesteps
        times = timesteps.flip(0)
        
        data = final_data # assume clone handling similar to run
        trajectory = [data] if return_trajectory else []
        iterator = range(len(times) - 1)
        if progress_bar:
            iterator = tqdm(iterator, desc="sdeint_bwd", leave=False)

        for i in iterator:
            t_current = times[i]
            t_next = times[i+1] # This is actually 'previous' time in forward sense
            dt = t_current - t_next # Positive magnitude > 0
            
            deterministic_prev, stochastic_prev = self.step_backward(data, time=t_current, dt=dt)
            data = deterministic_prev if (zero_last_step_noise and i == len(times) - 2) else stochastic_prev

            if center_every_step:
                data = self._apply_centering(data)

            if return_trajectory:
                trajectory.append(data)

        return trajectory if return_trajectory else data
    
    # ------ Run with Log Weights (Girsanov / Importance Sampling) ----------
    @torch.no_grad()
    def run_with_logweights(
        self,
        integrator: "BaseIntegrator", # The 'sde_uh' in original code
        prior_log_prob_fn: Callable,
        target_log_prob_fn: Callable,
        initial_data: DataType,
        timesteps: torch.Tensor | None = None,
        center_every_step: bool = True,
        zero_last_step_noise: bool = False,
        return_trajectory: bool = False,
        progress_bar: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward integration that calculates log-importance weights between 
        this integrator (Reference) and another integrator (Target/Model).
        """
        if timesteps is None:
            if not hasattr(self, "timesteps"):
                raise ValueError("No timesteps provided.")
            timesteps = self.timesteps

        timesteps = timesteps.to(initial_data.device)
        data = initial_data
        device = data.device

        batch_size = initial_data.shape[0]
        D = data.reshape(batch_size, -1).shape[1]
        log_Pref_Ph = torch.zeros(batch_size, device=device) # log(P_ref / P_backward)
        log_Pref_Pu = torch.zeros(batch_size, device=device) # log(P_ref / P_forward_model)
        log_Pref_Pu_det = torch.zeros(batch_size, device=device) # Low variance version

        trajectory = [data] if return_trajectory else []

        log_Pref_Ph += prior_log_prob_fn(data)
        iterator = range(len(timesteps) - 1)
        
        if progress_bar:
            iterator = tqdm(iterator, desc="sdeint", leave=False)

        same_integrator = integrator is self
        
        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i+1]
            dt = t_next - t
            # Single forward call for reference: deterministic and stochastic outcomes are returned together.
            fwd_mean_ref, next_data_stochastic = self.step(data, time=t, dt=dt)
            if same_integrator:
                fwd_mean = fwd_mean_ref
            else:
                fwd_mean, _ = integrator.step(data.clone(), time=t, dt=dt)

            is_last_step = i == len(timesteps) - 2
            next_data = fwd_mean_ref if (zero_last_step_noise and is_last_step) else next_data_stochastic
            
            if center_every_step:
                next_data = self._apply_centering(next_data)
                
            bwd_mean, _ = integrator.step_backward(next_data.clone(), time=t_next, dt=dt)
            
            # Perform the update
            diff_sq = self.sde.base_sde.diffsquare_integral(t_next) - self.sde.base_sde.diffsquare_integral(t)
            sigma_step = diff_sq.clamp(min=1e-20).sqrt()
            fwd_ref_pos = fwd_mean_ref
            fwd_pos = fwd_mean
            next_pos = next_data
            bwd_pos = bwd_mean
            data_pos = data

            diff_means = (fwd_ref_pos - fwd_pos).reshape(batch_size, D)
            var_step = sigma_step.pow(2)
            log_Pref_Pu_det += 0.5 * (diff_means.pow(2) / var_step).sum(dim=-1)
            
            if sigma_step.numel() == 1:
                sigma_step_expand = torch.full((batch_size, D), sigma_step.item(), device=device)
            else:
                sigma_step_expand = sigma_step

            # --- Method B: Log Likelihoods (High Variance / Check) ---
            log_prob_ref_step = gaussian_log_prob(
                next_pos.reshape(batch_size, D),
                fwd_ref_pos.reshape(batch_size, D),
                sigma_step_expand, 
                D
            )
            log_prob_model_fwd_step = gaussian_log_prob(
                next_pos.reshape(batch_size, D),
                fwd_pos.reshape(batch_size, D),
                sigma_step_expand, 
                D
            )
            log_prob_model_bwd_step = gaussian_log_prob(
                data_pos.reshape(batch_size, D),
                bwd_pos.reshape(batch_size, D),
                sigma_step_expand, 
                D
            )

            log_Pref_Ph += (log_prob_ref_step - log_prob_model_bwd_step)
            log_Pref_Pu += (log_prob_ref_step - log_prob_model_fwd_step)
                
            # Update state
            data = next_data
            if return_trajectory:
                trajectory.append(data)

        # Final Target Log Prob subtraction
        target_log_prob = target_log_prob_fn(data)
        log_Pref_Ph -= target_log_prob

        return {
            "data_1": data,
            "trajectory": trajectory if return_trajectory else [initial_data, data],
            "log_Ph_Pref": -log_Pref_Ph,
            "log_Pu_Pref": -log_Pref_Pu,
            "log_Pref_Pu_det": log_Pref_Pu_det,
        }

class EulerMaruyamaIntegrator(BaseIntegrator):
    """Euler-Maruyama integrator for SDEs."""

    @torch.no_grad()
    def step(
        self,
        data: DataType,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
    ) -> Tuple[DataType, DataType]:
        """Return deterministic and stochastic next states for Euler-Maruyama."""
        drift = self.sde.drift(time, data) * dt
        diffusion = self.sde.diffusion(time) * dt.sqrt() * self.sde.randn_like(data)
        deterministic_next = data + drift
        stochastic_next = deterministic_next + diffusion
        return deterministic_next, stochastic_next
    
class ExponentialIntegrator(BaseIntegrator):
    """Exponential integrator for SDEs."""

    @torch.no_grad()
    def step(
        self,
        data: DataType,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
    ) -> Tuple[DataType, DataType]:
        """Return deterministic and stochastic next states for exponential integrator."""
        drift = self.sde.drift(time, data) * dt
        diff_sq = self.sde.base_sde.diffsquare_integral(time + dt) - self.sde.base_sde.diffsquare_integral(time)
        diffusion = diff_sq.sqrt() * self.sde.randn_like(data)
        deterministic_next = data + drift
        stochastic_next = deterministic_next + diffusion
        return deterministic_next, stochastic_next
    
    @torch.no_grad()
    def step_backward(
        self,
        data: DataType,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
    ) -> Tuple[DataType, DataType]:
        """Return deterministic and stochastic previous states for reverse exponential step."""
        drift = (-self.sde.bwd_drift(time, data)) * (-dt)
        diff_sq = self.sde.base_sde.diffsquare_integral(time) - self.sde.base_sde.diffsquare_integral(time - dt)
        diffusion = diff_sq.sqrt() * self.sde.randn_like(data)
        deterministic_prev = data + drift
        stochastic_prev = deterministic_prev + diffusion
        return deterministic_prev, stochastic_prev
