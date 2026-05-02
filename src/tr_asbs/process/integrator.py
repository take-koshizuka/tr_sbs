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
from tr_asbs.utils.geometry import subtract_mean_batch, subtract_com_vector
from tr_asbs.utils.gaussian_utils import gaussian_log_prob


def _is_graph_data(data: DataType) -> bool:
    return hasattr(data, "pos") and hasattr(data, "batch")


def _data_device(data: DataType) -> torch.device:
    return data.pos.device if _is_graph_data(data) else data.device

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
        zero_noise: bool = False,
    ) -> DataType:
        """Perform a single step of the forward SDE integration."""
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

        timesteps = timesteps.to(_data_device(initial_data))
        data = initial_data.clone() if hasattr(initial_data, "clone") else initial_data

        trajectory = [data] if return_trajectory else []

        iterator = range(len(timesteps) - 1)
        if progress_bar:
            iterator = tqdm(iterator, desc="sdeint_fwd", leave=False)

        for i in iterator:
            dt = timesteps[i + 1] - timesteps[i]
            data = self.step(
                data,
                time=timesteps[i],
                dt=dt,
                zero_noise=(zero_last_step_noise and i == len(timesteps) - 2),
            )
            
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
        if _is_graph_data(data):
            data.pos = subtract_mean_batch(data.pos, data.batch)
        else:
            data = subtract_com_vector(data, data.shape[-2], data.shape[-1])
        return data
        
    @abstractmethod
    @torch.no_grad()
    def step_backward(
        self,
        data: DataType,
        time: torch.Tensor,
        dt: torch.Tensor,  # distinct from forward dt, usually positive magnitude
        zero_noise: bool = False,
    ) -> DataType:
        """
        Perform a single step of the reverse SDE integration.
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
        
        timesteps = timesteps.to(_data_device(final_data))
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
            
            data = self.step_backward(
                data,
                time=t_current,
                dt=dt,
                zero_noise=(zero_last_step_noise and i == len(times) - 2),
            )

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

        timesteps = timesteps.to(_data_device(initial_data))
        is_graph = _is_graph_data(initial_data)
        data = initial_data
        device = _data_device(data)

        batch_size = initial_data.pos.shape[0] if is_graph else initial_data.shape[0]
        D = data.pos.shape[1] if is_graph else data.reshape(batch_size, -1).shape[1]
        log_Pref_Ph = torch.zeros(batch_size, device=device) # log(P_ref / P_backward)
        log_Pref_Pu = torch.zeros(batch_size, device=device) # log(P_ref / P_forward_model)
        log_Pref_Pu_det = torch.zeros(batch_size, device=device) # Low variance version

        trajectory = [data] if return_trajectory else []

        log_Pref_Ph += prior_log_prob_fn(data)
        iterator = range(len(timesteps) - 1)
        
        if progress_bar:
            iterator = tqdm(iterator, desc="sdeint", leave=False)
        
        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i+1]
            dt = t_next - t
            # 1. Step Forward (Reference Process - self)
            # We assume step returns just data. Energy calc might be inside step or needed separately.
            # Here we follow the pattern of calling step(), but for weights we usually need drift/diff separately.
            
            # To get exact drift/diff for KL, we calculate them BEFORE the update
            fwd_mean_ref = self.step(data.clone(), time=t, dt=dt, zero_noise=True)
            fwd_mean = integrator.step(data.clone(), time=t, dt=dt, zero_noise=True)
            next_data = self.step(data, time=t, dt=dt)
            bwd_mean = integrator.step_backward(next_data.clone(), time=t_next, dt=dt, zero_noise=True)
            
            # Perform the update
            diff_sq = self.sde.base_sde.diffsquare_integral(t_next) - self.sde.base_sde.diffsquare_integral(t)
            sigma_step = diff_sq.clamp(min=1e-20).sqrt()
            if is_graph:
                fwd_ref_pos = fwd_mean_ref.pos
                fwd_pos = fwd_mean.pos
                next_pos = next_data.pos
                bwd_pos = bwd_mean.pos
                data_pos = data.pos
            else:
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

            if center_every_step:
                next_data = self._apply_centering(next_data)
                
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
        zero_noise: bool = False,
    ) -> DataType:
        """Perform a single step of the Euler-Maruyama integration."""
        drift = self.sde.drift(time, data) * dt
        if zero_noise:
            if _is_graph_data(data):
                data.pos = data.pos + drift
            else:
                data = data + drift
            return data
        
        diffusion = self.sde.diffusion(time) * dt.sqrt() * self.sde.randn_like(data)
        if _is_graph_data(data):
            data.pos = data.pos + drift + diffusion
        else:
            data = data + drift + diffusion
            
        return data
    
class ExponentialIntegrator(BaseIntegrator):
    """Exponential integrator for SDEs."""

    @torch.no_grad()
    def step(
        self,
        data: DataType,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
        zero_noise: bool = False,
    ) -> DataType:
        """Perform a single step of the Exponential integration."""
        drift = self.sde.drift(time, data) * dt
        if zero_noise:
            if _is_graph_data(data):
                data.pos = data.pos + drift
            else:
                data = data + drift
            return data
        
        diff_sq = self.sde.base_sde.diffsquare_integral(time + dt) - self.sde.base_sde.diffsquare_integral(time)
        diffusion = diff_sq.sqrt() * self.sde.randn_like(data)
        if _is_graph_data(data):
            data.pos = data.pos + drift + diffusion
        else:
            data = data + drift + diffusion
        return data
    
    @torch.no_grad()
    def step_backward(
        self,
        data: DataType,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
        zero_noise: bool = False,
    ) -> DataType:
        """Perform a single step of the Exponential integration backward."""
        drift = (-self.sde.bwd_drift(time, data)) * (-dt)
        if zero_noise:
            if _is_graph_data(data):
                data.pos = data.pos + drift
            else:
                data = data + drift
            return data
        
        diff_sq = self.sde.base_sde.diffsquare_integral(time) - self.sde.base_sde.diffsquare_integral(time - dt)
        diffusion = diff_sq.sqrt() * self.sde.randn_like(data)
        if _is_graph_data(data):
            data.pos = data.pos + drift + diffusion
        else:
            data = data + drift + diffusion
            
        return data
