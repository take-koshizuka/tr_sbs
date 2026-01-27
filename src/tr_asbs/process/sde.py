# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
from fairchem.core.datasets.atomic_data import AtomicData
from tr_asbs.utils.geometry import is_mean_free_data, subtract_mean_batch, subtract_com_vector
from mace.tools.torch_geometric import Batch

DataType = Union[torch.Tensor, Batch]

class BaseSDE(nn.Module, ABC):
    """Base class for Forward SDEs: dX_t = f(t, X_t) dt + g(t) dW_t."""

    def __init__(self, is_mean_free):
        super().__init__()
        self.is_mean_free = is_mean_free
        
    @abstractmethod
    def drift(self, time: torch.Tensor, data: DataType) -> torch.Tensor:
        """Forward Drift term f(t, X_t)."""
        pass

    @abstractmethod
    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        """Diffusion term g(t)."""
        pass

    def randn_like(self, data: DataType) -> torch.Tensor:
        """Generate random noise with the same shape as positions."""
        if isinstance(data, AtomicData):
            noise = torch.randn_like(data.pos)
            if self.is_mean_free:
                noise = subtract_mean_batch(noise, data.batch)
        else:
            noise = torch.randn_like(data)
            if self.is_mean_free:
                noise = subtract_com_vector(noise, n_particles=data.shape[-2], n_dimensions=data.shape[-1]) 
        return noise


class FwdBwdSDE(BaseSDE):
    """Base class for SDEs that support both Forward and Backward dynamics.
    Forward:  dX_t = f(t, X_t) dt + g(t) dW_t
    Backward: dY_t = b(t, Y_t) dt + g(t) dW_t (where b is the backward drift)
    """

    @abstractmethod
    def bwd_drift(self, time: torch.Tensor, data: DataType) -> torch.Tensor:
        """Backward Drift term."""
        pass


class ZeroDriftSDE(FwdBwdSDE, ABC):
    """Base class for mean-free SDEs where the base drift is zero for both directions.
    Ideally used as a reference process.
    """

    def drift(self, time: torch.Tensor, data: DataType) -> torch.Tensor:
        """Return zero drift term."""
        return self._zeros_like_data(data)

    def bwd_drift(self, time: torch.Tensor, data: DataType) -> torch.Tensor:
        """Return zero backward drift term."""
        return self._zeros_like_data(data)

    def _zeros_like_data(self, data: DataType) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return torch.zeros_like(data)
        else:
            return torch.zeros_like(data.pos)

    @abstractmethod
    def diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        """Compute integral_0^t g(s)^2 ds."""
        pass

    @property
    def total_variance(self) -> torch.DoubleTensor:
        return self.diffsquare_integral(1.0)

    def reparametrize_time(self, time: torch.Tensor) -> torch.DoubleTensor:
        return self.diffsquare_integral(time) / self.total_variance

    @torch.no_grad()
    def sample_posterior(
        self,
        time: torch.Tensor,  # [num_graphs,]
        data_0: DataType,    # state at t=0
        data_1: DataType,    # state at t=1
        eps: float = 1e-10,
    ) -> DataType:
        """Sample X_t ~ p_{t|0, 1} (X_t | X_0, X_1)."""
        self._validate_pair(data_0, data_1)
        
        if time.dim() == 1:
            time = time.unsqueeze(-1)

        if isinstance(data_0, AtomicData):
            data_0_pos, data_1_pos = data_0.pos, data_1.pos
            t_reparam = self.reparametrize_time(time[data_0.batch, None])
        else:
            data_size = data_0.shape
            data_0_pos, data_1_pos = data_0.reshape(data_size[0], -1), data_1.reshape(data_size[0], -1)
            t_reparam = self.reparametrize_time(time)
            
        var = t_reparam * (1 - t_reparam) * self.total_variance
        std = torch.sqrt(var + eps).to(data_0_pos)
        t_reparam = t_reparam.to(data_0_pos)
        
        mean = (1 - t_reparam) * data_0_pos + t_reparam * data_1_pos
        
        pos_t = mean + std * self.randn_like(data_0_pos)

        if isinstance(data_0, AtomicData):
            data_t = data_0.clone()
            data_t.pos = pos_t
            if not is_mean_free_data(data_t):
                raise ValueError("Sampled data_t is not mean-free")
            return data_t
        else:
            pos_t = pos_t.reshape(data_size)
            return pos_t

    @torch.no_grad()
    def cond_score_t0(self, time: torch.Tensor, data_0: DataType, data_t: DataType) -> torch.Tensor:
        """Compute ∇_{X_t} log p_{t|0} (X_t | X_0)."""
        self._validate_pair(data_0, data_t)
        
        if time.dim() == 1:
            time = time.unsqueeze(-1)
            
        if isinstance(data_0, AtomicData):
            data_size = data_0.pos.shape
            d0, dt = data_0.pos.reshape(data_size[0], -1), data_t.pos.reshape(data_size[0], -1)
            var = self.diffsquare_integral(time[data_0.batch, None])
        else:
            data_size = data_0.shape
            d0, dt = data_0.reshape(data_size[0], -1), data_t.reshape(data_size[0], -1)
            var = self.diffsquare_integral(time)
            
        val = (d0 - dt) / var.to(d0)
        return val.reshape(data_size)

    @torch.no_grad()
    def cond_score_1t(self, time: torch.Tensor, data_1: DataType, data_t: DataType) -> torch.Tensor:
        """Compute ∇_{X_t} log p_{1|t} (X_1 | X_t)."""
        self._validate_pair(data_1, data_t)
        
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        
        if isinstance(data_1, AtomicData):
            data_size = data_1.pos.shape
            d1, dt = data_1.pos.reshape(data_size[0], -1), data_t.pos.reshape(data_size[0], -1)
            var = self.total_variance - self.diffsquare_integral(time[data_1.batch, None])
        else:
            data_size = data_1.shape
            d1, dt = data_1.reshape(data_size[0], -1), data_t.reshape(data_size[0], -1)
            var = self.total_variance - self.diffsquare_integral(time)

        val = (d1 - dt) / var.to(d1)
        return val.reshape(data_size)

    def _validate_pair(self, d_a: DataType, d_b: DataType):
        if type(d_a) != type(d_b):
            raise ValueError("Data types must match.")
        if isinstance(d_a, AtomicData):
            if (d_a.batch != d_b.batch).any():
                raise ValueError("Batch indices must match.")
            if not (is_mean_free_data(d_a) and is_mean_free_data(d_b)):
                raise ValueError("Data must be mean-free.")


class VESDE(ZeroDriftSDE):
    """Variance exploding SDE."""
    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 1.0, is_mean_free: bool = False):
        super().__init__(is_mean_free=is_mean_free)
        self.register_buffer("sigma_min", torch.as_tensor(sigma_min, dtype=torch.double))
        self.register_buffer("sigma_max", torch.as_tensor(sigma_max, dtype=torch.double))

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        sigma_ratio = self.sigma_max / self.sigma_min
        diffusion = (
            self.sigma_min * (sigma_ratio ** (1 - time)) * torch.sqrt(2 * torch.log(sigma_ratio))
        )
        return diffusion.to(time)

    def diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.sigma_max.device)
        sigma_ratio = self.sigma_max / self.sigma_min
        return (self.sigma_max**2) * (1 - sigma_ratio ** (-2 * time))


class EDMSDE(ZeroDriftSDE):
    """EDM schedule from [Karras et al., 2022]."""
    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 1.0, rho: float = 7.0, is_mean_free: bool = False):
        super().__init__(is_mean_free=is_mean_free)
        self.register_buffer("sigma_min", torch.as_tensor(sigma_min, dtype=torch.double))
        self.register_buffer("sigma_max", torch.as_tensor(sigma_max, dtype=torch.double))
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.double))

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        return (
            (1 - time) * self.sigma_max ** (1 / self.rho)
            + time * self.sigma_min ** (1 / self.rho)
        ) ** self.rho

    def diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.sigma_max.device)
        # Note: Logic remains same as original
        return (
            self.sigma_max ** (2 + 1 / self.rho)
            - self.diffusion(time) ** (2 + 1 / self.rho)
        ) / (
            (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
            * (2 * self.rho + 1)
        )


class ControlledSDE(BaseSDE):
    """Forward Controlled SDE: dX_t = (f(t, X_t) + g(t)^2 u(t, X_t)) dt + g(t) dW_t"""

    def __init__(self, base_sde: BaseSDE, controller: nn.Module):
        super().__init__(base_sde.is_mean_free)
        self.base_sde = base_sde
        self.u_controller = controller

    def drift(
        self, time: torch.Tensor, data: DataType, return_control: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        diff = self.diffusion(time)  # shape: (B,)
        
        # --- 修正箇所: diff の次元を data に合わせる (B,) -> (B, 1) ---
        # data が (Batch, Dim, ...) の場合に対応できるよう、次元数が合うまで後ろに1を追加
        while diff.dim() < data.dim():
            diff = diff.unsqueeze(-1)
        # -----------------------------------------------------------

        # これで diff: (1000, 1), u_controller: (1000, 2) となり
        # Broadcasting によって (1000, 2) として計算されます
        control_term = diff * self.u_controller(time, data)
        
        base_drift = self.base_sde.drift(time, data)
        
        # ここも diff (1000, 1) * control_term (1000, 2) なのでOK
        drift = base_drift + diff * control_term
        
        if return_control:
            return drift, control_term
        return drift

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        return self.base_sde.diffusion(time)


class ControlledFBSDE(ControlledSDE, FwdBwdSDE):
    """Controlled Forward-Backward SDE.
    Forward: dX_t = (f + g^2 u) dt + g dW_t
    Backward: dY_t = (b + g^2 h) dt + g dW_t  (where b is base backward drift, usually -f + ...)
    """

    def __init__(self, base_sde: FwdBwdSDE, u_controller: nn.Module, h_controller: nn.Module):
        # Initialize Forward part via ControlledSDE
        ControlledSDE.__init__(self, base_sde, u_controller)
        self.h_controller = h_controller
        
        # Type check to ensure base supports backward
        if not isinstance(base_sde, FwdBwdSDE):
            raise TypeError("base_sde for ControlledFBSDE must inherit from FwdBwdSDE")

    # drift() is inherited from ControlledSDE

    def bwd_drift(
        self, time: torch.Tensor, data: DataType, return_control: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        diff = self.diffusion(time)
        control_term = diff * self.h_controller(time, data)
        base_drift = self.base_sde.bwd_drift(time, data)
        
        drift = base_drift + diff * control_term
        
        if return_control:
            return drift, control_term
        return drift
    




## Chemistry ##
class Chem:
    def randn_like(self, graph_state: Batch) -> torch.Tensor:
        return graph_utils.subtract_com_batch(
            torch.randn_like(graph_state["positions"]),
            graph_state["batch"],
        )

    def propagate(self, graph_state: Batch, d_position: torch.Tensor) -> Batch:
        # NOTE(ghliu): we do NOT create new graph_state (Batch)!!
        graph_state["positions"] = graph_state["positions"] + d_position
        graph_state["positions"] = graph_utils.subtract_com_batch(
            graph_state["positions"], graph_state["batch"]
        )
        return graph_state


# note: Chem goes before BaseSDE to override rand_like & propagate!
# NOTE(ghliu) rand_like & drift output positions not Batch!
class ChemVESDE(Chem, VESDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, *args, **kwargs):
        Chem.__init__(self)
        VESDE.__init__(self, *args, **kwargs)

    def drift(self, t, graph_state: Batch) -> torch.Tensor:
        return torch.zeros_like(graph_state["positions"])

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")

    def sample_posterior(
        self,
        t: torch.Tensor,
        graph_state0: Batch,
        graph_state1: Batch,
    ) -> Batch:
        assert graph_utils.is_zcom_graph(graph_state0)
        assert graph_utils.is_zcom_graph(graph_state1)

        (B, D), N = graph_state0["positions"].shape, t.shape

        x0 = graph_state0["positions"]
        x1 = graph_state1["positions"]
        tt = t[graph_state0["batch"], None]
        assert x0.shape == x1.shape == (B, D) and tt.shape == (B, 1)

        z = self.randn_like(graph_state0)
        xt = super().sample_posterior(tt, x0, x1, z=z)
        assert xt.shape == (B, D)

        graph_state_t = graph_utils.create_new_graph(
            graph_state0, xt, subtract_com=True
        )
        return graph_state_t

    def cond_score(
        self,
        graph_state0: Batch,
        t: torch.Tensor,
        graph_state_t: Batch,
    ) -> torch.Tensor:
        assert graph_utils.is_zcom_graph(graph_state0)
        assert graph_utils.is_zcom_graph(graph_state_t)

        (B, D), N = graph_state0["positions"].shape, t.shape

        x0 = graph_state0["positions"]
        xt = graph_state_t["positions"]
        tt = t[graph_state0["batch"], None]
        assert x0.shape == xt.shape == (B, D) and tt.shape == (B, 1)

        score = super().cond_score(x0, tt, xt)
        assert score.shape == (B, D)
        # TODO(ghliu) subtract zero?
        return score
