# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
from tr_asbs.utils.geometry import subtract_mean_batch, subtract_com_vector
from mace.tools.torch_geometric import Batch

DataType = Union[torch.Tensor, Batch]


def _is_graph_data(data: DataType) -> bool:
    return hasattr(data, "pos") and hasattr(data, "batch")

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
        if _is_graph_data(data):
            noise = torch.randn_like(data.pos)
            if self.is_mean_free:
                noise = subtract_mean_batch(noise, data.batch)
        else:
            noise = torch.randn_like(data)
            if self.is_mean_free:
                noise = subtract_com_vector(noise, n_particles=data.shape[-2], n_dimensions=data.shape[-1]) 
        return noise

    def backward_step_diffusion(self, time: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient used by a reverse step ending at ``time - dt``.

        Continuous schedules retain the existing right-endpoint convention.
        A schedule that is constant on each integration interval can override
        this to use that interval's coefficient instead.
        """
        return self.diffusion(time)


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

    def inverse_variance(self, variance: torch.Tensor) -> torch.Tensor:
        lo, hi = torch.zeros_like(variance), torch.ones_like(variance)
        for _ in range(56):
            mid = (lo + hi) / 2
            left = self.diffsquare_integral(mid) < variance
            lo, hi = torch.where(left, mid, lo), torch.where(left, hi, mid)
        return (lo + hi) / 2

    def adjoint_transport(self, time: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(time)

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

        if _is_graph_data(data_0):
            data_0_pos, data_1_pos = data_0.pos, data_1.pos
            # Index the per-graph time by data_0.batch BEFORE unsqueezing --
            # time[data_0.batch, None] on an already-(B,1)-shaped time tensor
            # inserts an extra spurious dimension ((N,1,1) instead of (N,1)).
            time_per_atom = time[data_0.batch]
            if time_per_atom.dim() == 1:
                time_per_atom = time_per_atom.unsqueeze(-1)
            t_reparam = self.reparametrize_time(time_per_atom)
        else:
            if time.dim() == 1:
                time = time.unsqueeze(-1)
            data_size = data_0.shape
            data_0_pos, data_1_pos = data_0.reshape(data_size[0], -1), data_1.reshape(data_size[0], -1)
            t_reparam = self.reparametrize_time(time)
            
        var = t_reparam * (1 - t_reparam) * self.total_variance
        std = torch.sqrt(var + eps).to(data_0_pos)
        t_reparam = t_reparam.to(data_0_pos)
        
        mean = (1 - t_reparam) * data_0_pos + t_reparam * data_1_pos
        
        # Project noise using the original particle/graph layout, not the
        # flattened bridge layout (which would mix samples when mean-free).
        noise = self.randn_like(data_0).reshape(data_0_pos.shape)
        pos_t = mean + std * noise

        if _is_graph_data(data_0):
            data_t = data_0.clone()
            data_t.pos = pos_t
            return data_t
        else:
            pos_t = pos_t.reshape(data_size)
            return pos_t

    @torch.no_grad()
    def cond_score_t0(self, time: torch.Tensor, data_0: DataType, data_t: DataType) -> torch.Tensor:
        """Compute ∇_{X_t} log p_{t|0} (X_t | X_0)."""
        self._validate_pair(data_0, data_t)

        if _is_graph_data(data_0):
            data_size = data_0.pos.shape
            d0, dt = data_0.pos.reshape(data_size[0], -1), data_t.pos.reshape(data_size[0], -1)
            # See sample_posterior for why data_0.batch must index time BEFORE
            # any unsqueezing.
            time_per_atom = time[data_0.batch]
            if time_per_atom.dim() == 1:
                time_per_atom = time_per_atom.unsqueeze(-1)
            var = self.diffsquare_integral(time_per_atom)
        else:
            if time.dim() == 1:
                time = time.unsqueeze(-1)
            data_size = data_0.shape
            d0, dt = data_0.reshape(data_size[0], -1), data_t.reshape(data_size[0], -1)
            var = self.diffsquare_integral(time)
            
        val = (d0 - dt) / var.to(d0)
        return val.reshape(data_size)

    @torch.no_grad()
    def cond_score_1t(self, time: torch.Tensor, data_1: DataType, data_t: DataType) -> torch.Tensor:
        """Compute ∇_{X_t} log p_{1|t} (X_1 | X_t)."""
        self._validate_pair(data_1, data_t)

        if _is_graph_data(data_1):
            data_size = data_1.pos.shape
            d1, dt = data_1.pos.reshape(data_size[0], -1), data_t.pos.reshape(data_size[0], -1)
            # See sample_posterior for why data_1.batch must index time BEFORE
            # any unsqueezing.
            time_per_atom = time[data_1.batch]
            if time_per_atom.dim() == 1:
                time_per_atom = time_per_atom.unsqueeze(-1)
            var = self.total_variance - self.diffsquare_integral(time_per_atom)
        else:
            if time.dim() == 1:
                time = time.unsqueeze(-1)
            data_size = data_1.shape
            d1, dt = data_1.reshape(data_size[0], -1), data_t.reshape(data_size[0], -1)
            var = self.total_variance - self.diffsquare_integral(time)

        val = (d1 - dt) / var.to(d1)
        return val.reshape(data_size)

    def _validate_pair(self, d_a: DataType, d_b: DataType):
        if type(d_a) != type(d_b):
            raise ValueError("Data types must match.")
        if _is_graph_data(d_a):
            if (d_a.batch != d_b.batch).any():
                raise ValueError("Batch indices must match.")


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


class PISCosineSDE(ZeroDriftSDE):
    """Zero-drift reference with the exact discrete PIS cosine coefficients.

    PIS consumes its one-indexed schedule in reverse order. Here coefficient
    ``i`` applies to the forward interval ``[i / num_steps, (i+1) / num_steps)``.
    Squared diffusion is therefore piecewise constant and its integral is
    piecewise linear, making every Gaussian transition variance exact on the
    matched uniform grid used by the exponential integrator and path weights.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        num_steps: int = 128,
        s: float = 0.008,
        power: float = 2.0,
        is_mean_free: bool = False,
    ):
        super().__init__(is_mean_free=is_mean_free)
        # Import here to keep the usual VE process independent of the
        # variational package and share PIS's exact schedule implementation.
        from tr_asbs.variational.schedules import cosine_diffusion_schedule

        values = cosine_diffusion_schedule(
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s=s,
            power=power,
            dtype=torch.double,
        ).flip(0)
        self.num_steps = int(num_steps)
        self.register_buffer("step_diffusions", values)
        self.register_buffer(
            "cumulative_variances",
            torch.cat((values.new_zeros(1), values.square().cumsum(0) / self.num_steps)),
        )

    def _scaled_time(self, time: torch.Tensor | float) -> torch.Tensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.step_diffusions.device)
        return (time * self.num_steps).clamp(0.0, float(self.num_steps))

    def _interval_index(self, time: torch.Tensor | float) -> torch.Tensor:
        return self._scaled_time(time).floor().long().clamp(max=self.num_steps - 1)

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        coefficient = self.step_diffusions[self._interval_index(time)]
        return coefficient.to(time)

    def backward_step_diffusion(self, time: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # The backward control is evaluated at the right endpoint, but its
        # diffusion must belong to the interval traversed by that step.
        return self.diffusion(time - dt)

    def diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        scaled_time = self._scaled_time(time)
        index = scaled_time.floor().long().clamp(max=self.num_steps - 1)
        fraction = scaled_time - index
        return self.cumulative_variances[index] + (
            fraction * self.step_diffusions[index].square() / self.num_steps
        )


class EDMSDE(ZeroDriftSDE):
    """EDM schedule from [Karras et al., 2022]."""
    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 1.0, rho: float = 7.0, is_mean_free: bool = False):
        super().__init__(is_mean_free=is_mean_free)
        self.register_buffer("sigma_min", torch.as_tensor(sigma_min, dtype=torch.double))
        self.register_buffer("sigma_max", torch.as_tensor(sigma_max, dtype=torch.double))
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.double))

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        # self.sigma_max/sigma_min/rho are double buffers (see __init__) for
        # numerical precision, but a 0-dim double result combined with a
        # 0-dim float `time` promotes to double under standard (non-"weak")
        # tensor promotion -- and once broadcast-unsqueezed against a
        # multi-dim float32 state elsewhere (drift()/bwd_drift()), that
        # double is no longer scalar-exempt and upcasts the ENTIRE state,
        # eventually reaching a float32-weight network and crashing with a
        # dtype mismatch (only ever exercised by a flat, non-atomic
        # integrator + EDMSDE combination -- the atomic integrators already
        # work around this downstream via their own explicit post-step
        # casts). Cast back to `time`'s dtype here instead, matching
        # VESDE.diffusion()'s existing `.to(time)`, to fix it at the source.
        diffusion = (
            (1 - time) * self.sigma_max ** (1 / self.rho)
            + time * self.sigma_min ** (1 / self.rho)
        ) ** self.rho
        return diffusion.to(time)

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

        diff = self.diffusion(time)  # shape: () or (B,)

        if _is_graph_data(data):
            # diff is either scalar (broadcasts fine as-is) or per-graph (B,);
            # in the latter case expand to per-atom via data.batch so it
            # broadcasts against u_controller's per-atom (N_total_atoms, D)
            # output, instead of data.dim() (which graph data doesn't have).
            if diff.dim() > 0:
                diff = diff[data.batch]
            while diff.dim() < data.pos.dim():
                diff = diff.unsqueeze(-1)
        else:
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
        self,
        time: torch.Tensor,
        data: DataType,
        return_control: bool = False,
        interval_dt: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        diff = (
            self.diffusion(time)
            if interval_dt is None
            else self.base_sde.backward_step_diffusion(time, interval_dt)
        )

        if _is_graph_data(data):
            # See ControlledSDE.drift for why graph data needs explicit
            # per-graph -> per-atom expansion instead of relying on
            # unsqueeze-based broadcasting.
            if diff.dim() > 0:
                diff = diff[data.batch]
            while diff.dim() < data.pos.dim():
                diff = diff.unsqueeze(-1)

        else:
            while diff.dim() < data.dim():
                diff = diff.unsqueeze(-1)

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
