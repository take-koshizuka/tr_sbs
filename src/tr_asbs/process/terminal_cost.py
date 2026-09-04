# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import ase.units as units
import torch
import torch.nn as nn

from tr_asbs.data.atomic_data import ThermoAtomicData
from tr_asbs.potential.base_potential import BasePotential


class BaseTerminalCost(nn.Module, ABC):
    """Base class for terminal cost functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def forward(self, data_1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute terminal cost for data at time 1."""
        pass


class BaseGradEnergy:
    """Compute clipped gradient of energy E(X_1)."""

    def __init__(self, potential, grad_clip_val: float | None = None, **kwargs):
        self.potential = potential
        self.grad_clip_val = grad_clip_val

    def clip(self, grad_E: torch.Tensor) -> torch.Tensor:
        if self.grad_clip_val is not None:
            norm = torch.linalg.vector_norm(grad_E, dim=-1).detach()
            clip_coefficient = torch.clamp(self.grad_clip_val / (norm + 1e-6), max=1)
            clip_coefficient = clip_coefficient.unsqueeze(-1)
        else:
            clip_coefficient = torch.ones_like(grad_E)
        return grad_E * clip_coefficient

    def potential_grad(
        self,
        data: torch.Tensor,
        eps: float = 1e-10,
        return_grad_norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        energy = self.potential.eval(data)
        grad = self.potential.grad_E(data)
        clip_grad = self.clip(grad)
        result = {
            "energy": energy,
            "grad": clip_grad,
        }
        if return_grad_norm:
            grad_norm = torch.sqrt(grad.pow(2).sum(dim=-1, keepdim=True) + eps)
            result["grad_norm"] = grad_norm.squeeze(-1)
        return result

    def grad_E(self, data: torch.Tensor) -> torch.Tensor:
        grad_E = self.potential.grad_E(data)
        return self.clip(grad_E)


class ASBSTerminalCost(BaseTerminalCost, BaseGradEnergy):
    """Compute (∇E + ∇log h)(X_1), where h is the corrector."""

    def __init__(self, corrector, potential, grad_clip_val: float | None = None, **kwargs):
        BaseTerminalCost.__init__(self)
        BaseGradEnergy.__init__(self, potential, grad_clip_val, **kwargs)
        self.corrector = corrector

    def forward(
        self,
        data_1: torch.Tensor,
        is_init_stage: bool = False,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        grad_result = self.potential_grad(data_1, return_grad_norm=return_grad_norm)
        time_1 = torch.ones(data_1.shape[0]).to(data_1)

        if is_init_stage:
            result = {"terminal_cost": grad_result["grad"]}
        else:
            with torch.no_grad():
                corrector = self.corrector(time_1, data_1)
            result = {"terminal_cost": grad_result["grad"] + corrector}

        if return_energy:
            result["energy"] = grad_result["energy"]
        if return_grad_norm:
            result["grad_norm"] = grad_result["grad_norm"]
        return result


class AtomicPotentialGradientMixin:
    """Mixin providing potential-gradient functionality for graph-batch
    (ThermoAtomicData) potentials -- counterpart of BaseGradEnergy above for
    the Ala2/OpenMM+PaiNN atomic pipeline, whose potentials implement
    forward(data) -> {"energy", "forces"} rather than .eval()/.grad_E()."""

    def __init__(self, potential: BasePotential, grad_clip_val: float | None = None):
        self.potential = potential
        self.potential.eval()
        self.grad_clip_val = grad_clip_val

    @torch.no_grad()
    def potential_grad(
        self,
        data: ThermoAtomicData,
        eps: float = 1e-10,
        return_grad_norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the gradient of the potential energy (negative forces) divided by
        the temperature, and return the original potential energy.
        NOTE: The gradients are grad(U)/k_B T in units of Angstrom^-1, and the
        potential energy is in eV."""
        pot_result = self.potential(data)
        grad = -pot_result["forces"] / (units.kB * data.temperature[data.batch, None])
        grad_norm = torch.sqrt(grad.pow(2).sum(dim=-1, keepdim=True) + eps)
        if self.grad_clip_val is not None:
            clip_coefficient = torch.clamp(self.grad_clip_val / grad_norm, max=1.0)
        else:
            clip_coefficient = torch.ones_like(grad)
        result = {
            "energy": pot_result["energy"],
            "grad": grad * clip_coefficient,
        }
        if return_grad_norm:
            result["grad_norm"] = grad_norm.squeeze(-1)
        return result


class AtomicASBSTerminalCost(BaseTerminalCost, AtomicPotentialGradientMixin):
    """Terminal cost function for ASBS on graph-batch (ThermoAtomicData)
    potentials -- counterpart of ASBSTerminalCost above for the Ala2/
    OpenMM+PaiNN atomic pipeline."""

    def __init__(
        self,
        potential: BasePotential,
        corrector: nn.Module,
        grad_clip_val: float | None = None,
    ):
        BaseTerminalCost.__init__(self)
        AtomicPotentialGradientMixin.__init__(self, potential, grad_clip_val)
        self.corrector = corrector

    @torch.no_grad()
    def forward(
        self,
        data_1: ThermoAtomicData,
        is_init_stage: bool = False,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        grad_result = self.potential_grad(data_1, return_grad_norm=return_grad_norm)
        time_1 = torch.ones(data_1.num_graphs).to(data_1.pos)
        if is_init_stage:  # for initial stage, corrector is not trained yet
            result = {"terminal_cost": grad_result["grad"]}
        else:
            corrector = self.corrector(time_1, data_1)
            result = {"terminal_cost": grad_result["grad"] + corrector}
        if return_energy:
            result["energy"] = grad_result["energy"]
        if return_grad_norm:
            result["grad_norm"] = grad_result["grad_norm"]
        return result
