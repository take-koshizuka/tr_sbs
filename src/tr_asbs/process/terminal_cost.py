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
from tr_asbs.process.sde import BaseSDE

class BaseTerminalCost(nn.Module, ABC):
    """Base class for terminal cost functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def forward(self, data_1: ThermoAtomicData) -> torch.Tensor:
        """Compute the terminal cost for a given atomic data instance (at time 1)."""
        pass


class PotentialGradientMixin:
    """Mixin class to provide potential gradient functionality."""

    def __init__(self, potential: BasePotential, grad_clip_val: float | None = None):
        self.potential = potential
        self.potential.eval()
        self.grad_clip_val = grad_clip_val

    @torch.no_grad()
    def potential_grad(
        self,
        data: ThermoAtomicData,
        eps: float = 1e-10,
        return_grad_norm: float = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the gradient of the potential energy (negative forces) divided by
        the temperature, and return the original potential energy.
        NOTE: The gradients are ∇U/k_B T in units of Angstrom^-1, and the potential
        energy is in eV."""
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


class ASTerminalCostAtom(BaseTerminalCost, PotentialGradientMixin):
    """Terminal cost function for Adjoint Sampling."""

    def __init__(
        self,
        base_sde: BaseSDE,
        potential: BasePotential,
        grad_clip_val: float | None = None,
    ):
        raise NotImplementedError("ASTerminalCost is not implemented yet.")
        BaseTerminalCost.__init__(self)
        PotentialGradientMixin.__init__(self, potential, grad_clip_val)
        self.base_sde = base_sde

    @torch.no_grad()
    def forward(
        self,
        data_1: ThermoAtomicData,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        raise NotImplementedError("ASTerminalCost is not implemented yet.")
        potential_grad, energy = self.potential_grad(data_1)
        time_1 = torch.ones(data_1.num_graphs).to(data_1.pos)
        score = self.base_sde.marginal_score(time_1, data_1)
        terminal_cost = potential_grad + score
        return (terminal_cost, energy) if return_energy else terminal_cost

class BaseGradEnergy:
    """ Compute ∇E(X_1)
    """
    def __init__(
        self, 
        potential: BasePotential,
        grad_clip_val: float | None = None, 
        **kwargs
    ):
        self.potential = potential
        self.grad_clip_val = grad_clip_val

    def clip(self, grad_E):
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
        return_grad_norm: float = False
    ):
        energy = self.potential.eval(data)
        grad = self.potential.grad_E(data) # ∇E
        clip_grad = self.clip(grad)
        result = {
            "energy": energy,
            "grad": clip_grad,
        }
        if return_grad_norm:
            grad_norm = torch.sqrt(grad.pow(2).sum(dim=-1, keepdim=True) + eps)
            result["grad_norm"] = grad_norm.squeeze(-1)
        return result
        
    def grad_E(self, data):
        grad_E = self.potential.grad_E(data)
        return self.clip(grad_E)
    
class ASBSTerminalCost(BaseTerminalCost, BaseGradEnergy):
    """ Compute (∇E + ∇log h)(X_1), where h is the corrector of ASBS.
    """
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
    ) -> torch.Tensor:
        grad_result = self.potential_grad(data_1, return_grad_norm=return_grad_norm)
        time_1 = torch.ones(data_1.shape[0]).to(data_1)
        if is_init_stage:  # for initial stage, corrector is not trained yet
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
    
class ASBSTerminalCostAtom(BaseTerminalCost, PotentialGradientMixin):
    """Terminal cost function for ASBS."""

    def __init__(
        self,
        potential: BasePotential,
        corrector: nn.Module,
        grad_clip_val: float | None = None,
    ):
        BaseTerminalCost.__init__(self)
        PotentialGradientMixin.__init__(self, potential, grad_clip_val)
        self.corrector = corrector

    @torch.no_grad()
    def forward(
        self,
        data_1: ThermoAtomicData,
        is_init_stage: bool = False,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the terminal cost for ASBS."""
        grad_result = self.potential_grad(data_1, return_grad_norm=return_grad_norm)
        time_1 = torch.ones(data_1.num_graphs).to(data_1.pos)
        if is_init_stage:  # for initial stage, corrector is not trained yet
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

class ChemGradTermCost(BaseTerminalCost):
    """ Compute (∇E + corrector)(X_1) based on ASBSTerminalCost structure.
    """
    def __init__(self, corrector, potential, grad_clip_val: float | None = None, **kwargs):
        BaseTerminalCost.__init__(self)
        self.corrector = corrector
        self.potential = potential
        self.grad_clip_val = grad_clip_val

    def forward(
        self,
        graph_state1,
        is_init_stage: bool = False,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> dict:
        # 1. Calculate Gradient (and Energy if needed)
        grad_result = self.grad_E(
            graph_state1, 
            return_energy=return_energy, 
            return_grad_norm=return_grad_norm
        )
        
        # 2. Setup time tensor
        N = graph_state1["batch"][-1] + 1
        t1 = torch.ones(N).to(graph_state1["positions"])

        # 3. Add Corrector logic
        if is_init_stage:
            # For initial stage, corrector is not used
            result = {"terminal_cost": grad_result["grad"]}
        else:
            with torch.no_grad():
                corrector_val = self.corrector(t1, graph_state1)
            result = {"terminal_cost": grad_result["grad"] + corrector_val}

        # 4. Construct Output
        if return_energy and "energy" in grad_result:
            result["energy"] = grad_result["energy"]
        if return_grad_norm and "grad_norm" in grad_result:
            result["grad_norm"] = grad_result["grad_norm"]
            
        return result

    def grad_E(self, graph_state1, return_energy=False, return_grad_norm=False):
        # NOTE: Keeping original logic but returning a dictionary
        with torch.no_grad():
            graph_state1_clone = graph_state1.clone()

        F = self.potential(graph_state1_clone)
        grad_E = - F["forces"] - F["reg_forces"]

        # Clip gradients if value is set
        if self.grad_clip_val is not None:
            # Assuming self.clip uses self.grad_clip_val or similar logic exists
            # If self.clip is not defined in GradTermCost, use explicit clipping:
            # grad_E = torch.clamp(grad_E, -self.grad_clip_val, self.grad_clip_val)
            grad_E = self.clip(grad_E) 
        
        grad_E = graph_utils.subtract_com_batch(
            grad_E, graph_state1["batch"]
        )

        res = {"grad": grad_E}
        
        if return_energy:
            # Assuming F contains "energy" or it needs to be calculated
            if "energy" in F:
                res["energy"] = F["energy"]
            else:
                # Handle cases where energy key might differ or is missing
                res["energy"] = None 

        if return_grad_norm:
            res["grad_norm"] = torch.norm(grad_E, p=2, dim=-1).mean()

        return res
    
