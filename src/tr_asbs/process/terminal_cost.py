# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
