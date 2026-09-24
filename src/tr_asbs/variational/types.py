from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class TargetDensity(Protocol):
    """Minimal target-density interface used by variational samplers."""

    dim: int

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return one log-density value per sample."""

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``grad_x log_prob(x)`` with the same shape as ``x``."""


@dataclass
class RolloutResult:
    """Outputs shared by stochastic-control variational methods.

    All cost tensors contain one scalar per sample.  ``trajectory``, when
    requested, has shape ``[num_steps, batch_size, dim]`` and stores the state
    after each integration step.
    """

    samples: torch.Tensor
    running_cost: torch.Tensor
    stochastic_cost: torch.Tensor
    terminal_cost: torch.Tensor
    trajectory: torch.Tensor | None = None

    @property
    def neg_elbo(self) -> torch.Tensor:
        """Per-sample negative ELBO integrand."""

        return self.running_cost + self.terminal_cost

    @property
    def log_weight(self) -> torch.Tensor:
        """Per-sample reverse importance log weight."""

        return -(self.running_cost + self.stochastic_cost + self.terminal_cost)


class VariationalMethod(nn.Module, ABC):
    """Base class for trainable variational sampling dynamics."""

    @abstractmethod
    def rollout(self, batch_size: int, **kwargs: object) -> RolloutResult:
        """Simulate a batch and return its pathwise costs."""
