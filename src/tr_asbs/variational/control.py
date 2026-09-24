# Portions adapted from variational_sampling_methods commit 32de621.
# Copyright (c) 2023 Denis Blessing. Licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for attribution.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tr_asbs.model.mlp import FourierMLP


_TRUNCATED_NORMAL_STDDEV = 0.8796256610342398


def _lecun_normal_(layer: nn.Linear) -> None:
    """Initialize a linear layer like Flax's default ``lecun_normal``."""

    fan_in = layer.weight.shape[1]
    target_std = math.sqrt(1.0 / fan_in)
    untruncated_std = target_std / _TRUNCATED_NORMAL_STDDEV
    nn.init.trunc_normal_(
        layer.weight,
        mean=0.0,
        std=untruncated_std,
        a=-2.0 * untruncated_std,
        b=2.0 * untruncated_std,
    )
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ScoreConditionedControl(nn.Module):
    """Score-conditioned Fourier MLP used by PIS and DIS.

    The raw, one-indexed integration step is embedded with Fourier features.
    The network combines a learned state drift with a time-dependent scaling
    of a detached target-score hint.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        score_hint_mode: str = "provided",
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if hidden_dim is not None:
            if num_hid != 64 and num_hid != hidden_dim:
                raise ValueError(
                    "Specify only one hidden width via num_hid or hidden_dim"
                )
            num_hid = hidden_dim
        if num_hid <= 0:
            raise ValueError(f"num_hid must be positive, got {num_hid}")
        if outer_clip <= 0.0 or inner_clip <= 0.0:
            raise ValueError("outer_clip and inner_clip must be positive")
        if score_hint_mode not in {"provided", "zero"}:
            raise ValueError(
                "score_hint_mode must be 'provided' or 'zero', "
                f"got {score_hint_mode!r}"
            )

        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.num_hid = int(num_hid)
        self.outer_clip = float(outer_clip)
        self.inner_clip = float(inner_clip)
        self.weight_init = float(weight_init)
        self.bias_init = float(bias_init)
        self.score_hint_mode = score_hint_mode
        self.uses_score_hint = score_hint_mode == "provided"

        self.timestep_phase = nn.Parameter(torch.zeros(1, self.num_hid))
        self.register_buffer(
            "timestep_coeff",
            torch.linspace(0.1, 100.0, self.num_hid).unsqueeze(0),
        )

        self.time_state_in = nn.Linear(2 * self.num_hid, self.num_hid)
        self.time_state_out = nn.Linear(self.num_hid, self.num_hid)

        self.time_grad_in = nn.Linear(2 * self.num_hid, self.num_hid)
        self.time_grad_hidden = nn.ModuleList(
            [nn.Linear(self.num_hid, self.num_hid) for _ in range(self.num_layers)]
        )
        self.time_grad_out = nn.Linear(self.num_hid, self.dim)

        self.state_hidden = nn.ModuleList(
            [nn.Linear(self.dim + self.num_hid, self.num_hid)]
            + [
                nn.Linear(self.num_hid, self.num_hid)
                for _ in range(self.num_layers - 1)
            ]
        )
        self.state_out = nn.Linear(self.num_hid, self.dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the hidden and final-layer initializers from the Flax model."""

        hidden_layers = [
            self.time_state_in,
            self.time_state_out,
            self.time_grad_in,
            *self.time_grad_hidden,
            *self.state_hidden,
        ]
        for layer in hidden_layers:
            _lecun_normal_(layer)

        nn.init.constant_(self.time_grad_out.weight, self.weight_init)
        nn.init.constant_(self.time_grad_out.bias, self.bias_init)
        nn.init.constant_(self.state_out.weight, 1e-8)
        nn.init.zeros_(self.state_out.bias)
        nn.init.zeros_(self.timestep_phase)

    def _broadcast_step(
        self,
        step: torch.Tensor | float | int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        step_tensor = torch.as_tensor(step, device=x.device, dtype=x.dtype)
        batch_size = x.shape[0]

        if step_tensor.numel() == 1:
            return step_tensor.reshape(1, 1).expand(batch_size, 1)
        if step_tensor.shape == (batch_size,):
            return step_tensor.unsqueeze(-1)
        if step_tensor.shape == (batch_size, 1):
            return step_tensor
        raise ValueError(
            "step must be scalar, [B], or [B, 1], "
            f"got {tuple(step_tensor.shape)} for batch size {batch_size}"
        )

    def _fourier_features(self, step: torch.Tensor) -> torch.Tensor:
        phase = self.timestep_coeff * step + self.timestep_phase
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        # jax.nn.gelu, used by the reference Flax model, defaults to the tanh
        # approximation.
        return F.gelu(x, approximate="tanh")

    def forward(
        self,
        step: torch.Tensor | float | int,
        x: torch.Tensor,
        score_hint: torch.Tensor,
    ) -> torch.Tensor:
        """Return the control ``u(step, x, score_hint)`` with shape ``[B, D]``."""

        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [B, {self.dim}], got {tuple(x.shape)}")
        if score_hint.shape != x.shape:
            raise ValueError(
                "score_hint must have the same shape as x, "
                f"got {tuple(score_hint.shape)} and {tuple(x.shape)}"
            )

        step_tensor = self._broadcast_step(step, x)
        time_features = self._fourier_features(step_tensor)

        time_state = self.time_state_in(time_features)
        time_state = self._gelu(time_state)
        time_state = self.time_state_out(time_state)

        score_scale = self.time_grad_in(time_features)
        for layer in self.time_grad_hidden:
            score_scale = layer(self._gelu(score_scale))
        score_scale = self.time_grad_out(score_scale)

        state = torch.cat((x, time_state), dim=-1)
        for layer in self.state_hidden:
            state = self._gelu(layer(state))
        state = self.state_out(state)
        state = state.clamp(min=-self.outer_clip, max=self.outer_clip)

        effective_score_hint = (
            torch.zeros_like(score_hint)
            if self.score_hint_mode == "zero"
            else score_hint
        )
        clipped_score = effective_score_hint.clamp(
            min=-self.inner_clip,
            max=self.inner_clip,
        )
        return state + score_scale * clipped_score


class ASBSFourierControl(FourierMLP):
    """Use the GMM-ASBS controller unchanged in the PIS/DIS API.

    PIS and DIS enumerate their transitions from ``T`` down to ``1``.  ASBS
    conditions its :class:`~tr_asbs.model.mlp.FourierMLP` on continuous time
    from zero (source) to one (target), so the adapter only converts

    ``t = 1 - step / T``.

    The third argument is accepted solely for API compatibility and is never
    used.  ``uses_score_hint=False`` also lets the samplers avoid evaluating
    the analytic target score altogether.
    """

    def __init__(
        self,
        dim: int,
        num_steps: int = 128,
        num_layers: int = 4,
        channels: int = 256,
        clip_norm: float | None = 100.0,
        score_hint_mode: str = "none",
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if score_hint_mode != "none":
            raise ValueError(
                "ASBSFourierControl does not consume a score hint; "
                f"score_hint_mode must be 'none', got {score_hint_mode!r}"
            )
        super().__init__(
            dim=dim,
            num_layers=num_layers,
            channels=channels,
            clip_norm=clip_norm,
        )
        self.num_steps = int(num_steps)
        self.score_hint_mode = score_hint_mode
        self.uses_score_hint = False

    def normalized_time(
        self,
        step: torch.Tensor | float | int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        step_tensor = torch.as_tensor(step, device=x.device, dtype=x.dtype)
        if step_tensor.numel() == 1:
            step_tensor = step_tensor.reshape(1).expand(x.shape[0])
        elif step_tensor.shape == (x.shape[0], 1):
            step_tensor = step_tensor.squeeze(-1)
        elif step_tensor.shape != (x.shape[0],):
            raise ValueError(
                "step must be scalar, [B], or [B, 1], "
                f"got {tuple(step_tensor.shape)} for batch size {x.shape[0]}"
            )
        return (1.0 - step_tensor / self.num_steps).clamp(0.0, 1.0)

    def forward(
        self,
        step: torch.Tensor | float | int,
        x: torch.Tensor,
        score_hint: torch.Tensor,
    ) -> torch.Tensor:
        if score_hint.shape != x.shape:
            raise ValueError(
                "score_hint must have the same shape as x, "
                f"got {tuple(score_hint.shape)} and {tuple(x.shape)}"
            )
        return super().forward(self.normalized_time(step, x), x)


class TRIWScoreConditionedControl(nn.Module):
    """Expose :class:`ScoreConditionedControl` through TR-IW's ``u(t, x)`` API.

    PIS numbers its integration steps from ``T`` down to ``1`` while TR-IW
    integrates over continuous time from zero (source) to one (target).  This
    adapter keeps the PIS/DIS control network unchanged and translates the
    inputs as

    ``step = T * (1 - t)`` and ``score_hint = t * target.score(x)``.

    The target score is detached before it reaches the network.  Consequently
    it is side information, as in PIS/DIS, rather than another pathwise loss.
    ``set_potential`` is deliberately separate from construction so Hydra can
    instantiate the model before the already device-local experiment
    potential is attached.
    """

    def __init__(
        self,
        dim: int,
        num_steps: int = 128,
        num_layers: int = 2,
        num_hid: int = 64,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.dim = int(dim)
        self.num_steps = int(num_steps)
        self.backbone = ScoreConditionedControl(
            dim=dim,
            num_layers=num_layers,
            num_hid=num_hid,
            outer_clip=outer_clip,
            inner_clip=inner_clip,
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self._potential = None

    def set_potential(self, potential: object) -> None:
        """Attach the experiment target used to compute the detached score."""

        if not callable(getattr(potential, "score", None)):
            raise TypeError("potential must provide a callable score(x) method")
        self._potential = potential

    def _progress(self, t: torch.Tensor | float | int, x: torch.Tensor) -> torch.Tensor:
        progress = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        batch_size = x.shape[0]
        if progress.numel() == 1:
            return progress.reshape(1, 1).clamp(0.0, 1.0)
        if progress.shape == (batch_size,):
            return progress.unsqueeze(-1).clamp(0.0, 1.0)
        if progress.shape == (batch_size, 1):
            return progress.clamp(0.0, 1.0)
        raise ValueError(
            "t must be scalar, [B], or [B, 1], "
            f"got {tuple(progress.shape)} for batch size {batch_size}"
        )

    def conditioning_inputs(
        self,
        t: torch.Tensor | float | int,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the raw PIS-style step and detached target-score hint."""

        if self._potential is None:
            raise RuntimeError(
                "TRIWScoreConditionedControl requires set_potential() before use"
            )
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [B, {self.dim}], got {tuple(x.shape)}")
        progress = self._progress(t, x)
        with torch.enable_grad():
            target_score = self._potential.score(x.detach())
        if target_score.shape != x.shape:
            raise ValueError(
                "potential.score(x) must have the same shape as x, "
                f"got {tuple(target_score.shape)} and {tuple(x.shape)}"
            )
        score_hint = (progress * target_score).detach()
        raw_step = self.num_steps * (1.0 - progress)
        return raw_step, score_hint

    def forward(
        self,
        t: torch.Tensor | float | int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raw_step, score_hint = self.conditioning_inputs(t, x)
        return self.backbone(raw_step, x, score_hint)
