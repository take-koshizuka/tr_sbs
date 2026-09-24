"""Evaluation utilities shared by variational sampling methods.

The rollout is deliberately evaluated in chunks.  This keeps the peak memory
usage independent of the total number of evaluation samples while retaining
the exact path-wise quantities needed for reverse importance sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Number
from typing import Any, Mapping

import torch
from wandb.sdk.data_types.base_types.wb_value import WBValue


@dataclass(frozen=True)
class EvaluationOutput:
    """Scalar evaluation metrics and the generated terminal samples."""

    metrics: dict[str, float]
    samples: torch.Tensor
    log_weights: torch.Tensor
    media: dict[str, Any] = field(default_factory=dict)


def _scalar_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Convert scalar target metrics to logger-friendly Python floats."""

    metrics: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                metrics[name] = float(value.detach().cpu())
        elif isinstance(value, Number):
            metrics[name] = float(value)
    return metrics


def _wandb_media(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep W&B media while excluding arrays and other non-scalar details."""

    media: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, WBValue):
            media[name] = value
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, WBValue) for item in value)
        ):
            media[name] = value
    return media


def _finite_or_raise(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    count = int((~finite).sum().item())
    total = value.numel()
    raise FloatingPointError(
        f"Non-finite {name} during variational evaluation ({count}/{total} values)."
    )


@torch.no_grad()
def evaluate_variational_method(
    method: Any,
    target: Any,
    *,
    num_samples: int,
    chunk_size: int,
    include_target_details: bool = True,
    visualize: bool = True,
) -> EvaluationOutput:
    """Evaluate reverse ELBO, log-normalizer and ESS from chunked rollouts.

    ``method`` only needs to implement ``rollout(batch_size=...)`` and expose
    the usual :class:`RolloutResult` fields. Keeping this function protocol based
    allows PIS and DIS to share the exact evaluation path.
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    sample_chunks: list[torch.Tensor] = []
    running_chunks: list[torch.Tensor] = []
    stochastic_chunks: list[torch.Tensor] = []
    terminal_chunks: list[torch.Tensor] = []

    remaining = int(num_samples)
    while remaining:
        current_size = min(int(chunk_size), remaining)
        result = method.rollout(
            batch_size=current_size,
            return_trajectory=False,
        )
        _finite_or_raise("samples", result.samples)
        _finite_or_raise("running cost", result.running_cost)
        _finite_or_raise("stochastic cost", result.stochastic_cost)
        _finite_or_raise("terminal cost", result.terminal_cost)

        # Evaluation quantities do not need to remain on the accelerator.
        sample_chunks.append(result.samples.detach().cpu())
        running_chunks.append(result.running_cost.detach().reshape(-1).cpu())
        stochastic_chunks.append(result.stochastic_cost.detach().reshape(-1).cpu())
        terminal_chunks.append(result.terminal_cost.detach().reshape(-1).cpu())
        remaining -= current_size

    samples = torch.cat(sample_chunks, dim=0)
    running_cost = torch.cat(running_chunks, dim=0)
    stochastic_cost = torch.cat(stochastic_chunks, dim=0)
    terminal_cost = torch.cat(terminal_chunks, dim=0)
    neg_elbo = running_cost + terminal_cost
    log_weights = -(running_cost + stochastic_cost + terminal_cost)

    _finite_or_raise("negative ELBO", neg_elbo)
    _finite_or_raise("log weights", log_weights)

    sample_count = log_weights.numel()
    log_count = math.log(sample_count)
    log_z = torch.logsumexp(log_weights, dim=0) - log_count
    # This is the normalized ESS in [1/N, 1], matching the existing project
    # convention and the original variational_sampling_methods implementation.
    log_ess = (
        2.0 * torch.logsumexp(log_weights, dim=0)
        - torch.logsumexp(2.0 * log_weights, dim=0)
        - log_count
    )
    ess = torch.exp(log_ess)

    target_log_prob = target.log_prob(samples)
    target_log_prob = target_log_prob.reshape(sample_count, -1).sum(dim=-1)
    _finite_or_raise("target log probability", target_log_prob)

    reverse_elbo = -neg_elbo.mean()
    metrics = {
        "eval/reverse_elbo": float(reverse_elbo),
        "eval/neg_elbo": float(neg_elbo.mean()),
        "eval/log_z": float(log_z),
        "eval/ess": float(ess),
        # Keep the checkpoint/reporting key used by the ASBS GMM grid.  This
        # is the same normalized ESS, computed from the untempered path
        # importance weights before any display-time normalization.
        "eval/raw_path_ess": float(ess),
        "eval/target_log_prob": float(target_log_prob.mean()),
        # Names used by the source JAX benchmark are retained as aliases so
        # existing result-analysis scripts can compare runs directly.
        "KL/elbo": float(reverse_elbo),
        "logZ/reverse": float(log_z),
        "ESS/reverse": float(ess),
        "other/target_log_prob": float(target_log_prob.mean()),
    }

    media: dict[str, Any] = {}
    if include_target_details and hasattr(target, "log_details"):
        details = target.log_details(
            {
                "data_1": samples,
                "log_RN[h|u]": log_weights,
            },
            visualize=visualize,
        )
        metrics.update(_scalar_metrics(details))
        media.update(_wandb_media(details))

        # Match the ASBS evaluation convention by logging an explicit target
        # sample visualization in addition to the generated-sample figures.
        # Some targets (for example reference-free molecular targets) cannot
        # sample, so keep this capability-gated.
        if (
            visualize
            and getattr(target, "can_sample", True)
            and hasattr(target, "sample")
            and hasattr(target, "visualize")
        ):
            target_samples = target.sample((sample_count,))
            target_media = target.visualize(
                target_samples,
                title="gt_data_1",
            )
            media.update(_wandb_media(target_media))

    return EvaluationOutput(
        metrics=metrics,
        samples=samples,
        log_weights=log_weights,
        media=media,
    )


__all__ = ["EvaluationOutput", "evaluate_variational_method"]
