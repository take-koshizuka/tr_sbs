"""Fabric runtime for trainable variational samplers."""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import hydra
from lightning.fabric import Fabric, seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm.auto import tqdm

from tr_asbs.utils.helper import JointMetricDivergenceStopper

from .evaluation import EvaluationOutput, evaluate_variational_method


_DTYPES = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
}


def _instantiate_logger(logger_cfg: DictConfig | None) -> Any | None:
    """Instantiate a logger after removing merged, unsupported options."""

    if logger_cfg is None:
        return None
    target = str(logger_cfg.get("_target_", ""))
    if not target:
        return None

    logger_cls = hydra.utils.get_class(target)
    options = OmegaConf.to_container(logger_cfg, resolve=True)
    if not isinstance(options, dict):
        raise TypeError("fabric.loggers must be a mapping or null.")
    options.pop("_target_", None)
    options.pop("_delete_", None)

    signature = inspect.signature(logger_cls.__init__)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        valid = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        options = {key: value for key, value in options.items() if key in valid}
    return logger_cls(**options)


def _safe_checkpoint_name(metric_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name)
    return normalized.strip("_") or "metric"


def _resolve_dtype(value: Any) -> torch.dtype:
    name = str(value).lower()
    if name not in _DTYPES:
        supported = ", ".join(sorted(_DTYPES))
        raise ValueError(f"Unsupported dtype {value!r}; choose one of {supported}.")
    return _DTYPES[name]


def _resolve_max_diffusion(cfg: DictConfig) -> float:
    configured = cfg.algorithm.get("max_diffusion")
    if configured is not None:
        value = float(configured)
    else:
        base_sde_cfg = cfg.get("base_sde")
        sigma_max = None if base_sde_cfg is None else base_sde_cfg.get("sigma_max")
        if sigma_max is None:
            raise ValueError(
                "algorithm.max_diffusion is null, but base_sde.sigma_max is "
                "not available as a fallback."
            )
        value = float(sigma_max)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"max_diffusion must be finite and positive, got {value}.")
    return value


class VariationalTrainer:
    """Common optimizer, evaluation and checkpoint runtime.

    Algorithm configs provide a control model and a ``VariationalMethod``
    target. The runtime only relies on ``rollout`` and ``RolloutResult``, so
    PIS and DIS share training and evaluation without sharing dynamics.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dtype = _resolve_dtype(cfg.get("dtype", "float32"))

        logger_cfg = cfg.fabric.get("loggers")
        logger_target = (
            "" if logger_cfg is None else str(logger_cfg.get("_target_", "")).lower()
        )
        self._wandb_logging = "wandb" in logger_target
        self.logger = _instantiate_logger(logger_cfg)
        fabric_cfg = OmegaConf.create(OmegaConf.to_container(cfg.fabric, resolve=True))
        fabric_cfg.pop("loggers", None)
        self.fabric: Fabric = hydra.utils.instantiate(
            fabric_cfg,
            loggers=self.logger,
        )
        self.fabric.launch()

        self.rank_seed = int(cfg.seed) ^ int(self.fabric.global_rank)
        seed_everything(self.rank_seed, workers=False)

        if self.logger is not None:
            hyperparameters = OmegaConf.to_container(cfg, resolve=True)
            try:
                self.logger.log_hyperparams(hyperparameters)
            except NotImplementedError:
                # Fabric's lightweight CSVLogger intentionally implements
                # scalar logging only.
                pass

        if self.fabric.world_size != 1:
            raise NotImplementedError(
                "The variational runtime currently supports one device. "
                "Single-device execution is required for exact RNG resume."
            )

        with self.fabric.init_module():
            self.target = hydra.utils.instantiate(
                cfg.potential,
                device=str(self.fabric.device),
            )
            control = hydra.utils.instantiate(
                cfg.algorithm.model,
                dim=int(self.target.dim),
            )

        if bool(cfg.get("compile_control", False)):
            compile_options = OmegaConf.to_container(
                cfg.get("compile_options", {}), resolve=True
            )
            if not isinstance(compile_options, dict):
                raise TypeError("compile_options must be a mapping.")
            control = torch.compile(control, **compile_options)

        max_diffusion = _resolve_max_diffusion(cfg)
        schedule_cfg = cfg.algorithm.get("schedule", {})
        schedule_power = schedule_cfg.get(
            "pow",
            schedule_cfg.get("power", 2.0),
        )
        method_kwargs: dict[str, Any] = {}
        if cfg.algorithm.get("init_std") is not None:
            method_kwargs["init_std"] = float(cfg.algorithm.init_std)
        self.method = hydra.utils.instantiate(
            cfg.algorithm.method,
            control=control,
            target=self.target,
            num_steps=int(cfg.algorithm.num_steps),
            max_diffusion=max_diffusion,
            sigma_min=float(schedule_cfg.get("sigma_min", 0.01)),
            s=float(schedule_cfg.get("s", 0.008)),
            power=float(schedule_power),
            noise_clip=float(cfg.algorithm.get("noise_clip", 4.0)),
            **method_kwargs,
        )
        self.method.to(dtype=self.dtype)

        optimizer = hydra.utils.instantiate(
            cfg.algorithm.optimizer,
            params=self.method.parameters(),
        )
        self.method, self.optimizer = self.fabric.setup(
            self.method,
            optimizer,
        )

        if bool(getattr(self.target, "is_mean_free", False)):
            raise ValueError(
                "Variational training currently supports flat vector targets "
                "only; mean-free particle targets are out of scope."
            )

        self.iteration = 0
        self.best_metrics: dict[str, float] = {}
        self.evaluation_count = 0
        self.divergence_stopper = JointMetricDivergenceStopper(
            cfg.get("divergence_early_stop", {})
        )
        self.early_stop_requested = False
        self._resume_loaded = False
        self.checkpoint_directory = Path(str(cfg.checkpoint.directory)).expanduser()
        if self.fabric.global_rank == 0:
            self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.fabric.barrier()

    @property
    def control(self) -> torch.nn.Module:
        """Return the configured control module (possibly Fabric-wrapped)."""

        return self.method.control

    def _validate_rollout(self, result: Any, context: str) -> None:
        tensors = {
            "samples": result.samples,
            "running_cost": result.running_cost,
            "stochastic_cost": result.stochastic_cost,
            "terminal_cost": result.terminal_cost,
            "negative ELBO": result.neg_elbo,
        }
        problems = []
        for name, value in tensors.items():
            finite = torch.isfinite(value)
            if not bool(finite.all()):
                problems.append(f"{name}={int((~finite).sum().item())}")
        if problems:
            diagnostic = ", ".join(problems)
            raise FloatingPointError(
                f"Non-finite values during {context} at iteration "
                f"{self.iteration}: {diagnostic}."
            )

    def _sanitize_gradients(self) -> int:
        """Replace individual NaN/Inf gradient entries and return their count."""

        count = 0
        with torch.no_grad():
            for parameter in self.method.parameters():
                if parameter.grad is None:
                    continue
                invalid = ~torch.isfinite(parameter.grad)
                invalid_count = int(invalid.sum().item())
                if invalid_count:
                    parameter.grad.masked_fill_(invalid, 0.0)
                    count += invalid_count
        return count

    def training_step(self) -> dict[str, float]:
        """Take one optimizer step, optionally using gradient microbatches."""

        self.method.train()
        batch_size = int(self.cfg.algorithm.batch_size)
        configured_microbatch = self.cfg.algorithm.get("microbatch_size")
        microbatch_size = (
            batch_size
            if configured_microbatch is None
            else min(int(configured_microbatch), batch_size)
        )
        if batch_size <= 0 or microbatch_size <= 0:
            raise ValueError("batch_size and microbatch_size must be positive.")

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.fabric.device, dtype=self.dtype)
        processed = 0
        while processed < batch_size:
            current_size = min(microbatch_size, batch_size - processed)
            result = self.method.rollout(
                batch_size=current_size,
                return_trajectory=False,
            )
            self._validate_rollout(result, "training")
            microbatch_loss = result.neg_elbo.mean()
            weight = current_size / batch_size
            self.fabric.backward(microbatch_loss * weight)
            total_loss += microbatch_loss.detach() * weight
            processed += current_size

        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(
                f"Non-finite loss at iteration {self.iteration}: "
                f"{float(total_loss.detach().cpu())}."
            )

        nonfinite_gradients = self._sanitize_gradients()
        clip_value = self.cfg.algorithm.get("grad_clip")
        if clip_value is not None:
            self.fabric.clip_gradients(
                self.method,
                self.optimizer,
                clip_val=float(clip_value),
                error_if_nonfinite=False,
            )
        self.optimizer.step()

        return {
            "train/neg_elbo": float(total_loss.detach().cpu()),
            "train/nonfinite_gradients": float(nonfinite_gradients),
        }

    def evaluate(
        self,
        *,
        num_samples: int | None = None,
        visualize: bool | None = None,
    ) -> EvaluationOutput:
        """Run chunked reverse evaluation and restore the training mode."""

        was_training = self.method.training
        self.method.eval()
        try:
            return evaluate_variational_method(
                self.method,
                self.target,
                num_samples=(
                    int(self.cfg.evaluation.num_samples)
                    if num_samples is None
                    else int(num_samples)
                ),
                chunk_size=int(self.cfg.evaluation.chunk_size),
                include_target_details=bool(
                    self.cfg.evaluation.get("include_target_details", True)
                ),
                visualize=(
                    bool(self.cfg.get("visualize", True))
                    if visualize is None
                    else bool(visualize)
                ),
            )
        finally:
            self.method.train(was_training)

    def _checkpoint_state(self) -> dict[str, Any]:
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        return {
            "model": self.method,
            "optimizer": self.optimizer,
            "iteration": int(self.iteration),
            "best_metrics": dict(self.best_metrics),
            "evaluation_count": int(self.evaluation_count),
            "divergence_state": {
                "best_primary": self.divergence_stopper.best_primary,
                "best_secondary": self.divergence_stopper.best_secondary,
                "bad_count": self.divergence_stopper.bad_count,
                "nonfinite_count": self.divergence_stopper.nonfinite_count,
                "reason": self.divergence_stopper.reason,
            },
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": cuda_rng_state,
        }

    def save_checkpoint(self, filename: str = "last.pt") -> Path:
        """Save model, optimizer, progress and exact torch RNG state."""

        path = self.checkpoint_directory / filename
        self.fabric.save(path, self._checkpoint_state())
        return path

    def load_checkpoint(self, checkpoint: str | os.PathLike[str]) -> Path:
        """Load an explicitly selected checkpoint and restore its RNG state."""

        path = Path(checkpoint).expanduser()
        if path.is_dir():
            path = path / "last.pt"
        if not path.is_absolute():
            path = Path(hydra.utils.to_absolute_path(str(path)))
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        remainder = self.fabric.load(
            path,
            {
                "model": self.method,
                "optimizer": self.optimizer,
            },
        )
        self.iteration = int(remainder.pop("iteration"))
        best_metrics = remainder.pop("best_metrics", {})
        self.best_metrics = {
            str(name): float(value) for name, value in best_metrics.items()
        }
        self.evaluation_count = int(remainder.pop("evaluation_count", 0))
        divergence_state = remainder.pop("divergence_state", {})
        for name in (
            "best_primary",
            "best_secondary",
            "bad_count",
            "nonfinite_count",
            "reason",
        ):
            if name in divergence_state:
                setattr(self.divergence_stopper, name, divergence_state[name])

        cpu_rng_state = remainder.pop("cpu_rng_state")
        torch.set_rng_state(cpu_rng_state.cpu())
        cuda_rng_state = remainder.pop("cuda_rng_state_all", [])
        if torch.cuda.is_available() and cuda_rng_state:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        self._resume_loaded = True
        return path

    def _update_best_checkpoints(self, metrics: dict[str, float]) -> None:
        monitor_values = list(self.cfg.get("monitor_values", []))
        if not monitor_values:
            monitor_values = ["eval/neg_elbo"]

        configured_modes = self.cfg.get("monitor_modes", {})
        default_mode = str(self.cfg.checkpoint.get("mode", "min")).lower()
        if default_mode not in {"min", "max"}:
            raise ValueError("checkpoint.mode must be 'min' or 'max'.")

        for metric_name in monitor_values:
            metric_name = str(metric_name)
            mode = str(configured_modes.get(metric_name, default_mode)).lower()
            if mode not in {"min", "max"}:
                raise ValueError(
                    f"Monitor mode for {metric_name!r} must be 'min' or 'max', "
                    f"got {mode!r}."
                )
            if metric_name not in metrics:
                available = ", ".join(sorted(metrics))
                raise KeyError(
                    f"Monitored metric {metric_name!r} was not produced. "
                    f"Available scalar metrics: {available}"
                )
            value = float(metrics[metric_name])
            previous = self.best_metrics.get(metric_name)
            improved = previous is None or (
                value < previous if mode == "min" else value > previous
            )
            if improved:
                self.best_metrics[metric_name] = value
                safe_name = _safe_checkpoint_name(metric_name)
                self.save_checkpoint(f"{safe_name}_best.pt")

    def _log(self, metrics: dict[str, float]) -> None:
        self.fabric.log_dict(metrics, step=self.iteration)

    def _log_media(self, media: dict[str, Any]) -> None:
        """Log target figures only through a W&B-capable logger."""

        if media and self._wandb_logging and self.logger is not None:
            self.logger.log_metrics(media, step=self.iteration)

    def _set_summary(self, name: str, value: Any) -> None:
        """Set a W&B summary value without coupling the CSV test logger to W&B."""

        if not self._wandb_logging or self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        summary = getattr(experiment, "summary", None)
        if summary is not None:
            summary[name] = value

    def _process_evaluation(
        self,
        output: EvaluationOutput,
        *,
        update_checkpoints: bool,
    ) -> None:
        """Log one evaluation and apply the shared ASBS stopping contract."""

        self._log(output.metrics)
        self._log_media(output.media)
        if update_checkpoints:
            self._update_best_checkpoints(output.metrics)

        if self.divergence_stopper.update(
            output.metrics,
            self.evaluation_count,
        ):
            self.early_stop_requested = True
            reason = str(self.divergence_stopper.reason)
            print(f"Early stopping at iteration {self.iteration}: {reason}")
            self._log(
                {
                    "early_stop/triggered": 1.0,
                    "early_stop/iteration": float(self.iteration),
                }
            )
            self._set_summary("early_stop/triggered", True)
            self._set_summary("early_stop/iteration", int(self.iteration))
            self._set_summary("early_stop/reason", reason)
        self.evaluation_count += 1

    def _report_selected_checkpoints(self) -> dict[str, dict[str, Any]]:
        """Reevaluate retained checkpoints with the common final sample budget."""

        monitor_values = list(self.cfg.get("monitor_values", []))
        if not monitor_values:
            monitor_values = ["eval/neg_elbo"]
        selections = [("final", self.checkpoint_directory / "last.pt", None)]
        selections.extend(
            (
                f"best_{_safe_checkpoint_name(str(metric_name))}",
                self.checkpoint_directory
                / f"{_safe_checkpoint_name(str(metric_name))}_best.pt",
                str(metric_name),
            )
            for metric_name in monitor_values
        )

        final_num_samples = int(
            self.cfg.evaluation.get(
                "final_num_samples",
                self.cfg.evaluation.num_samples,
            )
        )
        if final_num_samples <= 0:
            raise ValueError("evaluation.final_num_samples must be positive.")
        report_seed = int(self.cfg.evaluation.get("report_seed", 12345))
        final_iteration = int(self.iteration)
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        artifact_directory = self.checkpoint_directory / "evaluation_artifacts"
        if self.fabric.global_rank == 0:
            artifact_directory.mkdir(parents=True, exist_ok=True)
        self.fabric.barrier()

        records: dict[str, dict[str, Any]] = {}
        try:
            for selection, checkpoint_path, monitor_metric in selections:
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Selected checkpoint is missing: {checkpoint_path}"
                    )
                remainder = self.fabric.load(
                    checkpoint_path,
                    {"model": self.method},
                )
                selected_iteration = int(remainder["iteration"])
                checkpoint_best_metrics = remainder.get("best_metrics", {})
                online_selection_value = (
                    None
                    if monitor_metric is None
                    else float(checkpoint_best_metrics[monitor_metric])
                )

                torch.manual_seed(report_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(report_seed)
                output = self.evaluate(
                    num_samples=final_num_samples,
                    visualize=bool(self.cfg.get("visualize", True)),
                )
                records[selection] = {
                    "checkpoint": str(checkpoint_path),
                    "selected_iteration": selected_iteration,
                    "online_selection_metric": monitor_metric,
                    "online_selection_value": online_selection_value,
                    "num_samples": final_num_samples,
                    "report_seed": report_seed,
                    "metrics": dict(output.metrics),
                }

                prefixed_metrics = {
                    f"report/{selection}/{name}": value
                    for name, value in output.metrics.items()
                }
                self._log(prefixed_metrics)
                for name, value in output.metrics.items():
                    self._set_summary(f"report/{selection}/{name}", value)
                self._set_summary(
                    f"report/{selection}/selected_iteration",
                    selected_iteration,
                )

                if output.media and self._wandb_logging and self.logger is not None:
                    self.logger.log_metrics(
                        {
                            f"report/{selection}/{name}": value
                            for name, value in output.media.items()
                        },
                        step=final_iteration,
                    )
                if self.fabric.global_rank == 0:
                    torch.save(
                        {
                            "samples": output.samples,
                            "log_weights": output.log_weights,
                            "selected_iteration": selected_iteration,
                            "metrics": dict(output.metrics),
                        },
                        artifact_directory / f"{selection}.pt",
                    )
        finally:
            self.fabric.load(
                self.checkpoint_directory / "last.pt",
                {"model": self.method},
            )
            self.iteration = final_iteration
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available() and cuda_rng_state:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        if self.fabric.global_rank == 0:
            report_path = self.checkpoint_directory / "checkpoint_evaluations.json"
            control = self.method.control
            report_path.write_text(
                json.dumps(
                    {
                        "algorithm": str(self.cfg.algorithm.name),
                        "score_hint_mode": str(
                            getattr(
                                control,
                                "score_hint_mode",
                                self.cfg.algorithm.model.get(
                                    "score_hint_mode",
                                    "provided",
                                ),
                            )
                        ),
                        "uses_score_hint": bool(
                            getattr(control, "uses_score_hint", True)
                        ),
                        "control_class": (
                            f"{control.__class__.__module__}."
                            f"{control.__class__.__qualname__}"
                        ),
                        "control_parameters": sum(
                            parameter.numel() for parameter in control.parameters()
                        ),
                        "seed": int(self.cfg.seed),
                        "final_iteration": final_iteration,
                        "early_stopped": bool(self.early_stop_requested),
                        "early_stop_reason": self.divergence_stopper.reason,
                        "selections": records,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            self._set_summary("report/path", str(report_path))
        self.fabric.barrier()
        return records

    def fit(self) -> dict[str, float]:
        """Train through the configured iteration count and return final metrics."""

        resume_from = self.cfg.get("resume_from")
        if resume_from and not self._resume_loaded:
            self.load_checkpoint(str(resume_from))

        total_iterations = int(self.cfg.algorithm.iters)
        if total_iterations < 0:
            raise ValueError("algorithm.iters must be non-negative.")
        if self.iteration > total_iterations:
            raise ValueError(
                f"Checkpoint iteration {self.iteration} exceeds configured "
                f"algorithm.iters={total_iterations}."
            )

        num_evaluations = int(self.cfg.evaluation.num_evaluations)
        if num_evaluations < 0:
            raise ValueError("evaluation.num_evaluations must be non-negative.")
        scheduled_evaluations = min(num_evaluations, total_iterations)
        evaluation_steps = (
            {
                math.ceil(index * total_iterations / scheduled_evaluations)
                for index in range(1, scheduled_evaluations + 1)
            }
            if scheduled_evaluations
            else set()
        )
        log_interval = max(int(self.cfg.get("log_interval", 1)), 1)
        checkpoint_interval = int(self.cfg.checkpoint.interval)
        progress = tqdm(
            total=total_iterations,
            initial=self.iteration,
            desc=f"{self.cfg.algorithm.name} training",
            disable=(
                self.fabric.global_rank != 0
                or not bool(self.cfg.get("progress_bar", True))
            ),
        )

        final_metrics: dict[str, float] = {}
        last_evaluation_iteration = -1
        run_status = "failed"
        try:
            if (
                self.iteration == 0
                and num_evaluations > 0
                and bool(self.cfg.evaluation.get("evaluate_initial_state", True))
            ):
                output = self.evaluate()
                final_metrics = output.metrics
                self._process_evaluation(
                    output,
                    update_checkpoints=bool(
                        self.cfg.get("monitor_initial_state", True)
                    ),
                )
                self.save_checkpoint("last.pt")
                last_evaluation_iteration = self.iteration

            while self.iteration < total_iterations and not self.early_stop_requested:
                started = time.perf_counter()
                train_metrics = self.training_step()
                self.iteration += 1
                train_metrics["stats/iteration_seconds"] = time.perf_counter() - started
                if (
                    self.iteration % log_interval == 0
                    or self.iteration == total_iterations
                ):
                    self._log(train_metrics)

                should_evaluate = self.iteration in evaluation_steps
                if should_evaluate:
                    output = self.evaluate()
                    final_metrics = output.metrics
                    self._process_evaluation(output, update_checkpoints=True)
                    self.save_checkpoint("last.pt")
                    last_evaluation_iteration = self.iteration
                    if self.early_stop_requested:
                        progress.update(1)
                        break

                if (
                    checkpoint_interval > 0
                    and self.iteration % checkpoint_interval == 0
                ):
                    self.save_checkpoint(f"step_{self.iteration:08d}.pt")
                    if not should_evaluate:
                        self.save_checkpoint("last.pt")
                progress.update(1)

            if last_evaluation_iteration != self.iteration:
                output = self.evaluate()
                final_metrics = output.metrics
                self._process_evaluation(output, update_checkpoints=True)
            self.save_checkpoint("last.pt")
            if bool(self.cfg.evaluation.get("report_checkpoints", True)):
                self._report_selected_checkpoints()
            run_status = "success"
        finally:
            progress.close()
            if self.logger is not None:
                self.logger.finalize(run_status)

        return final_metrics


__all__ = ["VariationalTrainer"]
