# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import os
import math
import ase.io
import hydra
import torch
import numpy as np
import wandb
from lightning.fabric import seed_everything
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize_scalar
import json
import hashlib
import uuid
from typing import Any

from omegaconf import OmegaConf

from tr_asbs.process.sde import ControlledSDE, ControlledFBSDE
from tr_asbs.process.reference import LinearGaussianReference
from tr_asbs.utils.training import EMA, copy_model
from tr_asbs.utils.gradient_clip import clip_gradients_with_overflow_fallback
from tr_asbs.data.buffer import BatchBuffer
from tr_asbs.utils.eval_logweight import compute_raw_path_ess, compute_reverse_ess, normalize_log_weights
from tr_asbs.utils.helper import Checkpointer, CheckpointConfig, JointMetricDivergenceStopper

import time
import torch._dynamo

def stable_group_id(cfg: dict, exclude=("seed", "name")):
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    d = {k: cfg[k] for k in sorted(cfg.keys()) if k not in exclude}
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]

class GammaASBSModule:
    """Module for ASBS training."""

    TASKS: list[str] = ["adjoint_matching", "corrector_matching"]

    ############################################################################
    # Initialization
    ############################################################################

    def __init__(self, cfg: DictConfig):
        super().__init__()
        from tr_asbs.experiment.reference_config import configure_reference
        configure_reference(cfg)
        self.cfg = cfg
        # group_id determines checkpoint_directory (see below). A fresh random_hash
        # is appended on every process launch by default -- meaning `resume=true`
        # can never actually find a previous process's checkpoint directory (it's
        # always new/empty), silently discarding all prior training progress and
        # restarting from scratch (epoch 0, fresh weights) while still appearing to
        # "resume" if the wandb run id is separately pinned. Pass an explicit
        # checkpoint_group_id (the group_id printed/logged by the run being resumed,
        # i.e. the checkpoint subdirectory name under checkpoint_directory) to
        # actually reuse the same checkpoint directory across restarts.
        explicit_group_id = getattr(cfg, "checkpoint_group_id", None)
        random_hash = uuid.uuid4().hex[:8]
        self.group_id = explicit_group_id or (str(stable_group_id(cfg, exclude=("name"))) + f"{random_hash}")
        logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
        if "wandb" in logger_target and "group" in cfg.fabric.loggers:
            cfg.fabric.loggers.group = stable_group_id(cfg)
        
        wandb_logger = hydra.utils.instantiate(cfg.fabric.loggers)
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(full_cfg)
    
        # step size
        self.gamma = cfg.gamma
        self.V0 = 1e-9
        assert 0 <= self.gamma <= 1
        self.gamma_schedule = cfg.gamma_schedule
        assert self.gamma_schedule in ["fixed", "trust_region"]
        self.trust_region_bound = cfg.trust_region_bound
        self.iw_clip_value = cfg.iw_clip_value

        # FC (u-update) loss: "cross_entropy" (default, importance-weighted forward-KL,
        # see adjoint_matching_step) or "log_variance" (importance-weight-free VarGrad-
        # style alternative, see adjoint_matching_log_variance_step). Gamma is still
        # computed via optimize_gamma() either way -- only the u-loss itself changes.
        self.fc_loss_mode = getattr(cfg, "fc_loss_mode", "cross_entropy")
        assert self.fc_loss_mode in ["cross_entropy", "log_variance", "path_kl"]
        self.fc_backward_mode = getattr(cfg, "fc_backward_mode", "native")
        if self.fc_backward_mode not in ("native", "loss_scaled_fp64_clip"):
            raise ValueError("Invalid fc_backward_mode")
        if self.fc_backward_mode != "native" and self.fc_loss_mode != "log_variance":
            raise ValueError("Scaled controller backward requires the LV loss")
        self.fc_backward_statistics = dict(attempts=0, updates=0, loss_nonfinite=0,
            gradient_nonfinite=0, minimum_scale=1., maximum_scale=0.)
        self.bc_loss_weight = getattr(cfg, "bc_loss_weight", "legacy")
        self.fc_loss_weight = getattr(cfg, "fc_loss_weight", "legacy")
        if self.fc_loss_weight not in ("legacy", "diffusion_energy"):
            raise ValueError("Invalid fc_loss_weight")
        if self.bc_loss_weight not in ("legacy", "none", "reference_variance"):
            raise ValueError("Invalid bc_loss_weight")
        self.bridge_time_epsilon = float(getattr(cfg, "bridge_time_epsilon", 0.001))
        if not 0 < self.bridge_time_epsilon < 0.5:
            raise ValueError("bridge_time_epsilon must be in (0, .5)")

        # Variance-reduction trick from the reference ASBS repo (adjoint_matcher.
        # duplicates in facebookresearch/adjoint_samplers): each freshly simulated
        # trajectory is stored in the buffer `adjoint_buffer_duplicates` times
        # (default 1 = no-op) instead of once. This costs no extra simulation --
        # only extra buffer slots -- but means the SAME (X_0, X_1) pair gets
        # resampled with more independent bridge times (adjoint_matching_step's
        # random `t ~ U(0,T)` draw is per buffer *access*, not per trajectory)
        # over the course of training, at the cost of unique-trajectory diversity
        # for a given buffer capacity.
        self.adjoint_buffer_duplicates = int(getattr(cfg, "adjoint_buffer_duplicates", 1))

        # Per-time-anchored CE target (default False = original behavior):
        # adjoint_matching_step's target is normally anchored at control_1 =
        # u_i(X_T, T) for ALL THREE evaluation times {0, t, T} -- fine at T
        # (target reduces to a no-op there, since the network already equals
        # u_i at the start of the outer loop), but as gamma -> 0 the SAME
        # T-anchored value becomes the target at t=0/t too, where the frozen
        # policy's actual output is generally different (a correct control
        # varies with time/state). That's an actively corrupting signal, not
        # a neutral one, for as long as gamma stays near 0 -- empirically this
        # is exactly when Eq-EMD2/W_2 measurably WORSEN rather than merely
        # stalling (see the oscillation analysis on run 2agyos3s). Enabling
        # this anchors each of the three terms to the frozen policy's OWN
        # output at that SAME time (control_0/control_t/control_1 instead of
        # control_1 uniformly), which reduces to gamma=1 -> terminal_cost
        # (unchanged, still ASBS-equivalent, verified algebraically) and
        # gamma=0 -> "reproduce your own output at this specific time" (a
        # genuine no-op at every tau, not just at T).
        self.per_time_anchor_target = bool(getattr(cfg, "per_time_anchor_target", False))

        # Gamma floor (default 0.0 = no-op, original behavior): optimize_gamma()
        # solves the dual exactly per outer loop from a noisy, finite-sample MC
        # estimate of the heavy-tailed LJ log-weight distribution -- empirically
        # this makes gamma swing between a healthy ~0.1-0.2 and a collapsed
        # ~1e-4 from one outer loop to the next, rather than settling. Especially
        # combined with the per_time_anchor_target issue above, letting gamma
        # actually reach ~0 is measurably harmful, not just wasteful. Clamping
        # gamma to never go below gamma_floor prevents the worst of that
        # collapse, at the cost of occasionally exceeding the nominal
        # trust_region_bound when the dual solve's true optimum is below the floor.
        self.gamma_floor = float(getattr(cfg, "gamma_floor", 0.0))

        # Optional autocast dtype for the two EGNN forward calls inside
        # adjoint_matching_log_variance_step (the batch_size*integrator_steps
        # flattened forward pass, the dominant memory/compute cost of the dense
        # log-variance loss). Only the network forward calls are cast -- the
        # surrounding Girsanov log-ratio arithmetic (variance division, squared
        # norms) stays fp32, since that's where numerical blowups from the LJ
        # potential's singular repulsive core would actually bite.
        logvar_amp_dtype = getattr(cfg, "logvar_amp_dtype", None)
        self.logvar_amp_dtype = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[logvar_amp_dtype]
            if logvar_amp_dtype else None
        )

        # Weight-checkpoint rollback (opt-in): the trust-region adaptive scheme only
        # ever adjusts epsilon *going forward* -- once an overly-aggressive gamma
        # step has pushed the controller into a bad region (RN[h|u]/ess collapses
        # and, empirically, never recovers on LJ-38), shrinking epsilon afterward
        # doesn't undo the damage already baked into the weights. This snapshots
        # controller+corrector (+ their optimizer states) whenever ESS looks
        # healthy, and restores that snapshot if ESS collapses below threshold for
        # `weight_rollback_patience` consecutive detail-logging evals -- a direct
        # fix for the collapse observed in run yh5mnga0 (lj38_logvar/lj38_ce).
        self.weight_rollback_enabled = getattr(cfg, "weight_rollback_enabled", False)
        self.weight_rollback_ess_threshold = getattr(cfg, "weight_rollback_ess_threshold", 1e-3)
        self.weight_rollback_patience = getattr(cfg, "weight_rollback_patience", 1)
        self._weight_rollback_bad_count = 0
        self._weight_rollback_checkpoint = None
        self._weight_rollback_count = 0

        # Adaptive trust-region radius (opt-in): mirrors classical trust-region
        # methods (Conn, Gould & Toint), which expand the radius when steps are
        # starved for progress and shrink it when the trust region is essentially
        # inactive, rather than keeping epsilon fixed for the whole run.
        self.trust_region_adaptive = getattr(cfg, "trust_region_adaptive", False)
        self.trust_region_gamma_low = getattr(cfg, "trust_region_gamma_low", 0.05)
        self.trust_region_gamma_high = getattr(cfg, "trust_region_gamma_high", 0.9)
        self.trust_region_growth_factor = getattr(cfg, "trust_region_growth_factor", 3.0)
        self.trust_region_shrink_factor = getattr(cfg, "trust_region_shrink_factor", 0.5)
        self.trust_region_patience = getattr(cfg, "trust_region_patience", 1)
        self.trust_region_bound_min = getattr(cfg, "trust_region_bound_min", 1e-3)
        self.trust_region_bound_max = getattr(cfg, "trust_region_bound_max", 1000.0)
        # Energy-based stability check: if logged energy blows up past this multiple of
        # the last known-good baseline, treat the current epsilon as unsafe (roll back +
        # cooldown) rather than continuing to react to gamma alone.
        self.trust_region_energy_blowup_factor = getattr(cfg, "trust_region_energy_blowup_factor", 3.0)
        self._trust_region_stuck_low_count = 0
        self._trust_region_stuck_high_count = 0
        self._trust_region_growth_cooldown = 0
        self._trust_region_reference_energy = None
        self._last_logged_energy = None
        # Bisection bounds: loosest epsilon known to give gamma stuck near 0 (a valid
        # lower bound on the "good" range) and tightest epsilon known to give gamma
        # near 1 / trust region inactive (a valid upper bound). Once both are known,
        # we search their geometric mean instead of blindly multiplying by a fixed
        # factor, so the search narrows in on the (possibly narrow) transition window
        # rather than jumping straight over it.
        self._trust_region_low_bound = None
        self._trust_region_high_bound = None

        # Lightning fabric setup
        fabric_overrides = {"loggers": wandb_logger}
        if cfg.fabric.get("devices", 1) not in (1, "1") and cfg.fabric.get("strategy", "auto") == "auto":
            # Not every model parameter participates in every forward pass here
            # (e.g. the u-step only exercises part of the controller/corrector),
            # which trips DDP's default "all parameters must receive a gradient
            # every iteration" bucketing assumption once devices > 1.
            from lightning.fabric.strategies import DDPStrategy

            fabric_overrides["strategy"] = DDPStrategy(find_unused_parameters=True)
        self.fabric = hydra.utils.instantiate(cfg.fabric, **fabric_overrides)
        self.fabric.launch()
        self.rank_seed = cfg.seed ^ self.fabric.global_rank
        seed_everything(self.rank_seed, workers=False)

        with self.fabric.init_module():
            self.potential = hydra.utils.instantiate(cfg.potential, device=self.fabric.device)
            # Models
            controller = hydra.utils.instantiate(cfg.controller)
            corrector = hydra.utils.instantiate(cfg.corrector)
            for model in (controller, corrector):
                set_potential = getattr(model, "set_potential", None)
                if callable(set_potential):
                    set_potential(self.potential)
            if cfg.ema_decay is not None:
                self.controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)
                self.corrector = EMA(corrector, cfg.ema_decay, cfg.ema_start_step)
            else:
                self.controller = controller
                self.corrector = corrector
                
            # Buffers
            self.adjoint_buffer = hydra.utils.instantiate(cfg.adjoint_buffer)
            self.corrector_buffer = hydra.utils.instantiate(cfg.corrector_buffer)
            
            # Process and integration
            self.source = hydra.utils.instantiate(cfg.source, device=self.fabric.device)
            self.base_sde = hydra.utils.instantiate(cfg.base_sde, is_mean_free=self.potential.is_mean_free)
            for model in (self.controller, self.corrector):
                for module in list(model.modules()):
                    setter = getattr(module, "set_reference", None)
                    if callable(setter):
                        setter(self.base_sde)
            self.sde = ControlledFBSDE(
                base_sde=self.base_sde,
                u_controller=self.controller, 
                h_controller=self.corrector
            )
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            if isinstance(self.base_sde, LinearGaussianReference):
                if self.fc_loss_mode == "cross_entropy" and not (
                    self.gamma_schedule == "fixed" and self.gamma == 1):
                    raise ValueError("Linear-reference TR updates require path_kl or log_variance; CE supports fixed gamma=1 only")
                if self.base_sde.kind == "vp":
                    expected = float(self.base_sde.stationary_scale.square())
                    covariance = self.source.cov
                    if self.potential.is_mean_free:
                        n, d = self.source.n_particles, self.source.spatial_dim
                        projector = torch.eye(n, device=covariance.device)-torch.ones(n, n, device=covariance.device)/n
                        expected_covariance = expected*torch.kron(projector, torch.eye(d, device=covariance.device))
                    else:
                        expected_covariance = expected*torch.eye(covariance.shape[0], device=covariance.device)
                    if not torch.allclose(covariance, expected_covariance.to(covariance)):
                        raise ValueError("VP reference and Gaussian source scales must match")
            if self.fc_loss_mode == "path_kl":
                if any(type(m).__name__ == "TRIWScoreConditionedControl"
                       for model in (self.controller, self.corrector) for m in model.modules()):
                    raise ValueError("path_kl needs state derivatives; detached score hints are supported by log_variance instead")
                # Straight-through output clipping changes state Jacobians and
                # invalidates the pathwise derivative. Use actual derivatives.
                for model in (self.controller, self.corrector):
                    for module in model.modules():
                        if hasattr(module, "clip_gradient_mode"):
                            module.clip_gradient_mode = "exact"

            # Potential and cost function
            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential, corrector=self.corrector
            )
            
            self.prev_integrator = deepcopy(self.integrator)
            self.prev_corrector = deepcopy(self.corrector)

        # Corrector-side trust-region damping (opt-in; default "none" preserves
        # existing behavior exactly). corrector_matching_step is otherwise a
        # fully unconstrained per-outer-loop score-matching refit -- no gamma,
        # no KL bound -- so h can move arbitrarily far from h_i every outer
        # loop even when the controller's own trust region is behaving exactly
        # as designed. This directly violates the convergence theorem's own
        # precondition (paper.tex line ~1094: Var_{P^{u_i}}(log dP^{h_i}/dP^{u_i})
        # must stay bounded across iterations) since nothing keeps h from
        # drifting arbitrarily far from its own previous iterate.
        #
        # "reuse_u_gamma"/"independent" damp h relative to h_i (its OWN
        # previous iterate) -- empirically this made no difference at all
        # (tracks the undamped baseline's trajectory almost exactly), because
        # the convergence theorem's actual precondition (paper.tex Theorem
        # \ref{thm:convergence_rate}) bounds Var_{P^{u_i}}(log dP^h/dP^u) --
        # the divergence between h and u DIRECTLY -- not h's divergence from
        # its own past. "joint_reuse"/"joint_independent" target that
        # directly: they damp h's regression target toward the CONTROLLER's
        # current output (not h_i), using the same Girsanov-KL-of-diffusions
        # approximation (KL(P^a|P^b) ≈ (1/2) E[∫||a-b||^2 dt]) but with u
        # substituted for h_i as the reference.
        #   "none":              unchanged behavior (plain score-matching to ground truth).
        #   "reuse_u_gamma":      mix toward h_i using u's own gamma.
        #   "independent":        mix toward h_i using an independently-derived gamma_h.
        #   "joint_reuse":        mix toward u's current output using u's own gamma.
        #   "joint_independent":  mix toward u's current output using an independently-
        #                         derived gamma_h (KL(h_{i+1}|u) <= corrector_trust_region_bound).
        self.corrector_gamma_mode = getattr(cfg, "corrector_gamma_mode", "none") or "none"
        assert self.corrector_gamma_mode in (
            "none", "reuse_u_gamma", "independent", "joint_reuse", "joint_independent",
        )
        self.corrector_trust_region_bound = getattr(cfg, "corrector_trust_region_bound", None)
        if self.corrector_trust_region_bound is None:
            self.corrector_trust_region_bound = cfg.trust_region_bound
        self.gamma_h = 1.0
        if isinstance(self.base_sde, LinearGaussianReference) and self.corrector_gamma_mode != "none":
            raise ValueError("Corrector gamma's legacy MSE proxy is not an OU/VP kernel KL; use corrector_gamma_mode=none")

        # Opt-in (default off, preserving existing behavior): clear the
        # adjoint buffer at the start of every outer loop so it never mixes
        # samples from two different (u, h) snapshots. See inner_loop_u for
        # the exact mechanism/rationale.
        self.clear_adjoint_buffer_per_outer_loop = getattr(
            cfg, "clear_adjoint_buffer_per_outer_loop", False
        )
        if self.fc_loss_mode == "log_variance":
            # Stored RN ratios and old-controller means must belong to the same
            # policy snapshot; stale paths are not valid VarGrad training data.
            self.clear_adjoint_buffer_per_outer_loop = True

        # Process the training curriculum
        self.max_outer_loops = cfg.max_outer_loops
        self.adjoint_matching_epochs = cfg.adjoint_matching_epochs
        self.corrector_matching_epochs = cfg.corrector_matching_epochs
        initial_epochs = getattr(cfg, "initial_adjoint_matching_epochs", None)
        if initial_epochs is not None and (int(initial_epochs) != initial_epochs or initial_epochs < 1):
            raise ValueError("initial_adjoint_matching_epochs must be a positive integer")
        self.initial_adjoint_matching_epochs = int(
            self.adjoint_matching_epochs if initial_epochs is None else initial_epochs)
        self.total_epochs = self.max_outer_loops * (
            self.adjoint_matching_epochs + self.corrector_matching_epochs
        ) + self.initial_adjoint_matching_epochs - self.adjoint_matching_epochs
        
        # Optimizers and schedulers
        self.controller_optimizer = hydra.utils.instantiate(
            cfg.controller_optimizer,
            params=self.controller.parameters(),
        )
        self.corrector_optimizer = hydra.utils.instantiate(
            cfg.corrector_optimizer,
            params=self.corrector.parameters(),
        )
        self.controller, self.controller_optimizer = self.fabric.setup(
            self.controller, self.controller_optimizer
        )
        self.corrector, self.corrector_optimizer = self.fabric.setup(
            self.corrector, self.corrector_optimizer
        )
        
        self.zeros_tensor = self.fabric.to_device(torch.zeros(self.cfg.train_batch_size))
        self.ones_tensor = self.fabric.to_device(torch.ones(self.cfg.train_batch_size))
        self.divergence_stopper = JointMetricDivergenceStopper(self.cfg.divergence_early_stop)
        self.early_stop_requested = False
        self.outer_step, self.phase, self.phase_epoch = 0, "ready", 0

        self.model_capacity = {
            "forward_controller_parameters": sum(
                parameter.numel() for parameter in self.controller.parameters()
            ),
            "corrector_parameters": sum(
                parameter.numel() for parameter in self.corrector.parameters()
            ),
            "total_trainable_parameters": sum(
                parameter.numel()
                for model in (self.controller, self.corrector)
                for parameter in model.parameters()
            ),
            "integration_steps": int(len(self.integrator.timesteps) - 1),
        }
        if self.fabric.global_rank == 0 and wandb.run is not None:
            for name, value in self.model_capacity.items():
                wandb.run.summary[f"capacity/{name}"] = int(value)
        
        ############################################################################
        self.ckpts = {}
        self.checkpoint_directory = Path(self.cfg.checkpoint_directory) / self.group_id
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        for name in self.cfg.monitor_values:
            name_safe = name.replace("/", "_")
            self.ckpts[name_safe] = Checkpointer(
                CheckpointConfig(
                    enable=True,
                    checkpoint_dir=self.checkpoint_directory,
                    filename_fmt="{tag}.pt",
                    mode=str(self.cfg.monitor_modes.get(name, "min")),
                ),
                make_payload=lambda: {
                    "fbsde": self.sde.state_dict(),
                    "sampling_metadata": self.sampling_metadata(),
                },
            )
        
        if wandb.run is not None:
            wandb.run.summary["group_id"] = self.group_id
            wandb.run.summary["checkpoint_directory"] = str(self.checkpoint_directory)
        
    @property
    def lm(self) -> float:
        """lagrangian multiplier expressed from gamma. Kept for compatibility."""
        return (1 - self.gamma) / self.gamma if self.gamma else math.inf

    def _update_divergence_early_stop(self, metrics):
        eval_index = self.current_epoch // self.cfg.log_detail_interval_epoch
        if not self.divergence_stopper.update(metrics, eval_index):
            return
        self.early_stop_requested = True
        print(f"Early stopping at epoch {self.current_epoch}: {self.divergence_stopper.reason}")
        if wandb.run is not None:
            wandb.run.summary["early_stop/triggered"] = True
            wandb.run.summary["early_stop/epoch"] = int(self.current_epoch)
            wandb.run.summary["early_stop/reason"] = self.divergence_stopper.reason
    
    def inner_loop_u(self):
        current_task = "adjoint_matching"
        resuming = getattr(self, "phase", "ready") == "u" and getattr(self, "phase_epoch", 0) > 0
        if not resuming:
            self.prev_integrator = copy_model(self.prev_integrator, self.integrator)
        if self.clear_adjoint_buffer_per_outer_loop and not resuming:
            # Without this, the buffer's FIFO eviction alone leaves stale
            # entries from the previous outer loop's (u_{i-1}, h_{i-1})
            # snapshot mixed in with fresh (u_i, h_i) entries for the first
            # ~max_size/buffer_samples_per_epoch epochs of every outer loop --
            # normalize_log_weights then self-normalizes importance weights
            # from two different reference measures together, which isn't
            # valid. Clearing here guarantees every sample in the buffer was
            # generated from the SAME frozen snapshot for the whole phase.
            self.adjoint_buffer.clear()
        start_epoch = self.phase_epoch if resuming else 0
        self.phase = "u"
        regular_epochs = getattr(self, "adjoint_matching_epochs", self.cfg.adjoint_matching_epochs)
        phase_epochs = (getattr(self, "initial_adjoint_matching_epochs", regular_epochs)
                        if getattr(self, "outer_step", 0) == 0 else regular_epochs)
        for j in range(start_epoch, phase_epochs):
            self.populate_buffer_adjoint_matching()
            dataloader = self.create_dataloader(current_task)
            for batch in tqdm(
                dataloader,
                desc=f"Epoch {self.current_epoch + 1}/{self.total_epochs}",
                leave=False,
                disable=self.fabric.global_rank != 0 or not self.cfg.progress_bar,
            ):
                flg = self.training_step(batch, current_task)
                if flg:
                    self.global_step += 1
        
            self.current_epoch += 1
            self.phase_epoch = j + 1
            if hasattr(self.potential, "update_epoch"):
                self.potential.update_epoch(self.current_epoch)

            if (self.current_epoch % self.cfg.log_detail_interval_epoch == 0):
                metrics = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch)
                self._weight_rollback_check(metrics)
                self.save_checkpoint(periodic=True)
                # Check for early stopping
                for name in self.cfg.monitor_values:
                    name_safe = name.replace("/", "_")
                    info = self.ckpts[name_safe].update(
                        value=metrics[name],
                        step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                        tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                        extra_status={"eval_step": metrics["eval_step"], "epoch": metrics["epoch"]},
                    )
                self._update_divergence_early_stop(metrics)
                del metrics
                torch.cuda.empty_cache()
                if self.early_stop_requested:
                    break
            
    def inner_loop_h(self):
        current_task = "corrector_matching"
        resuming = getattr(self, "phase", "ready") == "h" and getattr(self, "phase_epoch", 0) > 0
        if self.corrector_gamma_mode != "none" and not resuming:
            self.prev_corrector = copy_model(self.prev_corrector, self.corrector)
            if self.corrector_gamma_mode in ("reuse_u_gamma", "joint_reuse"):
                self.gamma_h = self.gamma
            elif self.corrector_gamma_mode == "independent":
                self.gamma_h = self.compute_corrector_gamma(reference=self.prev_corrector)
            elif self.corrector_gamma_mode == "joint_independent":
                self.gamma_h = self.compute_corrector_gamma(reference=self.controller)
            self.fabric.log("train/corrector_gamma", self.gamma_h, step=self.global_step)
        start_epoch = self.phase_epoch if resuming else 0
        self.phase = "h"
        for j in range(start_epoch, self.cfg.corrector_matching_epochs):
            self.populate_buffer_corrector_matching()
            dataloader = self.create_dataloader(current_task)
            for batch in tqdm(
                dataloader,
                desc=f"Epoch {self.current_epoch + 1}/{self.total_epochs}",
                leave=False,
                disable=self.fabric.global_rank != 0 or not self.cfg.progress_bar,
            ):
                flg = self.training_step(batch, current_task)
                if flg:
                    self.global_step += 1
            
            self.current_epoch += 1
            self.phase_epoch = j + 1
            if hasattr(self.potential, "update_epoch"):
                self.potential.update_epoch(self.current_epoch)

            if (self.current_epoch % self.cfg.log_detail_interval_epoch == 0):
                metrics = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch)
                self._weight_rollback_check(metrics)
                self.save_checkpoint(periodic=True)
                for name in self.cfg.monitor_values:
                    name_safe = name.replace("/", "_")
                    info = self.ckpts[name_safe].update(
                        value=metrics[name],
                        step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                        tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                        extra_status={"eval_step": metrics["eval_step"], "epoch": metrics["epoch"]},
                    )
                self._update_divergence_early_stop(metrics)
                del metrics
                torch.cuda.empty_cache()
                if self.early_stop_requested:
                    break
    ############################################################################
    # Training loop structure
    ############################################################################

    def fit(self):
        """Main training loop for the ASBS."""
        self.global_step = 0
        self.current_epoch = 0
        if (
            hasattr(self.cfg, "pretrained_controller_checkpoint")
            and self.cfg.pretrained_controller_checkpoint
        ):
            self.load_pretrained_controller_checkpoint()
            
        if self.cfg.resume:
            self.load_checkpoint()
            
        metrics = self.log_details(0)
        for name in self.cfg.monitor_values:
            if not self.cfg.monitor_initial_state:
                continue
            name_safe = name.replace("/", "_")
            info = self.ckpts[name_safe].update(
                value=metrics[name],
                step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                extra_status={"eval_step": metrics["eval_step"], "epoch": metrics["epoch"]},
            )
        self._update_divergence_early_stop(metrics)

        for outer_step in range(getattr(self, "outer_step", 0), self.max_outer_loops):
            self.outer_step = outer_step
            phase = getattr(self, "phase", "ready")
            if phase == "ready" and self.gamma_schedule == "trust_region":
                self.gamma, self.V0 = self.optimize_gamma()
                if self.trust_region_adaptive:
                    self.adapt_trust_region_bound()
                self.fabric.log(
                    "train/trust_region_bound", self.trust_region_bound, step=self.global_step
                )
            elif phase == "ready" and self.gamma_schedule == "fixed":
                # V0 must still be refreshed every outer loop even though gamma
                # itself doesn't change -- see compute_V0_for_fixed_gamma for why
                # (leaving it at the __init__ default of 1e-9 makes the buffer's
                # rescaling factor r collapse to ~0, degenerating both the
                # RN[h|u]/ess diagnostic AND -- for fc_loss_mode=cross_entropy --
                # the CE loss's own importance weights to a trivial iw=1 for every
                # sample, silently turning "importance weighting" into a no-op).
                self.V0 = self.compute_V0_for_fixed_gamma()
            if phase in ("ready", "u"):
                self.inner_loop_u()
                if self.early_stop_requested:
                    break
                self.phase, self.phase_epoch = "h", 0
            self.inner_loop_h()
            if self.early_stop_requested:
                break
            self.outer_step, self.phase, self.phase_epoch = outer_step + 1, "ready", 0
            self.save_checkpoint(periodic=True)

        self.save_checkpoint()
        self.log_final()
        
    ############################################################################
    # Buffer and dataloader management
    ############################################################################
    @torch.no_grad()
    def populate_buffer_adjoint_matching(self):
        """Populate the buffer with data from the controlled SDE integration and log the
        buffer size if the next task is adjoint matching."""
        buffer = self.adjoint_buffer
        buffer.bind_metadata(self.sampling_metadata())
        is_init_stage = self.current_epoch < getattr(
            self, "initial_adjoint_matching_epochs", self.cfg.adjoint_matching_epochs)
        # Determine the number of samples to populate the buffer
        # NOTE: Currently we don't save buffer to checkpoint, so restarting the training
        # will repopulate the buffer from scratch with initial_buffer_samples
        if len(buffer) == 0:
            num_samples = self.cfg.initial_buffer_samples
            num_extend_cycles = num_samples // self.cfg.inference_batch_size
            assert num_extend_cycles > 0, "Initial buffer samples must be at least as large as inference batch size."
        else:
            num_samples = self.cfg.buffer_samples_per_epoch
            num_extend_cycles = num_samples // self.cfg.inference_batch_size
            assert num_extend_cycles > 0, "Buffer samples per epoch must be at least as large as inference batch size."
            
        if not num_samples % self.cfg.inference_batch_size == 0:
            rank_zero_warn(
                f"Number of samples ({num_samples}) is not divisible by "
                f"inference batch size ({self.cfg.inference_batch_size}). "
                "Adjusting to the nearest lower multiple."
            )

        if self.fc_loss_mode == "path_kl":
            # Only x0 is needed by a current-policy pathwise loss. Avoid frozen
            # rollouts and terminal energy/force calls merely to fill this buffer.
            for _ in range(num_extend_cycles):
                buffer.extend(data_0=self.source.sample((self.cfg.inference_batch_size,)))
            self.fabric.log("train/adjoint_matching_buffer_size", len(buffer), step=self.global_step)
            return

        # Put the models into evaluation mode
        self.prev_integrator.eval()

        # Populate the buffer with data from the controlled SDE integration
        want_dense_trajectory = self.fc_loss_mode == "log_variance"
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            integral_results = self.prev_integrator.run_with_logweights(
                integrator=self.prev_integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=want_dense_trajectory,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )

            # filter
            # r is a magnitude-only damping factor in (0, 1] that keeps exp(gamma *
            # r * log_Ph_Pref) from overflowing downstream in _compute_iw. V0 = -log
            # E[w^-1] (from optimize_gamma) and raw_log_Ph_Pref_mean are BOTH generically
            # <= 0 -- E_{P^{u_i}}[log w] = -KL(P^{u_i} | P^h) <= 0 is a basic property of
            # a Radon-Nikodym derivative, not an edge case -- so `-(V0 / mean)` (the
            # previous formula) is generically NEGATIVE, not the intended small-positive
            # damping factor. That sign flip corrupts the stored log-weight: it inverts
            # the ranking of samples, giving WORSE (more negative raw log w) samples a
            # LARGER exponentiated importance weight than better ones. Using magnitudes
            # directly gives the intended non-negative damping factor regardless of sign.
            raw_log_Ph_Pref_mean = integral_results['log_Ph_Pref'].mean().item()
            r = min(abs(self.V0) / max(abs(raw_log_Ph_Pref_mean), 1e-12), 1.0)
            populate_data = {
                "data_0": data_0.detach().cpu(),
                "data_1": integral_results['data_1'].detach().cpu(),
                "log_Ph_Pref": (r * integral_results['log_Ph_Pref']).detach().cpu(), # dP^h/dP^u
                # Undamped version for adjoint_matching_log_variance_step: the "r" rescaling
                # above exists to avoid exp() overflow in the cross-entropy loss's importance
                # weight, which doesn't apply to the log-variance loss (no exponentiation).
                "log_Ph_Pref_raw": integral_results['log_Ph_Pref'].detach().cpu(),
                "log_Pu_Pref": integral_results['log_Pu_Pref'].detach().cpu(),
            }

            if want_dense_trajectory:
                # Full dense trajectory (T+1 states), stacked along a new time axis, so
                # adjoint_matching_log_variance_step can compute the exact discrete-time
                # Girsanov log-ratio summed over every step (matching the discretization
                # run_with_logweights itself uses), not a coarse subsampled approximation.
                trajectory = integral_results["trajectory"]
                data_traj = torch.stack(trajectory, dim=1)  # (batch, T+1, ...)
                populate_data["data_traj"] = data_traj.detach().cpu()

            if "log_Pref_Pu_det" in integral_results:
                populate_data["log_Pref_Pu_det"] = integral_results["log_Pref_Pu_det"].unsqueeze(-1).detach()
            
            if self.fc_loss_mode == "cross_entropy":
                terminal_cost_results = self.terminal_cost(
                    data_1=integral_results['data_1'],
                    is_init_stage=is_init_stage,
                    return_energy=self.cfg.log_buffer_energy,
                    return_grad_norm=self.cfg.log_buffer_grad_norm,
                )
            
                populate_data["terminal_cost"] = terminal_cost_results["terminal_cost"].detach().cpu()
                time_1 = self.fabric.to_device(torch.ones(self.cfg.inference_batch_size, dtype=torch.float))
                populate_data["control_1"] = self.prev_integrator.sde.u_controller(time_1, integral_results['data_1']).detach().cpu()
                if self.per_time_anchor_target:
                    # Frozen policy's own output at X_0 (time=0), needed so
                    # adjoint_matching_step can anchor the tau=0 target term to
                    # u_i(X_0, 0) instead of u_i(X_T, T) -- see per_time_anchor_target's
                    # docstring in __init__. X_0 doesn't change once sampled, so this
                    # is computed once here rather than per training step.
                    time_0 = self.fabric.to_device(torch.zeros(self.cfg.inference_batch_size, dtype=torch.float))
                    populate_data["control_0"] = self.prev_integrator.sde.u_controller(time_0, data_0).detach().cpu()

                if self.cfg.log_buffer_energy:
                    populate_data["energy"] = terminal_cost_results["energy"].detach().cpu()
                if self.cfg.log_buffer_grad_norm:
                    populate_data["grad_norm"] = terminal_cost_results["grad_norm"].detach().cpu()

            else:
                if self.cfg.log_buffer_energy:
                    populate_data["energy"] = -integral_results["target_log_prob"].detach().cpu()
                if self.cfg.log_buffer_grad_norm:
                    gradient = self.potential.grad_E(integral_results["data_1"])
                    populate_data["grad_norm"] = gradient.reshape(gradient.shape[0], -1).norm(dim=-1).detach().cpu()

            if self.adjoint_buffer_duplicates > 1:
                populate_data = {
                    k: v.repeat(self.adjoint_buffer_duplicates, *([1] * (v.dim() - 1)))
                    for k, v in populate_data.items()
                }
            buffer.extend(**populate_data)

        # Log the buffer statistics
        # Log buffer median energy
        if self.global_step % self.cfg.log_loss_interval_step == 0:

            if self.cfg.log_buffer_energy:
                median_energy = torch.as_tensor(buffer.storage["energy"]).median()
                median_energy = self.fabric.all_reduce(
                    self.fabric.to_device(median_energy), reduce_op="mean"
                )
                self.fabric.log(
                    "train/adjoint_matching_buffer_median_energy",
                    median_energy,
                    step=self.global_step,
                )
                # Save samples
                if self.cfg.save_samples and self.fabric.global_rank == 0:
                    if self.current_epoch % self.cfg.save_sample_interval_epoch == 0:
                        self.save_samples(integral_results['data_1'])
            # Log buffer max grad norm
            if self.cfg.log_buffer_grad_norm:
                median_grad_norm = torch.stack(
                    buffer.storage["grad_norm"]
                ).median()
                median_grad_norm = self.fabric.all_reduce(
                    self.fabric.to_device(median_grad_norm), reduce_op="max"
                )
                self.fabric.log(
                    "train/adjoint_matching_buffer_median_grad_norm",
                    median_grad_norm,
                    step=self.global_step,
                )
            
            self.fabric.log(
                f"train/adjoint_matching_buffer_size", len(buffer), step=self.global_step
            )
            self.fabric.log(
                "train/adjoint_matching_buffer_rescale_r", r, step=self.global_step
            )
            self.fabric.log(
                "train/adjoint_matching_buffer_raw_log_Ph_Pref_mean",
                raw_log_Ph_Pref_mean,
                step=self.global_step,
            )
            # Restore the training state of the models
            self.fabric.barrier()
        self.integrator.train()
        
    @torch.no_grad()
    def populate_buffer_corrector_matching(self):
        buffer = self.corrector_buffer
        buffer.bind_metadata(self.sampling_metadata())
        # is_init_stage = self.current_epoch < self.cfg.corrector_matching_epochs
        # Determine the number of samples to populate the buffer
        # NOTE: Currently we don't save buffer to checkpoint, so restarting the training
        # will repopulate the buffer from scratch with initial_buffer_samples
        if len(buffer) == 0:
            num_samples = self.cfg.initial_buffer_samples
            num_extend_cycles = num_samples // self.cfg.inference_batch_size
            assert num_extend_cycles > 0, "Initial buffer samples must be at least as large as inference batch size."
            
        else:
            num_samples = self.cfg.buffer_samples_per_epoch
            num_extend_cycles = num_samples // self.cfg.inference_batch_size
            assert num_extend_cycles > 0, "Buffer samples per epoch must be at least as large as inference batch size."
        
        if not num_samples % self.cfg.inference_batch_size == 0:
            rank_zero_warn(
                f"Number of samples ({num_samples}) is not divisible by "
                f"inference batch size ({self.cfg.inference_batch_size}). "
                "Adjusting to the nearest lower multiple."
            )
            
         # Put the models into evaluation mode
        self.integrator.eval()

        # Populate the buffer with data from the controlled SDE integration
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            populate_data = {
                "data_0": data_0.detach().cpu(),
                "data_1": data_1.detach().cpu(),
            }
            buffer.extend(**populate_data)
        
        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                f"train/corrector_matching_buffer_size", len(buffer), step=self.global_step
            )
            # Restore the training state of the models
            self.fabric.barrier()
        
        self.integrator.train()
    
    def populate_buffer_log_weights(self, buffer):
        self.integrator.eval()
        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        num_iters = self.cfg.num_samples_for_dual_loss[0] // (self.cfg.inference_batch_size // self.cfg.num_samples_for_dual_loss[1])
        for j in range(num_iters):
            data_0 = self.source.replicate_sample((self.cfg.inference_batch_size//batch_size_path,), batch_size_path)
            integral_results = self.integrator.run_with_logweights(
                integrator=self.integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=self.cfg.progress_bar,
            )
            log_Pu_Ph = -integral_results["log_Ph_Pref"].reshape(self.cfg.inference_batch_size//batch_size_path, batch_size_path)
            # the expectation wrt paths using self.batch_size_path samples
            # dP^u/dP^h
            buffer.extend(**{"log_weights": log_Pu_Ph.detach().cpu()})
        
        self.integrator.train()

    def save_samples(self, data):
        """Save the data samples to an xyz file."""
        if self.fabric.global_rank == 0:
            sample_directory = Path(self.cfg.sample_directory)
            if not sample_directory.exists():
                sample_directory.mkdir(parents=True, exist_ok=True)
            if isinstance(data, torch.Tensor):  # Save as XYZ if data is tensor
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.npy"
                np.save(filename, data.cpu().numpy())
            else:  # default to XYZ format
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.xyz"
                ase.io.write(str(filename), data.to_ase())
                
    def save_final_samples(self, data_0, data_1, name: str):
        """Save final samples next to checkpoints."""
        if self.fabric.global_rank != 0:
            return

        output_dir = Path(self.checkpoint_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(data_0, torch.Tensor):
            np.save(output_dir / f"{name}_data_0.npy", data_0.cpu().numpy())
            np.save(output_dir / f"{name}_data_1.npy", data_1.cpu().numpy())
        else:
            ase.io.write(output_dir / f"{name}_data_0.xyz", data_0.to_ase())
            ase.io.write(output_dir / f"{name}_data_1.xyz", data_1.to_ase())

    def create_dataloader(self, current_task) -> DataLoader:
        # Select the appropriate dataset (buffer) based on the current task
        if current_task == "adjoint_matching":
            dataset = self.adjoint_buffer
        elif current_task == "corrector_matching":
            dataset = self.corrector_buffer
        # Create a sampler for the dataset
        # NOTE: Not using distributed sampler for now, since we have a separate buffer
        # for each process
        num_samples = self.cfg.train_batch_size * self.cfg.steps_per_epoch
        epoch_seed = self.rank_seed ^ (self.current_epoch << 16)
        generator = torch.Generator().manual_seed(epoch_seed)
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=num_samples, generator=generator
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)

    ############################################################################
    # Training steps: adjoint and corrector matching
    ############################################################################

    def training_step(self, batch, current_task) -> None:
        # Get the appropriate step function based on the current task
        if current_task == "adjoint_matching":
            if self.fc_loss_mode == "path_kl":
                losses = self.forward_path_kl_step(batch)
            elif self.fc_loss_mode == "log_variance":
                losses = self.adjoint_matching_log_variance_step(batch)
            else:
                losses = self.adjoint_matching_step(batch)
            model = self.controller
            optimizer = self.controller_optimizer
            
        elif current_task == "corrector_matching":
            losses = self.corrector_matching_step(batch)
            model = self.corrector
            optimizer = self.corrector_optimizer

        # Perform the optimizer step
        optimizer.zero_grad()
        loss = losses["loss"]
        scaled_controller = (current_task == "adjoint_matching" and
                             getattr(self, "fc_backward_mode", "native") == "loss_scaled_fp64_clip")
        if scaled_controller:
            if self.logvar_amp_dtype is not None or self.fabric.world_size != 1:
                raise RuntimeError("Scaled LV backward requires one process without autocast")
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise RuntimeError("Scaled LV backward requires FP32 parameters")
            self.fc_backward_statistics["attempts"] += 1
        
        if not torch.isfinite(loss):
            if scaled_controller:
                self.fc_backward_statistics["loss_nonfinite"] += 1
            return False
        
        if not scaled_controller:
            self.fabric.backward(loss)
        try:
            if scaled_controller:
                from tr_asbs.utils.scaled_backward_diagnostic import power_of_two_scale, unscale_and_clip
                # A positive binary factor changes only backward arithmetic.
                # Unscale in FP64 before the original norm bound and Adam step.
                scale = power_of_two_scale(loss)
                self.fabric.backward(loss * scale)
                unscale_and_clip(model.parameters(), scale, self.cfg.gradient_clip_val)
                stats = self.fc_backward_statistics
                stats["minimum_scale"] = min(stats["minimum_scale"], scale)
                stats["maximum_scale"] = max(stats["maximum_scale"], scale)
            else:
                clip_gradients_with_overflow_fallback(
                    self.fabric, model, optimizer, self.cfg.gradient_clip_val)
        except RuntimeError as e:
            if "non-finite" in str(e):
                if scaled_controller:
                    self.fc_backward_statistics["gradient_nonfinite"] += 1
                optimizer.zero_grad(set_to_none=True)
                return False
            raise
        optimizer.step()
        if scaled_controller:
            self.fc_backward_statistics["updates"] += 1
        self.last_training_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
        
        del loss
        torch.cuda.empty_cache()

        # Update the EMA
        if self.cfg.ema_decay is not None:
            model.update_ema(self.global_step)

        # Log the lr and loss for the current task
        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                f"train/{current_task}_lr",
                optimizer.param_groups[0]["lr"],
                step=self.global_step,
            )
            for key, loss in losses.items():
                loss = self.fabric.all_reduce(loss.detach(), reduce_op="mean")
                self.fabric.log(
                    f"train/{current_task}_{key}", loss, step=self.global_step
                )
        return True
            
            
    def _compute_iw(self, batch, gamma) -> torch.Tensor:
        log_Ph_Pref = self.fabric.to_device(batch["log_Ph_Pref"]).squeeze()
        log_Pu_Pref = self.fabric.to_device(batch["log_Pu_Pref"]).squeeze()
        scaled_log_weights = normalize_log_weights(gamma * log_Ph_Pref + (1 - gamma) * log_Pu_Pref)
        iw = torch.exp(scaled_log_weights).clamp(max=self.iw_clip_value)
        return iw
        
    def adjoint_matching_step(self, batch) -> torch.Tensor:
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]  # ∇E + h
        control_1 = batch["control_1"]

        batch_size = self.cfg.train_batch_size

        iw = self._compute_iw(batch, self.gamma)  # (B,) を想定

        # time_0, time_1 を (B,) に揃える（self.zeros_tensor/self.ones_tensor がスカラーでもOKにする）
        time_0 = self.zeros_tensor
        time_1 = self.ones_tensor
        if time_0.ndim == 0:
            time_0 = time_0.expand(batch_size)
        if time_1.ndim == 0:
            time_1 = time_1.expand(batch_size)

        # t ~ U(1e-3, 1-1e-3)
        time = self.fabric.to_device(
            torch.rand(batch_size, dtype=torch.float)
        ) * (1 - 2*self.bridge_time_epsilon) + self.bridge_time_epsilon

        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        if self.per_time_anchor_target:
            # Anchor each of the three target terms to the frozen policy's OWN
            # output at that SAME time, instead of always at T -- see
            # per_time_anchor_target's docstring in __init__. Reduces to the
            # unmodified behavior exactly at gamma=1 (terminal_cost, control_tau
            # cancels out algebraically regardless of tau); at gamma=0 it
            # becomes "-control_tau" at every tau (a genuine no-op everywhere),
            # instead of "-control_1" everywhere (a no-op only at T).
            control_0 = batch["control_0"]
            with torch.no_grad():
                control_t = self.prev_integrator.sde.u_controller(time, data_t)
            target_0 = self.gamma * (terminal_cost + control_0) - control_0
            target_1 = self.gamma * (terminal_cost + control_1) - control_1
            target_t = self.gamma * (terminal_cost + control_t) - control_t
        else:
            target = self.gamma * (terminal_cost + control_1) - control_1
            target_0 = target_1 = target_t = target

        if isinstance(self.base_sde, LinearGaussianReference):
            # This branch is admitted only at fixed gamma=1. For general TR,
            # frozen-policy Jacobians/running costs require path_kl/log_variance.
            from tr_asbs.process.reference import coefficient
            target_0 = coefficient(self.base_sde.adjoint_transport(time_0), data_0)*target_0
            target_t = coefficient(self.base_sde.adjoint_transport(time), data_t)*target_t

        # 3つを dim0 でまとめる
        time_all = torch.cat([time_0, time_1, time], dim=0)          # (3B,)
        data_all = torch.cat([data_0, data_1, data_t], dim=0)        # (3B, ...)

        # target, iw も 3回ぶんに拡張（target は data と同じ batch 次元を持つ想定）
        target_all = torch.cat([target_0, target_1, target_t], dim=0)  # (3B, ...)
        iw_all = torch.cat([iw, iw, iw], dim=0)                      # (3B,)

        # forward は1回
        control_all = self.controller(time_all, data_all)            # (3B, ...)

        # 元の mse_with_iw を “3B版” にしてから、(3, B) に戻して各 loss を取る
        per_sample = (
            (control_all + target_all).pow(2)
            .reshape(3 * batch_size, -1)
            .mean(-1)                                                # (3B,)
            .mul(iw_all)                                             # (3B,)
        )                                                            # ここまではサンプル毎
        if self.fc_loss_weight == "diffusion_energy":
            per_sample = per_sample*self.base_sde.diffusion(time_all).square().reshape(-1)

        per_sample = per_sample.view(3, batch_size)
        loss_0 = per_sample[0].mean()
        loss_1 = per_sample[1].mean()
        loss_t = per_sample[2].mean()

        loss = loss_0 + loss_t + loss_1
        return dict(
            loss=loss,
            loss_0=loss_0.detach().cpu(),
            loss_t=loss_t.detach().cpu(),
            loss_1=loss_1.detach().cpu(),
        )

    def forward_path_kl_step(self, batch):
        """Finite-kernel TR objective on reparameterized current-policy paths.

        gamma*KL(P_new|P_h)+(1-gamma)*KL(P_new|P_old).
        P_h = nu(x_T) product q_h(x_k|x_{k+1}). nu may be unnormalized;
        its normalizer is an additive constant. Expected log p_new is replaced
        by negative kernel entropy. P_old and q_h parameters stay frozen, but
        their state derivatives are retained. This is a discrete objective.
        """
        self.prev_integrator.requires_grad_(False)
        x = self.fabric.to_device(batch["data_0"])
        initial_log_prob = self.source.log_prob(x)
        forward_reverse = torch.zeros_like(initial_log_prob)
        old_kl = torch.zeros_like(initial_log_prob)
        times = self.integrator.timesteps.to(x.device)
        for t, tn in zip(times[:-1], times[1:]):
            dt = tn-t
            new = self.integrator.transition(x, t, dt)
            old = self.prev_integrator.transition(x, t, dt)
            old_kl = old_kl + new.kl(old)
            xn = new.sample()
            reverse = self.prev_integrator.transition(xn, tn, dt, backward=True)
            forward_reverse = forward_reverse - new.entropy()-reverse.log_prob(x)
            x = xn
        target_cost = initial_log_prob-self.potential.log_prob(x)+forward_reverse
        loss = (self.gamma*target_cost+(1-self.gamma)*old_kl).mean()
        return dict(loss=loss, path_target_cost=target_cost.mean().detach().cpu(),
                    achieved_forward_kl=old_kl.mean().detach().cpu())

    def adjoint_matching_log_variance_step(self, batch):
        """VarGrad with exact densities of the configured discrete kernels."""
        data_traj = self.fabric.to_device(batch["data_traj"])
        log_Ph_Pref = self.fabric.to_device(batch["log_Ph_Pref_raw"]).reshape(-1)
        times = self.prev_integrator.timesteps.to(data_traj.device)
        batch_size, steps = data_traj.shape[:2]
        steps -= 1
        shape = data_traj.shape[2:]
        x = data_traj[:, :-1].reshape(batch_size*steps, *shape)
        xn = data_traj[:, 1:].reshape(batch_size*steps, *shape)
        t = times[:-1].expand(batch_size, steps).reshape(-1)
        dt = times.diff().expand(batch_size, steps).reshape(-1)
        # One network call per complete trajectory batch. Kernel densities stay
        # in state precision even when the controller is autocast.
        with torch.autocast(device_type=x.device.type, dtype=self.logvar_amp_dtype,
                            enabled=self.logvar_amp_dtype is not None):
            new = self.integrator.transition(x, t, dt)
            with torch.no_grad():
                old = self.prev_integrator.transition(x, t, dt)
        ratio = (new.log_prob(xn)-old.log_prob(xn)).reshape(batch_size, steps).sum(-1)
        log_weight = ratio-self.gamma*log_Ph_Pref
        loss = log_weight.var(unbiased=True)
        return dict(loss=loss, log_weight_mean=log_weight.mean().detach().cpu())

    def corrector_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the corrector matching and return the loss.
        Equation 15 in the ASBS paper.
        """
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]

        batch_size = self.cfg.train_batch_size

        # t ~ U(1e-3, 1-1e-3)
        time = self.fabric.to_device(torch.rand(batch_size) * (1 - 2*self.bridge_time_epsilon) + self.bridge_time_epsilon)
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        # time_1 を (B,) に揃える（self.ones_tensor がスカラーでもOK）
        time_1 = self.ones_tensor
        if time_1.ndim == 0:
            time_1 = time_1.expand(batch_size)

        # corrector forward を1回にまとめる
        time_all = torch.cat([time, time_1], dim=0)          # (2B,)
        data_all = torch.cat([data_t, data_1], dim=0)        # (2B, ...)

        h_all = self.corrector(time_all, data_all)           # (2B, ...)
        h_t, h_1 = h_all.split(batch_size, dim=0)

        # score は従来どおり（必要ならこれも concat で1回にできる）
        score_t = self.base_sde.cond_score_t0(time=time, data_0=data_0, data_t=data_t)
        score_1 = self.base_sde.cond_score_t0(time=time_1, data_0=data_0, data_t=data_1)

        if self.corrector_gamma_mode != "none":
            # Trust-region-damp the corrector's own target: instead of
            # regressing straight to the fresh ground-truth score every outer
            # loop, blend toward a reference model's output by (1 - gamma_h).
            # "reuse_u_gamma"/"independent" reference h_i (the corrector's own
            # previous iterate); "joint_reuse"/"joint_independent" reference
            # the CONTROLLER's current output instead -- directly targeting
            # Var_{P^{u_i}}(log dP^h/dP^u), the quantity the convergence
            # theorem actually needs bounded (see compute_corrector_gamma).
            reference = (
                self.controller
                if self.corrector_gamma_mode in ("joint_reuse", "joint_independent")
                else self.prev_corrector
            )
            with torch.no_grad():
                ref_all = reference(time_all, data_all)
            ref_t, ref_1 = ref_all.split(batch_size, dim=0)
            target_t = self.gamma_h * score_t + (1 - self.gamma_h) * ref_t
            target_1 = self.gamma_h * score_1 + (1 - self.gamma_h) * ref_1
        else:
            target_t = score_t
            target_1 = score_1

        if self.bc_loss_weight == "none":
            coef_t = torch.ones_like(time)
        elif self.bc_loss_weight == "reference_variance":
            if isinstance(self.base_sde, LinearGaussianReference):
                _, _, variance = self.base_sde.transition_coefficients(0, time)
            else:
                variance = self.base_sde.diffsquare_integral(time)
            coef_t = 1+self.base_sde.diffusion(time).square()*(1-time)/variance.to(time)
        else:
            coef_t = 1+self.base_sde.diffusion(time).square()*(1-time)/time

        loss_t = (
            (h_t - target_t)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)                                  # (B,)
            .div(coef_t.squeeze())                     # (B,)
            .mean()
        )

        loss_1 = (
            (h_1 - target_1)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)
            .mean()
        )

        loss = loss_t + loss_1
        return dict(loss=loss, loss_t=loss_t.detach().cpu(), loss_1=loss_1.detach().cpu())

    @torch.no_grad()
    def compute_corrector_gamma(self, reference, msd_floor: float = 1e-8) -> float:
        """Closed-form analogue of optimize_gamma() for the corrector.

        The controller's trust region bounds KL(P^{u_{i+1}}|P^{u_i}) via a full
        exponential-family dual solve (optimize_gamma) because its loss involves
        importance weights over whole path measures. The corrector has no such
        structure -- corrector_matching_step is plain L2 score-matching to a
        closed-form target -- so a much simpler bound applies: for two additive-
        noise diffusions sharing the same diffusion coefficient, KL(P^a|P^b)
        ≈ (1/2) E[∫ ||a - b||^2 dt]. If h_{i+1} is trained to hit
        target = gamma_h * score + (1 - gamma_h) * reference(t,x), then in the
        limit of good optimization h_{i+1} - reference ≈ gamma_h * (score -
        reference), giving an achieved-KL proxy of
        (1/2) * gamma_h^2 * E[(score - reference)^2]. Setting this equal to
        corrector_trust_region_bound and solving for gamma_h gives a closed-form
        answer -- no scipy dual optimization needed.

        `reference` is either `self.prev_corrector` (bounds KL(h_{i+1}|h_i),
        the "independent" mode) or `self.controller` (bounds KL(h_{i+1}|u) --
        the actual quantity the convergence theorem needs bounded, per
        paper.tex Theorem \\ref{thm:convergence_rate}: Var_{P^{u_i}}(log dP^h/dP^u)
        -- the "joint_independent" mode).
        """
        self.populate_buffer_corrector_matching()
        dataloader = self.create_dataloader("corrector_matching")
        batch = next(iter(dataloader))
        data_0 = self.fabric.to_device(batch["data_0"])
        data_1 = self.fabric.to_device(batch["data_1"])
        batch_size = data_0.shape[0]

        time = self.fabric.to_device(torch.rand(batch_size) * (1 - 2*self.bridge_time_epsilon) + self.bridge_time_epsilon)
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)
        time_1 = self.ones_tensor
        if time_1.ndim == 0:
            time_1 = time_1.expand(batch_size)
        elif time_1.shape[0] != batch_size:
            time_1 = time_1[:batch_size]

        score_t = self.base_sde.cond_score_t0(time=time, data_0=data_0, data_t=data_t)
        score_1 = self.base_sde.cond_score_t0(time=time_1, data_0=data_0, data_t=data_1)
        ref_t = reference(time, data_t)
        ref_1 = reference(time_1, data_1)

        msd = 0.5 * (
            (score_t - ref_t).pow(2).mean() + (score_1 - ref_1).pow(2).mean()
        ).item()
        if msd < msd_floor:
            return 1.0
        gamma_h = math.sqrt(2.0 * self.corrector_trust_region_bound / msd)
        return float(min(1.0, gamma_h))

    ############################################################################
    # Checkpoint management
    ############################################################################

    def load_pretrained_controller_checkpoint(self):
        """Load a pretrained controller (+ corrector, if present in the
        checkpoint) checkpoint if specified in the config. Used to warm-start
        the importance-weighted/trust-region trainer from a plain-ASBS
        checkpoint (e.g. train_asbs.py's epoch_XXXX.pt, which saves both
        nets) so the curriculum's second phase starts from an already-good
        controller/corrector pair instead of both being untrained."""
        checkpoint_path = Path(self.cfg.pretrained_controller_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained controller checkpoint not found: {checkpoint_path}"
            )
        state = {"controller": self.controller, "corrector": self.corrector}
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.validate_sampling_checkpoint(payload)
        self.fabric.load(checkpoint_path, state)

    def validate_sampling_checkpoint(self, payload):
        metadata = payload.get("sampling_metadata")
        if metadata == self.sampling_metadata():
            return
        legacy_kernel = (type(self.base_sde).__name__ in ("VESDE", "EDMSDE", "PISCosineSDE")
                         and type(self.integrator).__name__ in ("ExponentialIntegrator", "AtomicExponentialIntegrator")
                         and self.integrator.grid == "uniform_t"
                         and all(getattr(m, "time_coordinates", "physical") == "physical"
                                 for m in self.controller.modules()))
        if metadata is not None or not (legacy_kernel and getattr(self.cfg, "allow_legacy_checkpoint", False)):
            raise ValueError("Checkpoint sampling metadata differs; resume rejected")

    def load_checkpoint(self):
        """Load the resumable checkpoint (see save_checkpoint(periodic=True)) if
        it exists. Looks specifically for "latest.pt" rather than globbing all
        *.pt in checkpoint_directory -- the per-metric "best" checkpoints
        (self.ckpts[...], filename "{tag}.pt") saved alongside it use a
        different, incompatible payload (just {"fbsde": ..., "_checkpoint": ...},
        no "global_step"/"current_epoch"), so picking the most-recently-modified
        *.pt file could grab one of those instead and crash on the missing keys.
        """
        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            "controller_optimizer": self.controller_optimizer,
            "corrector_optimizer": self.corrector_optimizer,
        }
        checkpoint_path = Path(self.checkpoint_directory) / "latest.pt"
        if not checkpoint_path.exists():
            return
        # Validate before mutating models/optimizers. Old checkpoints are
        # readable only with an explicit compatibility opt-in.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.validate_sampling_checkpoint(payload)
        if "prev_integrator" in payload:
            state.update(prev_integrator=self.prev_integrator, prev_corrector=self.prev_corrector)
        remainder = self.fabric.load(checkpoint_path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")
        self.gamma = remainder.pop("gamma", self.gamma)
        self.V0 = remainder.pop("V0", self.V0)
        self.gamma_h = remainder.pop("gamma_h", self.gamma_h)
        self.outer_step = remainder.pop("outer_step", 0)
        self.phase = remainder.pop("phase", "ready")
        self.phase_epoch = remainder.pop("phase_epoch", 0)
        if "prev_integrator" not in payload:
            self.prev_integrator = copy_model(self.prev_integrator, self.integrator)
            self.prev_corrector = copy_model(self.prev_corrector, self.corrector)
        self.prev_integrator.requires_grad_(False)
        self.prev_corrector.requires_grad_(False)
        self.adjoint_buffer.clear()
        self.corrector_buffer.clear()

    def save_checkpoint(self, only_controller: bool = False, periodic: bool = False):
        """Save the current state of the model and optimizer to a checkpoint.

        periodic=True (used for resumability during a long run, see
        inner_loop_u/inner_loop_h) writes to a single fixed "latest.pt" that
        gets overwritten each time, instead of accumulating a new epoch_XXXX.pt
        file at every call -- these fire every log_detail_interval_epoch
        epochs, so an ever-growing set of full-state files would otherwise
        pile up over a multi-day run.
        """
        checkpoint_directory = Path(self.checkpoint_directory)
        if not checkpoint_directory.exists():
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
        filename = "latest.pt" if periodic else f"epoch_{self.current_epoch:04d}.pt"
        checkpoint_path = checkpoint_directory / filename
        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            "controller_optimizer": self.controller_optimizer,
            "corrector_optimizer": self.corrector_optimizer,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "sampling_metadata": self.sampling_metadata(),
            "gamma": self.gamma,
            "V0": self.V0,
            "gamma_h": self.gamma_h,
            "prev_integrator": self.prev_integrator,
            "prev_corrector": self.prev_corrector,
            "outer_step": getattr(self, "outer_step", 0),
            "phase": getattr(self, "phase", "ready"),
            "phase_epoch": getattr(self, "phase_epoch", 0),
        }
        self.fabric.save(checkpoint_path, state)

    def sampling_metadata(self):
        return dict(kernel=self.integrator.kernel_metadata(), fc_loss_mode=self.fc_loss_mode,
                    bc_loss_weight=getattr(self, "bc_loss_weight", "legacy"),
                    fc_loss_weight=getattr(self, "fc_loss_weight", "legacy"),
                    bridge_time_epsilon=getattr(self, "bridge_time_epsilon", 0.001),
                    source=OmegaConf.to_container(self.cfg.source, resolve=True),
                    controller=OmegaConf.to_container(self.cfg.controller, resolve=True),
                    corrector=OmegaConf.to_container(self.cfg.corrector, resolve=True),
                    target=OmegaConf.to_container(self.cfg.potential, resolve=True),
                    terminal_cost=OmegaConf.to_container(self.cfg.terminal_cost, resolve=True),
                    terminal_gradient_clip_accumulation_dtype=getattr(
                        self.terminal_cost, "gradient_clip_accumulation_dtype", None),
                    data_provenance=getattr(self.potential, "data_metadata", None))
    
    @torch.no_grad()
    @torch._dynamo.disable()  
    def log_details(self, step: int, visualize: bool = False, eval_samples:int = None) -> dict:
        visualize = visualize or self.cfg.visualize
        eval_samples = eval_samples if eval_samples is not None else self.cfg.eval_samples
        self.integrator.eval()
        data_0 = self.source.sample((eval_samples,))
        integral_results = self.integrator.run_with_logweights(
            integrator=self.prev_integrator,
            prior_log_prob_fn=self.source.log_prob,
            target_log_prob_fn=self.potential.log_prob,
            initial_data=data_0,
            center_every_step=self.potential.is_mean_free,
            zero_last_step_noise=False,
            return_trajectory=True,
            progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
        )
        results = {
            "data_0": data_0[:eval_samples].detach().cpu(),
            "data_1": integral_results['data_1'][:eval_samples].detach().cpu(),
            "log_Ph_Pref": integral_results['log_Ph_Pref'][:eval_samples].detach().cpu(),
            "log_Pu_Pref": integral_results['log_Pu_Pref'][:eval_samples].detach().cpu(),
            'log_RN[u_next|u]_eff': integral_results['log_Pref_Pu_det'][:eval_samples].detach().cpu(),
            'log_RN[u_next|u]': -integral_results['log_Pu_Pref'][:eval_samples].detach().cpu(),
            'predict_drift_fn': self.sde.drift,
        }
        
        if self.current_epoch > 0:
            # バッファが空でないか確認
            if len(self.adjoint_buffer) > 0 and self.adjoint_buffer.storage.get("log_Ph_Pref"):
                # BatchBuffer.storage から直接参照 (CPU上のTensor)
                # コピーは発生せず、参照だけなので爆速かつメモリ消費ゼロ
                log_Ph_Pref = torch.stack(self.adjoint_buffer.storage["log_Ph_Pref"])
                log_Pu_Pref = torch.stack(self.adjoint_buffer.storage["log_Pu_Pref"])
                
                # ベクトル化された演算で一気に計算 (Pythonループなし)
                # log_RN[h|u] は参照渡しで済ませる
                results['log_RN[h|u]'] = log_Ph_Pref
                
                # 線形結合の計算 (CPU上で実行)
                # ここだけ新しいメモリを確保しますが、リストを作るより遥かにマシです
                results['log_RN[u_next|u]_pred_buffer'] = (
                    self.gamma * log_Ph_Pref + (1 - self.gamma) * log_Pu_Pref
                )
                
                results['lm'] = self.lm
                results['gamma'] = self.gamma
                if self.corrector_gamma_mode != "none":
                    results['gamma_h'] = self.gamma_h
            else:
                # path_kl buffers contain x0 only; evaluate RN on fresh current
                # paths without inventing a predicted update from replay weights.
                results['log_RN[h|u]'] = results['log_Ph_Pref']
                results['gamma'] = self.gamma
        force_log_gt = getattr(self.cfg, "log_ground_truth_figures", False)

        if self.current_epoch == 0:
            logs = self.potential.log_details(results, visualize=visualize)
            if visualize or force_log_gt:
                logs["figures/gt_data_0"] = self.potential.visualize_raw(data_0, title="gt_data_0")
                logs["figures/gt_data_1"] = self.potential.visualize(self.potential.sample(data_0.shape[0:1]), title="gt_data_1")
        else:
            logs = self.potential.log_details(results, visualize=visualize)
            if visualize:
                data_0_hat = self.integrator.run_backward(
                    final_data=integral_results['data_1'],
                    center_every_step=self.potential.is_mean_free,
                    return_trajectory=False,
                    progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
                )
                logs["figures/data_0_hat"] = self.potential.visualize_raw(data_0_hat, title="data_0_hat")
        logs["eval/raw_path_ess"] = compute_raw_path_ess(results["log_Ph_Pref"])
        kl_samples = integral_results['log_Pref_Pu_det'].detach().float()
        logs["trust_region/achieved_forward_kl"] = kl_samples.mean().item()
        logs["trust_region/achieved_forward_kl_se"] = (kl_samples.std(unbiased=False)/math.sqrt(kl_samples.numel())).item()
        logs["trust_region/bound"] = self.trust_region_bound
        logs["trust_region/gamma"] = self.gamma
        
        logs["epoch"] = self.current_epoch
        logs["eval_step"] = step

        self.fabric.log_dict(logs, step=self.global_step)

        if "loss/neg_avg_energy" in logs:
            self._last_logged_energy = logs["loss/neg_avg_energy"]

        logs.update({"data_0": data_0.detach().cpu(), "data_1": integral_results['data_1'].detach().cpu()})

        del results, integral_results
        return logs

    ############################################################################
    # Weight-checkpoint rollback (see __init__ for rationale)
    ############################################################################
    @staticmethod
    def _to_device_recursive(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: GammaASBSModule._to_device_recursive(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [GammaASBSModule._to_device_recursive(v, device) for v in obj]
        return obj

    def _save_weight_rollback_checkpoint(self):
        # Held on CPU for the rest of training (until replaced or restored), so it
        # doesn't compete with the GPU memory budget of the batch_size*integrator_steps
        # forward pass -- an earlier GPU-resident version OOM'd the LJ-38 log-variance
        # config that otherwise runs fine without this checkpoint held.
        self._weight_rollback_checkpoint = {
            "controller": self._to_device_recursive(self.controller.state_dict(), "cpu"),
            "corrector": self._to_device_recursive(self.corrector.state_dict(), "cpu"),
            "controller_optimizer": self._to_device_recursive(self.controller_optimizer.state_dict(), "cpu"),
            "corrector_optimizer": self._to_device_recursive(self.corrector_optimizer.state_dict(), "cpu"),
        }

    def _restore_weight_rollback_checkpoint(self):
        ckpt = self._weight_rollback_checkpoint
        device = next(self.controller.parameters()).device
        self.controller.load_state_dict(self._to_device_recursive(ckpt["controller"], device))
        self.corrector.load_state_dict(self._to_device_recursive(ckpt["corrector"], device))
        self.controller_optimizer.load_state_dict(self._to_device_recursive(ckpt["controller_optimizer"], device))
        self.corrector_optimizer.load_state_dict(self._to_device_recursive(ckpt["corrector_optimizer"], device))

    def _weight_rollback_check(self, metrics: dict):
        """After each detail-logging eval, snapshot controller+corrector while ESS
        looks healthy, or restore the last snapshot if ESS has collapsed for
        `weight_rollback_patience` consecutive evals. Unlike adapt_trust_region_bound
        (which only ever adjusts epsilon going forward), this undoes the actual
        weight damage from an overly-aggressive gamma step.
        """
        if not self.weight_rollback_enabled:
            return
        ess = metrics.get("RN[h|u]/ess")
        if ess is None:
            return  # buffer not populated yet (e.g. very first eval)

        if ess < self.weight_rollback_ess_threshold:
            self._weight_rollback_bad_count += 1
            if self._weight_rollback_bad_count >= self.weight_rollback_patience:
                if self._weight_rollback_checkpoint is not None:
                    self._restore_weight_rollback_checkpoint()
                    self._weight_rollback_count += 1
                    rank_zero_warn(
                        f"Weight rollback #{self._weight_rollback_count}: "
                        f"RN[h|u]/ess={ess:.3e} < threshold "
                        f"{self.weight_rollback_ess_threshold} for "
                        f"{self._weight_rollback_bad_count} consecutive evals. "
                        "Restored controller/corrector to last known-good checkpoint."
                    )
                    self.fabric.log(
                        "train/weight_rollback_triggered", 1.0, step=self.global_step,
                    )
                    self.fabric.log(
                        "train/weight_rollback_count", self._weight_rollback_count, step=self.global_step,
                    )
                self._weight_rollback_bad_count = 0
        else:
            self._weight_rollback_bad_count = 0
            self._save_weight_rollback_checkpoint()

    def log_final(self):
        report_seed = int(getattr(self.cfg, "report_seed", 12345))
        final_epoch = int(self.current_epoch)
        final_global_step = int(self.global_step)
        final_state = deepcopy(self.sde.state_dict())
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        report: dict[str, Any] = {
            "algorithm": "score_conditioned_tr_iw",
            "seed": int(self.cfg.seed),
            "final_epoch": final_epoch,
            "final_global_step": final_global_step,
            "early_stopped": bool(self.early_stop_requested),
            "early_stop_reason": self.divergence_stopper.reason,
            "capacity": self.model_capacity,
            "selections": {},
        }

        def scalar_metrics(logs: dict[str, Any]) -> dict[str, float]:
            metrics: dict[str, float] = {}
            for key, value in logs.items():
                if isinstance(value, (int, float, np.number)):
                    metrics[key] = float(value)
                elif torch.is_tensor(value) and value.numel() == 1:
                    metrics[key] = float(value.detach().cpu())
            return metrics

        def reset_report_rng() -> None:
            torch.manual_seed(report_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(report_seed)

        def evaluate_loaded_state(
            selection: str,
            selected_epoch: int,
            monitor_metric: str | None,
            online_value: float | None,
        ) -> dict[str, Any]:
            reset_report_rng()
            self.sde.eval()
            integrator = hydra.utils.instantiate(self.cfg.integrator, sde=self.sde)
            data_0 = self.source.sample((self.cfg.eval_samples_final,))
            integral_results = integrator.run_with_logweights(
                integrator=integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            data_1 = integral_results["data_1"]
            self.save_final_samples(data_0, data_1, name=selection)
            results = {
                "data_0": data_0,
                "data_1": data_1,
                "log_Ph_Pref": integral_results["log_Ph_Pref"],
            }
            logs = self.potential.log_details(results, visualize=True)
            logs["eval/raw_path_ess"] = compute_raw_path_ess(
                results["log_Ph_Pref"]
            )
            metrics = scalar_metrics(logs)
            if self.fabric.global_rank == 0 and wandb.run is not None:
                for key, value in logs.items():
                    wandb.run.summary[f"report/{selection}/{key}"] = value
                wandb.run.summary[f"report/{selection}/epoch"] = selected_epoch
            return {
                "selected_epoch": selected_epoch,
                "online_selection_metric": monitor_metric,
                "online_selection_value": online_value,
                "num_samples": int(self.cfg.eval_samples_final),
                "report_seed": report_seed,
                "metrics": metrics,
            }

        try:
            report["selections"]["final"] = evaluate_loaded_state(
                "final", final_epoch, None, None
            )
            for name in self.ckpts.keys():
                ckpt_path = Path(self.checkpoint_directory) / f"{name}_best.pt"
                if not ckpt_path.is_file():
                    raise FileNotFoundError(f"Selected checkpoint is missing: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                info = ckpt["_checkpoint"]
                status = info["status"]
                self.sde.load_state_dict(ckpt["fbsde"])
                selection = f"best_{name}"
                report["selections"][selection] = evaluate_loaded_state(
                    selection=selection,
                    selected_epoch=int(
                        status.get(
                            "epoch",
                            int(status["step"]) * self.cfg.log_detail_interval_epoch,
                        )
                    ),
                    monitor_metric=name.replace("_", "/", 1),
                    online_value=float(status["value"]),
                )
                if self.fabric.global_rank == 0 and not self.cfg.keep_best_checkpoints:
                    ckpt_path.unlink(missing_ok=True)
        finally:
            self.sde.load_state_dict(final_state)
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available() and cuda_rng_state:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        if self.fabric.global_rank == 0:
            report_path = Path(self.checkpoint_directory) / "checkpoint_evaluations.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
            if wandb.run is not None:
                wandb.run.summary["seed"] = int(self.cfg.seed)
                wandb.run.summary["report/path"] = str(report_path)
        wandb.finish()
        
    # 既存の populate_buffer_log_weights を置き換えるか、あるいは
    # optimize_gamma 専用のこの関数を追加してください。
    @torch.no_grad()
    @torch._dynamo.disable()
    def collect_log_weights_for_optimization(self):
        """
        Gamma最適化のために log weights を収集し、単一の巨大なNumPy配列として返します。
        BatchBufferやDataLoaderを経由せず、メモリオーバーヘッドを最小化します。
        """
        self.integrator.eval()
        
        # 設定の読み出し
        total_samples = self.cfg.num_samples_for_dual_loss[0]
        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        inference_batch_size = self.cfg.inference_batch_size
        
        paths_per_batch = inference_batch_size // batch_size_path
        num_iters = total_samples // paths_per_batch
        
        all_log_weights_np = []

        # プログレスバー（必要に応じて）
        iterator = range(num_iters)
        if self.fabric.global_rank == 0 and self.cfg.progress_bar:
             iterator = tqdm(iterator, desc="Collecting weights", leave=False)

        for _ in iterator:
            # 念のためループごとにメモリ掃除
            torch.cuda.empty_cache()
            # データの準備
            data_0 = self.source.replicate_sample((paths_per_batch,), batch_size_path)
            
            # 積分実行
            integral_results = self.integrator.run_with_logweights(
                integrator=self.integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=False,
            )
            
            # Log weights 計算: dP^u/dP^h
            # shape: (paths_per_batch * batch_size_path, ) -> (paths_per_batch, batch_size_path)
            log_Pu_Ph = -integral_results["log_Ph_Pref"].reshape(paths_per_batch, batch_size_path)
            
            # 【重要】ここで即座に CPU NumPy 配列に変換してリストに追加
            # これにより GPU メモリと PyTorch の計算グラフ用メモリを即解放
            all_log_weights_np.append(log_Pu_Ph.detach().cpu().numpy().astype(np.float64))
            
            # 明示的にキャッシュクリア（念のため）
            del integral_results, log_Pu_Ph, data_0
        
        self.integrator.train()

        # 最後に結合 (100万サンプル程度なら数MB〜数十MBなので全く問題なし)
        if len(all_log_weights_np) > 0:
            combined = np.concatenate(all_log_weights_np, axis=0)
        else:
            combined = np.array([])

        # Multi-GPU (world_size > 1): each rank has so far only collected its
        # OWN local num_samples_for_dual_loss batch. Without this gather,
        # optimize_gamma() would solve a SEPARATE dual optimization per rank,
        # each landing on a slightly different gamma from a different random
        # sample -- i.e. different ranks training under different effective
        # loss functions, on top of not even getting the benefit of a larger
        # combined sample for the (already known to be noise-sensitive) dual
        # solve. Gathering here means every rank calls optimize_gamma() on
        # the IDENTICAL world_size-times-larger pool, guaranteeing they all
        # compute the same gamma while also directly reducing dual-solve
        # sampling noise.
        if self.fabric.world_size > 1 and combined.size > 0:
            local_tensor = self.fabric.to_device(torch.from_numpy(combined).float())
            gathered = self.fabric.all_gather(local_tensor)  # (world_size, total_samples, batch_size_path)
            combined = gathered.reshape(-1, gathered.shape[-1]).detach().cpu().numpy().astype(np.float64)

        return combined

    def adapt_trust_region_bound(self) -> None:
        """Adaptively adjust trust_region_bound via bisection between the tightest
        epsilon known to make gamma inactive (~1) and the loosest epsilon known to
        leave gamma stuck (~0), using observed training stability (logged energy)
        as an additional safety check -- not gamma alone, and not fixed-factor
        growth/shrink steps.

        Empirically (see lj38 EGNN sweep), gamma(epsilon) can be sharply bimodal:
        stuck near 0 below some critical epsilon, then jumping close to 1 just
        above it, with little smooth region in between. A fixed multiplicative
        growth factor (e.g. 3x per outer loop) can easily jump straight over a
        narrow "good" window instead of landing in it. So instead:

        - Whenever gamma is stuck low, the current epsilon is recorded as a valid
          LOWER bound on the good range (it's demonstrably still too tight).
        - Whenever gamma is inactive/high, the current epsilon is recorded as a
          valid UPPER bound (demonstrably too loose).
        - Once both bounds are known, the next epsilon tried is their geometric
          mean (bisection in log-space) rather than a blind multiplicative step,
          so the search narrows in on the transition instead of jumping past it.
        - Once gamma actually lands in [gamma_low, gamma_high], epsilon is held
          steady -- that's the target, not a waypoint.

        Energy-based rollback is unchanged: if logged energy blows up past
        `trust_region_energy_blowup_factor` times the last known-good baseline,
        the current epsilon is treated as unsafe (and recorded as an upper bound)
        regardless of gamma, shrinking back down with a growth cooldown.

        Bounded to [trust_region_bound_min, trust_region_bound_max]. No-op unless
        trust_region_adaptive=True.
        """
        current_energy = self._last_logged_energy
        unstable = (
            current_energy is not None
            and self._trust_region_reference_energy is not None
            and current_energy > self._trust_region_reference_energy * self.trust_region_energy_blowup_factor
        )

        if unstable:
            self._trust_region_high_bound = (
                self.trust_region_bound
                if self._trust_region_high_bound is None
                else min(self._trust_region_high_bound, self.trust_region_bound)
            )
            new_bound = max(
                self.trust_region_bound * self.trust_region_shrink_factor,
                self.trust_region_bound_min,
            )
            rank_zero_warn(
                f"Trust region: energy blew up ({current_energy:.4g} vs reference "
                f"{self._trust_region_reference_energy:.4g}) -- treating epsilon as unsafe "
                f"(upper bound <= {self._trust_region_high_bound:.4g}), shrinking "
                f"{self.trust_region_bound:.4g} -> {new_bound:.4g} and blocking growth for "
                f"{self.trust_region_patience} outer loop(s)."
            )
            self.trust_region_bound = new_bound
            self._trust_region_stuck_low_count = 0
            self._trust_region_stuck_high_count = 0
            self._trust_region_growth_cooldown = self.trust_region_patience
            return

        if current_energy is not None and (
            self._trust_region_reference_energy is None or current_energy < self._trust_region_reference_energy
        ):
            self._trust_region_reference_energy = current_energy

        if self._trust_region_growth_cooldown > 0:
            self._trust_region_growth_cooldown -= 1

        if self.trust_region_gamma_low <= self.gamma <= self.trust_region_gamma_high:
            # In the target range -- hold steady, this is the goal, not a waypoint.
            self._trust_region_stuck_low_count = 0
            self._trust_region_stuck_high_count = 0
            rank_zero_warn(
                f"Trust region: gamma={self.gamma:.3e} is within the target range "
                f"[{self.trust_region_gamma_low}, {self.trust_region_gamma_high}] -- "
                f"holding epsilon at {self.trust_region_bound:.4g}."
            )
            return

        if self.gamma > self.trust_region_gamma_high:
            self._trust_region_stuck_high_count += 1
            self._trust_region_stuck_low_count = 0
            if self._trust_region_stuck_high_count >= self.trust_region_patience:
                self._trust_region_high_bound = (
                    self.trust_region_bound
                    if self._trust_region_high_bound is None
                    else min(self._trust_region_high_bound, self.trust_region_bound)
                )
                if self._trust_region_low_bound is not None:
                    new_bound = math.sqrt(self._trust_region_low_bound * self._trust_region_high_bound)
                    method = "bisecting"
                else:
                    new_bound = max(
                        self.trust_region_bound * (self.trust_region_shrink_factor ** 2),
                        self.trust_region_bound_min,
                    )
                    method = "shrinking aggressively (no lower bound yet)"
                rank_zero_warn(
                    f"Trust region slack (gamma={self.gamma:.3e} > {self.trust_region_gamma_high} "
                    f"for {self._trust_region_stuck_high_count} outer loop(s), upper bound now "
                    f"<= {self._trust_region_high_bound:.4g}): {method} epsilon "
                    f"{self.trust_region_bound:.4g} -> {new_bound:.4g} and blocking growth for "
                    f"{self.trust_region_patience} outer loop(s)."
                )
                self.trust_region_bound = new_bound
                self._trust_region_stuck_high_count = 0
                self._trust_region_growth_cooldown = self.trust_region_patience
        elif self.gamma < self.trust_region_gamma_low:
            self._trust_region_stuck_high_count = 0
            self._trust_region_stuck_low_count += 1
            if self._trust_region_stuck_low_count >= self.trust_region_patience:
                self._trust_region_low_bound = (
                    self.trust_region_bound
                    if self._trust_region_low_bound is None
                    else max(self._trust_region_low_bound, self.trust_region_bound)
                )
                if self._trust_region_growth_cooldown > 0:
                    rank_zero_warn(
                        f"Trust region stuck (gamma={self.gamma:.3e}, lower bound now >= "
                        f"{self._trust_region_low_bound:.4g}) but growth is on cooldown "
                        f"({self._trust_region_growth_cooldown} outer loop(s) remaining) after a "
                        "recent instability/slack event -- holding epsilon steady this loop."
                    )
                else:
                    if self._trust_region_high_bound is not None:
                        new_bound = math.sqrt(self._trust_region_low_bound * self._trust_region_high_bound)
                        method = "bisecting"
                    else:
                        new_bound = min(
                            self.trust_region_bound * self.trust_region_growth_factor,
                            self.trust_region_bound_max,
                        )
                        method = "growing (no upper bound yet)"
                    if new_bound != self.trust_region_bound:
                        rank_zero_warn(
                            f"Trust region stuck (gamma={self.gamma:.3e} < {self.trust_region_gamma_low} "
                            f"for {self._trust_region_stuck_low_count} outer loop(s), lower bound now >= "
                            f"{self._trust_region_low_bound:.4g}): {method} epsilon "
                            f"{self.trust_region_bound:.4g} -> {new_bound:.4g}"
                        )
                    self.trust_region_bound = new_bound
                self._trust_region_stuck_low_count = 0

    def optimize_gamma(self, max_lm=1e6) -> float:
        # 1. データを収集 (NumPy配列)
        log_weights_np = self.collect_log_weights_for_optimization()
        
        if log_weights_np.size == 0:
            rank_zero_warn("No samples collected for gamma optimization. Skipping.")
            return self.gamma, 0.0

        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        
        # scipy.special.logsumexp を使用
        from scipy.special import logsumexp

        # 2. 高速計算用ローカル関数 (純粋な NumPy 計算)
        def calc_V(lm):
            alpha = -1.0 / (1.0 + lm)
            # axis=1 でパスごとの和を取る
            term = logsumexp(alpha * log_weights_np, axis=1) - math.log(batch_size_path)
            V = (1.0 + lm) * (-np.mean(term))
            return V

        # 3. 最適化対象の関数
        def objective(lm):
             V = calc_V(lm)
             return -V + lm * self.trust_region_bound

        # 4. 最適化実行 (scipy)
        # bounds 内でスカラー最適化
        res = minimize_scalar(objective, bounds=(0, max_lm), method="Bounded")
        best_lambda = res.x
        best_gamma = 1.0 / (1.0 + best_lambda)
        best_gamma = max(best_gamma, self.gamma_floor)

        val_0 = calc_V(0.0)

        return best_gamma, val_0

    def compute_V0_for_fixed_gamma(self) -> float:
        """V0 counterpart to optimize_gamma() for gamma_schedule=="fixed": gamma
        itself never changes, but V0 (the reference scale used to build the
        buffer's rescaling factor r = min(-(V0/raw_log_Ph_Pref_mean), 1.0) in
        populate_buffer_adjoint_matching) still needs to track the CURRENT
        samples' log-weight scale. Left at its __init__ default (1e-9), r
        collapses to ~0 every time, degenerating the stored log_Ph_Pref to ~0
        for every sample -- which breaks the RN[h|u]/ess diagnostic, and for
        fc_loss_mode=="cross_entropy" also makes _compute_iw's importance
        weights trivially 1.0 for every sample (a silent no-op, not genuine
        importance weighting).
        """
        log_weights_np = self.collect_log_weights_for_optimization()
        if log_weights_np.size == 0:
            rank_zero_warn("No samples collected for V0 update. Keeping previous V0.")
            return self.V0

        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        from scipy.special import logsumexp

        lm = self.lm  # (1 - gamma) / gamma, matching optimize_gamma's parametrization
        if self.gamma == 0:
            return float(np.mean(log_weights_np))  # Continuous gamma -> 0 limit.
        alpha = -1.0 / (1.0 + lm)
        term = logsumexp(alpha * log_weights_np, axis=1) - math.log(batch_size_path)
        V = (1.0 + lm) * (-np.mean(term))
        return V

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # NOTE: changing matmul precision may result in mean drift
    logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
    use_wandb_logger = "wandb" in logger_target
    if use_wandb_logger:
        os.environ["WANDB_MODE"] = str(cfg.wandb_mode)
        os.environ["WANDB_BASE_URL"] = str(cfg.wandb_host)
    torch.set_float32_matmul_precision("high")
    trainer = GammaASBSModule(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
