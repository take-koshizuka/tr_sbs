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

from omegaconf import OmegaConf, open_dict

from tr_asbs.process.sde import ControlledSDE, ControlledFBSDE
from tr_asbs.utils.training import EMA, copy_model
from tr_asbs.data.buffer import BatchBuffer
from tr_asbs.utils.eval_logweight import compute_reverse_ess, normalize_log_weights
from tr_asbs.utils.gaussian_utils import gaussian_log_prob
from tr_asbs.utils.helper import Checkpointer, CheckpointConfig

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
    OFFPOLICY_METADATA_KEYS = {
        "trajectory",
        "proposal_log_fwd",
        "proposal_step",
        "weight_step",
    }

    ############################################################################
    # Initialization
    ############################################################################

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        random_hash = uuid.uuid4().hex[:8]
        self.group_id = str(stable_group_id(cfg, exclude=("name"))) + f"{random_hash}"
        logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
        if "csvlogger" in logger_target or "csv_logs" in logger_target:
            with open_dict(cfg.fabric.loggers):
                for key in ("entity", "project", "tags", "group", "_delete_"):
                    if key in cfg.fabric.loggers:
                        del cfg.fabric.loggers[key]
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
        self.refresh_adjoint_buffer_weights = bool(
            getattr(cfg, "refresh_adjoint_buffer_weights", True)
        )
        self.refresh_adjoint_buffer_targets = bool(
            getattr(cfg, "refresh_adjoint_buffer_targets", True)
        )
        self.drop_legacy_adjoint_samples = bool(
            getattr(cfg, "drop_legacy_adjoint_samples", True)
        )
        self.offpolicy_refresh_batch_size = int(
            getattr(cfg, "offpolicy_refresh_batch_size", cfg.inference_batch_size)
        )
        self.adjoint_path_dtype = getattr(cfg, "adjoint_path_dtype", "float32")
        self.u_outer_step = 0
        
        # Lightning fabric setup
        self.fabric = hydra.utils.instantiate(cfg.fabric, loggers=wandb_logger)
        self.fabric.launch()
        self.rank_seed = cfg.seed ^ self.fabric.global_rank
        seed_everything(self.rank_seed, workers=False)

        with self.fabric.init_module():
            self.potential = hydra.utils.instantiate(cfg.potential, device=self.fabric.device)
            # Models
            controller = hydra.utils.instantiate(cfg.controller)
            corrector = hydra.utils.instantiate(cfg.corrector)
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
            self.sde = ControlledFBSDE(
                base_sde=self.base_sde,
                u_controller=self.controller, 
                h_controller=self.corrector
            )
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            # Potential and cost function
            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential, corrector=self.corrector
            )
            
            self.prev_integrator = deepcopy(self.integrator)
            
        # Process the training curriculum
        self.max_outer_loops = cfg.max_outer_loops
        self.adjoint_matching_epochs = cfg.adjoint_matching_epochs
        self.corrector_matching_epochs = cfg.corrector_matching_epochs
        self.total_epochs = self.max_outer_loops * (
            self.adjoint_matching_epochs + self.corrector_matching_epochs
        )
        
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
                    mode="min",                 # loss なら "min"
                ),
                make_payload=lambda: {
                    "fbsde": self.sde.state_dict(),
                },
            )
        
        if wandb.run is not None:
            wandb.run.summary["group_id"] = self.group_id
            wandb.run.summary["checkpoint_directory"] = str(self.checkpoint_directory)
        
    @property
    def lm(self) -> float:
        """lagrangian multiplier expressed from gamma. Kept for compatibility."""
        return (1 - self.gamma) / self.gamma

    def _path_storage_dtype(self):
        dtype_name = str(self.adjoint_path_dtype).lower()
        if dtype_name in {"float32", "torch.float32"}:
            return torch.float32
        if dtype_name in {"float64", "torch.float64"}:
            return torch.float64
        if dtype_name in {"none", "preserve", "null"}:
            return None
        raise ValueError(f"Unsupported adjoint_path_dtype: {self.adjoint_path_dtype}")

    def _make_training_collate_fn(self, dataset):
        def collate_fn(batch):
            if not batch:
                return {}
            keys = [
                key for key in batch[0].keys()
                if key not in self.OFFPOLICY_METADATA_KEYS
            ]
            return {key: torch.stack([item[key] for item in batch]) for key in keys}

        return collate_fn

    def _sigma_step_expand(self, sigma_step, batch_size, dim, device, dtype):
        if sigma_step.numel() == 1:
            return torch.full(
                (batch_size, dim), sigma_step.item(), device=device, dtype=dtype
            )
        return sigma_step.to(device=device, dtype=dtype)

    def _forward_mean(self, integrator, data, time, dt):
        return data + integrator.sde.drift(time, data) * dt

    def _backward_mean(self, integrator, data, time, dt):
        return data + integrator.sde.bwd_drift(time, data) * dt

    @torch.no_grad()
    def _replay_path_logweights(self, trajectory, proposal_log_fwd=None):
        """Recompute log dP^h/dP^q and log dP^u/dP^q on saved paths."""
        integrator = self.prev_integrator
        timesteps = integrator.timesteps.to(trajectory.device)
        batch_size = trajectory.shape[0]
        dim = trajectory[:, 0].reshape(batch_size, -1).shape[1]
        device = trajectory.device

        current_log_fwd = torch.zeros(batch_size, device=device)
        current_log_bwd = torch.zeros(batch_size, device=device)

        for step_idx in range(len(timesteps) - 1):
            time = timesteps[step_idx]
            time_next = timesteps[step_idx + 1]
            dt = time_next - time
            data = trajectory[:, step_idx]
            next_data = trajectory[:, step_idx + 1]

            fwd_mean = self._forward_mean(integrator, data, time, dt)
            bwd_mean = self._backward_mean(integrator, next_data, time_next, dt)

            diff_sq = (
                integrator.sde.base_sde.diffsquare_integral(time_next)
                - integrator.sde.base_sde.diffsquare_integral(time)
            )
            sigma_step = diff_sq.clamp(min=1e-20).sqrt()
            sigma_step_expand = self._sigma_step_expand(
                sigma_step, batch_size, dim, device, next_data.dtype
            )

            current_log_fwd += gaussian_log_prob(
                next_data.reshape(batch_size, dim),
                fwd_mean.reshape(batch_size, dim),
                sigma_step_expand,
                dim,
            )
            current_log_bwd += gaussian_log_prob(
                data.reshape(batch_size, dim),
                bwd_mean.reshape(batch_size, dim),
                sigma_step_expand,
                dim,
            )

        if proposal_log_fwd is None:
            proposal_log_fwd = current_log_fwd.detach()
        else:
            proposal_log_fwd = proposal_log_fwd.reshape(batch_size).to(device)

        data_0 = trajectory[:, 0]
        data_1 = trajectory[:, -1]
        log_Pu_Pref = current_log_fwd - proposal_log_fwd
        log_Ph_Pref = (
            self.potential.log_prob(data_1)
            + current_log_bwd
            - self.source.log_prob(data_0)
            - proposal_log_fwd
        )

        return {
            "log_Ph_Pref": log_Ph_Pref,
            "log_Pu_Pref": log_Pu_Pref,
            "proposal_log_fwd": proposal_log_fwd,
        }

    def _assign_buffer_slice(self, buffer, key, start, values):
        values_cpu = list(torch.unbind(values.detach().cpu(), dim=0))
        buffer.storage[key][start:start + len(values_cpu)] = values_cpu

    def _adjoint_buffer_has_offpolicy_terms(self):
        if len(self.adjoint_buffer) == 0:
            return True
        return all(
            key in self.adjoint_buffer.storage
            and len(self.adjoint_buffer.storage[key]) == len(self.adjoint_buffer)
            for key in ("trajectory", "proposal_log_fwd")
        )

    @torch.no_grad()
    def refresh_adjoint_buffer_offpolicy_weights(self):
        if not self.refresh_adjoint_buffer_weights or len(self.adjoint_buffer) == 0:
            return

        buffer = self.adjoint_buffer
        if not self._adjoint_buffer_has_offpolicy_terms():
            message = (
                "Adjoint buffer contains legacy samples without trajectory/proposal "
                "log density; dropping them for off-policy replay."
            )
            if self.drop_legacy_adjoint_samples:
                rank_zero_warn(message)
                buffer.storage.clear()
                return
            raise RuntimeError(message)

        self.prev_integrator.eval()
        is_init_stage = self.current_epoch < self.cfg.adjoint_matching_epochs
        batch_size = max(1, self.offpolicy_refresh_batch_size)
        total = len(buffer)
        refreshed = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            trajectory = torch.stack(buffer.storage["trajectory"][start:end])
            proposal_log_fwd = torch.stack(
                buffer.storage["proposal_log_fwd"][start:end]
            )
            trajectory = self.fabric.to_device(trajectory)
            proposal_log_fwd = self.fabric.to_device(proposal_log_fwd)

            replay_results = self._replay_path_logweights(
                trajectory, proposal_log_fwd=proposal_log_fwd
            )
            self._assign_buffer_slice(
                buffer, "log_Ph_Pref", start, replay_results["log_Ph_Pref"]
            )
            self._assign_buffer_slice(
                buffer, "log_Pu_Pref", start, replay_results["log_Pu_Pref"]
            )

            if self.refresh_adjoint_buffer_targets:
                data_1 = trajectory[:, -1]
                terminal_cost_results = self.terminal_cost(
                    data_1=data_1,
                    is_init_stage=is_init_stage,
                    return_energy=self.cfg.log_buffer_energy,
                    return_grad_norm=self.cfg.log_buffer_grad_norm,
                )
                self._assign_buffer_slice(
                    buffer,
                    "terminal_cost",
                    start,
                    terminal_cost_results["terminal_cost"],
                )
                time_1 = self.fabric.to_device(
                    torch.ones(data_1.shape[0], dtype=torch.float)
                )
                control_1 = self.prev_integrator.sde.u_controller(
                    time_1, data_1
                ).detach()
                self._assign_buffer_slice(buffer, "control_1", start, control_1)

                if self.cfg.log_buffer_energy and "energy" in buffer.storage:
                    self._assign_buffer_slice(
                        buffer, "energy", start, terminal_cost_results["energy"]
                    )
                if self.cfg.log_buffer_grad_norm and "grad_norm" in buffer.storage:
                    self._assign_buffer_slice(
                        buffer, "grad_norm", start, terminal_cost_results["grad_norm"]
                    )

            weight_step = torch.full(
                (end - start,), self.u_outer_step, dtype=torch.long
            )
            self._assign_buffer_slice(buffer, "weight_step", start, weight_step)
            refreshed += end - start

        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                "train/offpolicy_refreshed_adjoint_samples",
                refreshed,
                step=self.global_step,
            )
    
    def inner_loop_u(self):
        current_task = "adjoint_matching"
        self.prev_integrator = copy_model(self.prev_integrator, self.integrator)
        self.u_outer_step += 1
        self.refresh_adjoint_buffer_offpolicy_weights()
        for j in range(self.cfg.adjoint_matching_epochs):
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
            
            if (self.current_epoch % self.cfg.log_detail_interval_epoch == 0):
                metrics = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch)
                # Check for early stopping
                for name in self.cfg.monitor_values:
                    name_safe = name.replace("/", "_")
                    info = self.ckpts[name_safe].update(
                        value=metrics[name],
                        step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                        tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                        extra_status={"eval_step": metrics["eval_step"]},
                    )
                del metrics
                torch.cuda.empty_cache()
            
    def inner_loop_h(self):
        current_task = "corrector_matching"
        for j in range(self.cfg.corrector_matching_epochs):
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
            
            if (self.current_epoch % self.cfg.log_detail_interval_epoch == 0):
                metrics = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch)
                for name in self.cfg.monitor_values:
                    name_safe = name.replace("/", "_")
                    info = self.ckpts[name_safe].update(
                        value=metrics[name],
                        step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                        tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                        extra_status={"eval_step": metrics["eval_step"]},
                    )
                del metrics
                torch.cuda.empty_cache()
    ############################################################################
    # Training loop structure
    ############################################################################

    def fit(self):
        """Main training loop for the ASBS."""
        self.global_step = 0
        self.current_epoch = 0
        self.u_outer_step = 0
        if (
            hasattr(self.cfg, "pretrained_controller_checkpoint")
            and self.cfg.pretrained_controller_checkpoint
        ):
            self.load_pretrained_controller_checkpoint()
            
        if self.cfg.resume:
            self.load_checkpoint()
            
        metrics = self.log_details(0)
        for name in self.cfg.monitor_values:
            name_safe = name.replace("/", "_")
            info = self.ckpts[name_safe].update(
                value=metrics[name],
                step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                extra_status={"eval_step": metrics["eval_step"]},
            )

        for outer_step in range(self.max_outer_loops):
            if self.gamma_schedule == "trust_region":
                self.gamma, self.V0 = self.optimize_gamma()
            self.inner_loop_u()
            self.inner_loop_h()

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
        is_init_stage = self.current_epoch < self.cfg.adjoint_matching_epochs
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
        self.prev_integrator.eval()

        max_abs_new_log_pu_pref = None
        path_dtype = self._path_storage_dtype()

        # Populate the buffer with data from the controlled SDE integration
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            trajectory_list = self.prev_integrator.run(
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=True,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            trajectory = torch.stack(trajectory_list, dim=1)
            data_1 = trajectory[:, -1]
            replay_results = self._replay_path_logweights(trajectory)
            new_log_pu_pref = replay_results["log_Pu_Pref"].abs().max().detach()
            max_abs_new_log_pu_pref = (
                new_log_pu_pref
                if max_abs_new_log_pu_pref is None
                else torch.maximum(max_abs_new_log_pu_pref, new_log_pu_pref)
            )

            trajectory_to_store = trajectory
            if path_dtype is not None:
                trajectory_to_store = trajectory_to_store.to(path_dtype)

            populate_data = {
                "data_0": data_0.detach().cpu(),
                "data_1": data_1.detach().cpu(),
                "log_Ph_Pref": replay_results["log_Ph_Pref"].detach().cpu(),
                "log_Pu_Pref": replay_results["log_Pu_Pref"].detach().cpu(),
                "trajectory": trajectory_to_store.detach().cpu(),
                "proposal_log_fwd": replay_results["proposal_log_fwd"].detach().cpu(),
                "proposal_step": torch.full(
                    (self.cfg.inference_batch_size,),
                    self.u_outer_step,
                    dtype=torch.long,
                    device=data_0.device,
                ),
                "weight_step": torch.full(
                    (self.cfg.inference_batch_size,),
                    self.u_outer_step,
                    dtype=torch.long,
                    device=data_0.device,
                ),
            }

            terminal_cost_results = self.terminal_cost(
                data_1=data_1,
                is_init_stage=is_init_stage,
                return_energy=self.cfg.log_buffer_energy,
                return_grad_norm=self.cfg.log_buffer_grad_norm,
            )
            
            populate_data["terminal_cost"] = terminal_cost_results["terminal_cost"].detach().cpu()
            time_1 = self.fabric.to_device(torch.ones(self.cfg.inference_batch_size, dtype=torch.float))
            populate_data["control_1"] = self.prev_integrator.sde.u_controller(time_1, data_1).detach().cpu()
            
            if self.cfg.log_buffer_energy:
                populate_data["energy"] = terminal_cost_results["energy"].detach().cpu()
            if self.cfg.log_buffer_grad_norm:
                populate_data["grad_norm"] = terminal_cost_results["grad_norm"].detach().cpu()

            buffer.extend(**populate_data)

        # Log the buffer statistics
        # Log buffer median energy
        if self.global_step % self.cfg.log_loss_interval_step == 0:
            if max_abs_new_log_pu_pref is not None:
                self.fabric.log(
                    "train/offpolicy_new_sample_max_abs_log_Pu_Pref",
                    self.fabric.all_reduce(max_abs_new_log_pu_pref, reduce_op="max"),
                    step=self.global_step,
                )

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
                        self.save_samples(data_1)
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
            # Restore the training state of the models
            self.fabric.barrier()
        self.integrator.train()
        
    @torch.no_grad()
    def populate_buffer_corrector_matching(self):
        buffer = self.corrector_buffer
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
            collate_fn=(
                self._make_training_collate_fn(dataset)
                if current_task == "adjoint_matching"
                else dataset.collate_fn
            ),
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
        
        if not torch.isfinite(loss):
            return False
        
        self.fabric.backward(loss)
        try:
            self.fabric.clip_gradients(model, optimizer, max_norm=self.cfg.gradient_clip_val)
        except RuntimeError as e:
            if "non-finite" in str(e):
                optimizer.zero_grad(set_to_none=True)
                return False
            raise
        optimizer.step()
        
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
        target = self.gamma * (terminal_cost + control_1) - control_1

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
        ) * (1 - 2e-3) + 1e-3

        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        # 3つを dim0 でまとめる
        time_all = torch.cat([time_0, time_1, time], dim=0)          # (3B,)
        data_all = torch.cat([data_0, data_1, data_t], dim=0)        # (3B, ...)

        # target, iw も 3回ぶんに拡張（target は data と同じ batch 次元を持つ想定）
        target_all = torch.cat([target, target, target], dim=0)      # (3B, ...)
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

    def corrector_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the corrector matching and return the loss.
        Equation 15 in the ASBS paper.
        """
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]

        batch_size = self.cfg.train_batch_size

        # t ~ U(1e-3, 1-1e-3)
        time = self.fabric.to_device(torch.rand(batch_size) * (1 - 2e-3) + 1e-3)
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

        coef_t = 1 + self.base_sde.diffusion(time).pow(2) * (1 - time) / time  # (B, ...?) 想定は (B,)

        loss_t = (
            (h_t - score_t)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)                                  # (B,)
            .div(coef_t.squeeze())                     # (B,)
            .mean()
        )

        loss_1 = (
            (h_1 - score_1)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)
            .mean()
        )
        
        loss = loss_t + loss_1
        return dict(loss=loss, loss_t=loss_t.detach().cpu(), loss_1=loss_1.detach().cpu())

    ############################################################################
    # Checkpoint management
    ############################################################################

    def load_pretrained_controller_checkpoint(self):
        """Load a pretrained controller checkpoint if specified in the config."""
        checkpoint_path = Path(self.cfg.pretrained_controller_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained controller checkpoint not found: {checkpoint_path}"
            )
        state = {"controller": self.controller}
        self.fabric.load(checkpoint_path, state)

    def load_checkpoint(self):
        """Load the latest checkpoint if it exists."""
        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            "controller_optimizer": self.controller_optimizer,
            "corrector_optimizer": self.corrector_optimizer,
        }
        checkpoint_directory = Path(self.checkpoint_directory)
        if not checkpoint_directory.exists():
            return
        checkpoint_files = list(checkpoint_directory.glob("*.pt"))
        if checkpoint_files:
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            remainder = self.fabric.load(checkpoint_path, state)
            self.global_step = remainder.pop("global_step")
            self.current_epoch = remainder.pop("current_epoch")
            self.u_outer_step = remainder.pop("u_outer_step", self.u_outer_step)

    def save_checkpoint(self, only_controller: bool = False):
        """Save the current state of the model and optimizer to a checkpoint."""
        checkpoint_directory = Path(self.checkpoint_directory)
        if not checkpoint_directory.exists():
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / f"epoch_{self.current_epoch:04d}.pt"
        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            "controller_optimizer": self.controller_optimizer,
            "corrector_optimizer": self.corrector_optimizer,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "u_outer_step": self.u_outer_step,
        }
        self.fabric.save(checkpoint_path, state)
    
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
            "data_0": data_0[:self.cfg.eval_samples].detach().cpu(),
            "data_1": integral_results['data_1'][:self.cfg.eval_samples].detach().cpu(),
            "log_Ph_Pref": integral_results['log_Ph_Pref'][:self.cfg.eval_samples].detach().cpu(),
            "log_Pu_Pref": integral_results['log_Pu_Pref'][:self.cfg.eval_samples].detach().cpu(),
            'log_RN[u_next|u]_eff': integral_results['log_Pref_Pu_det'][:self.cfg.eval_samples].detach().cpu(),
            'log_RN[u_next|u]': -integral_results['log_Pu_Pref'][:self.cfg.eval_samples].detach().cpu(),
            'predict_drift_fn': self.sde.drift,
        }
        
        if self.current_epoch > 0:
            # バッファが空でないか確認
            if len(self.adjoint_buffer) > 0:
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
            else:
                # バッファが空の場合のハンドリング（必要なら）
                pass
        force_log_gt = getattr(self.potential, "name", "") == "ring_gmm"

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
        
        logs["epoch"] = self.current_epoch
        logs["eval_step"] = step

        self.fabric.log_dict(logs, step=self.global_step)
        
        logs.update({"data_0": data_0.detach().cpu(), "data_1": integral_results['data_1'].detach().cpu()})
        
        del results, integral_results
        return logs
    
    def log_final(self):
        final_logs = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch, visualize=True, eval_samples=self.cfg.eval_samples * 2)
        data_0, data_1 = final_logs.pop("data_0"), final_logs.pop("data_1")
        self.save_final_samples(data_0, data_1, name=f"last_epoch")
        # rank0だけ summary に確定値を書き込む
        if self.fabric.global_rank == 0 and wandb.run is not None:
            wandb.run.summary["seed"] = int(self.cfg.seed)
            for key, value in final_logs.items():
                wandb.run.summary[f"final/{key}"] = value
            wandb.run.summary["final/epoch"] = int(self.current_epoch)
            wandb.run.summary["final/global_step"] = int(self.global_step)

        for name in self.ckpts.keys():
            ckpt_path = Path(self.checkpoint_directory) / f"{name}_best.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            info = ckpt["_checkpoint"]
            step = info["status"]["step"]
            best_val = info["status"]["value"]
            self.sde.load_state_dict(ckpt["fbsde"])
            self.sde.eval()
            
            integrator = hydra.utils.instantiate(self.cfg.integrator, sde=self.sde)
            data_0 = self.source.sample((self.cfg.eval_samples * 2,))
            data_1 = integrator.run(
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            self.save_final_samples(data_0, data_1, name=f"best_{name}")
            results = {
                "data_0": data_0[:self.cfg.eval_samples],
                "data_1": data_1[:self.cfg.eval_samples]
            }
            
            logs = self.potential.log_details(results, visualize=True)
            
            if self.fabric.global_rank == 0 and wandb.run is not None:
                for key, value in logs.items():
                    wandb.run.summary[f"best_{name}/{key}"] = value
                wandb.run.summary[f"best_{name}/step"] = int(step)
                wandb.run.summary[f"best_{name}/value"] = float(best_val)
            
            # ===== checkpoint 削除 =====
            if self.fabric.global_rank == 0:
                ckpt_path.unlink(missing_ok=True)
             
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
            return np.concatenate(all_log_weights_np, axis=0)
        else:
            return np.array([])

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
        
        val_0 = calc_V(0.0)
            
        return best_gamma, val_0

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # NOTE: changing matmul precision may result in mean drift
    logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
    use_wandb_logger = "wandb" in logger_target
    if use_wandb_logger:
        os.environ["WANDB_BASE_URL"] = str(cfg.wandb_host)
        os.environ["WANDB_API_KEY"] = str(cfg.wandb_key)
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    torch.set_float32_matmul_precision("high")
    trainer = GammaASBSModule(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
