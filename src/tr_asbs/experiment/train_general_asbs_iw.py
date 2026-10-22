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

from omegaconf import OmegaConf

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

class GeneralASBSModule:
    """Module for ASBS training."""

    TASKS: list[str] = ["adjoint_matching", "corrector_matching"]

    ############################################################################
    # Initialization
    ############################################################################

    def __init__(self, cfg: DictConfig):
        super().__init__()
        if getattr(cfg, "reference_preset", None) is not None or str(cfg.base_sde._target_).split(".")[-1] not in ("VESDE", "EDMSDE", "PISCosineSDE") or str(cfg.integrator._target_).split(".")[-1] not in ("ExponentialIntegrator", "EulerMaruyamaIntegrator"):
            raise ValueError("GeneralASBS consistency losses use legacy kernels; use GammaASBSModule for new references/solvers")
        self.cfg = cfg
        random_hash = uuid.uuid4().hex[:8]
        self.group_id = str(stable_group_id(cfg, exclude=("name"))) + f"{random_hash}"
        logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
        if "wandb" in logger_target and "group" in cfg.fabric.loggers:
            cfg.fabric.loggers.group = stable_group_id(cfg)
        
        logger_cfg = cfg.fabric.loggers
        logger_target = str(getattr(logger_cfg, "_target_", "")).lower()
        if "csvlogger" in logger_target:
            # Guard against merged Wandb-only keys (e.g., entity/project) that CSVLogger does not accept.
            allowed_keys = {
                "_target_",
                "save_dir",
                "name",
                "version",
                "prefix",
                "flush_logs_every_n_steps",
            }
            logger_cfg = OmegaConf.create(
                {k: logger_cfg[k] for k in allowed_keys if k in logger_cfg}
            )
        wandb_logger = hydra.utils.instantiate(logger_cfg)
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
        self.backward_loss_mode = str(getattr(cfg, "backward_loss_mode", "score_matching")).lower()
        self.backward_use_adjoint = False
        if self.backward_loss_mode in {"adjoint", "adjoint_matching"}:
            self.backward_use_adjoint = True
            default_terms = []
        elif self.backward_loss_mode in {"score", "score_matching", "score_mathing"}:
            default_terms = ["score_matching"]
        elif self.backward_loss_mode in {"consistency", "consistency_loss", "transition", "transition_matching", "mahalanobis"}:
            default_terms = ["consistency_loss"]
        elif self.backward_loss_mode in {"full_path_consistency", "consistency_loss_full_path"}:
            default_terms = ["consistency_loss_full_path"]
        elif self.backward_loss_mode in {"consistency_log_variance", "consistency_var_full_path"}:
            default_terms = ["consistency_var_full_path"]
        else:
            raise ValueError(
                f"Invalid backward_loss_mode: {self.backward_loss_mode}. "
                "Choose from {'score_matching', 'adjoint_matching', 'consistency_loss', 'consistency_loss_full_path', 'consistency_log_variance'}."
            )

        allowed_terms = {
            "score_matching",
            "consistency_loss",
            "consistency_loss_full_path",
            "consistency_var_full_path",
        }
        backward_loss_terms = getattr(cfg, "backward_loss_terms", None)
        if backward_loss_terms:
            parsed_terms = []
            for term in backward_loss_terms:
                t = str(term).lower()
                if t in {"score", "score_matching", "score_mathing"}:
                    parsed_terms.append("score_matching")
                elif t in {"consistency", "consistency_loss", "transition", "transition_matching", "mahalanobis"}:
                    parsed_terms.append("consistency_loss")
                elif t in {"full_path_consistency", "consistency_loss_full_path"}:
                    parsed_terms.append("consistency_loss_full_path")
                elif t in {"consistency_log_variance", "consistency_var_full_path"}:
                    parsed_terms.append("consistency_var_full_path")
                else:
                    raise ValueError(
                        f"Invalid backward loss term: {term}. "
                        "Allowed terms are score_matching, consistency_loss, consistency_loss_full_path, consistency_var_full_path."
                    )
            self.backward_terms = sorted(set(parsed_terms))
            if self.backward_use_adjoint:
                raise ValueError("adjoint_matching cannot be combined with backward_loss_terms.")
        else:
            self.backward_terms = default_terms

        self.backward_loss_weights = {name: 0.0 for name in allowed_terms}
        for term in self.backward_terms:
            self.backward_loss_weights[term] = 1.0

        if "score_matching_weight" in cfg:
            self.backward_loss_weights["score_matching"] = float(cfg.score_matching_weight)
        if "consistency_loss_weight" in cfg:
            self.backward_loss_weights["consistency_loss"] = float(cfg.consistency_loss_weight)
        if "consistency_loss_full_path_weight" in cfg:
            self.backward_loss_weights["consistency_loss_full_path"] = float(cfg.consistency_loss_full_path_weight)
        if "consistency_var_full_path_weight" in cfg:
            self.backward_loss_weights["consistency_var_full_path"] = float(cfg.consistency_var_full_path_weight)
        elif "consistency_log_variance_weight" in cfg:
            self.backward_loss_weights["consistency_var_full_path"] = float(cfg.consistency_log_variance_weight)

        if not self.backward_use_adjoint and all(w <= 0.0 for w in self.backward_loss_weights.values()):
            raise ValueError("At least one backward loss weight must be positive.")

        self.consistency_loss_weight = self.backward_loss_weights["consistency_loss"]
        self.consistency_full_path_weight = self.backward_loss_weights["consistency_loss_full_path"]
        self.consistency_var_full_path_weight = self.backward_loss_weights["consistency_var_full_path"]
        self.score_matching_weight = self.backward_loss_weights["score_matching"]
        self.consistency_log_variance_unbiased = bool(
            getattr(cfg, "consistency_log_variance_unbiased", False)
        )
        
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
    
    def inner_loop_u(self):
        current_task = "adjoint_matching"
        self.prev_integrator = copy_model(self.prev_integrator, self.integrator)
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

        # Populate the buffer with data from the controlled SDE integration
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            integral_results = self.prev_integrator.run_with_logweights(
                integrator=self.prev_integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            
            # filter 
            r = min(-(self.V0 / integral_results['log_Ph_Pref'].mean().item()), 1.0)   
            populate_data = {
                "data_0": data_0.detach().cpu(), 
                "data_1": integral_results['data_1'].detach().cpu(),
                "log_Ph_Pref": (r * integral_results['log_Ph_Pref']).detach().cpu(), # dP^h/dP^u
                "log_Pu_Pref": integral_results['log_Pu_Pref'].detach().cpu(),
            }
            
            if "log_Pref_Pu_det" in integral_results:
                populate_data["log_Pref_Pu_det"] = integral_results["log_Pref_Pu_det"].unsqueeze(-1).detach()
            
            terminal_cost_results = self.terminal_cost(
                data_1=integral_results['data_1'],
                is_init_stage=is_init_stage,
                return_energy=self.cfg.log_buffer_energy,
                return_grad_norm=self.cfg.log_buffer_grad_norm,
            )
            
            populate_data["terminal_cost"] = terminal_cost_results["terminal_cost"].detach().cpu()
            time_1 = self.fabric.to_device(torch.ones(self.cfg.inference_batch_size, dtype=torch.float))
            populate_data["control_1"] = self.prev_integrator.sde.u_controller(time_1, integral_results['data_1']).detach().cpu()
            
            if self.cfg.log_buffer_energy:
                populate_data["energy"] = terminal_cost_results["energy"].detach().cpu()
            if self.cfg.log_buffer_grad_norm:
                populate_data["grad_norm"] = terminal_cost_results["grad_norm"].detach().cpu()

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
        if len(buffer) == 0:
            num_samples = self.cfg.initial_buffer_samples
        else:
            num_samples = self.cfg.buffer_samples_per_epoch

        need_score = self.score_matching_weight > 0.0
        need_consistency_light = self.consistency_loss_weight > 0.0
        need_consistency_full = self.consistency_full_path_weight > 0.0
        need_consistency_var = self.consistency_var_full_path_weight > 0.0
        need_full_path = need_consistency_full or need_consistency_var
        need_transition = need_consistency_light and not need_full_path

        if need_transition or need_full_path:
            if not hasattr(self.integrator, "timesteps"):
                raise ValueError("integrator.timesteps is required for consistency_* backward losses")

        if need_transition:
            num_transitions = max(len(self.integrator.timesteps) - 1, 1)
            samples_per_cycle = self.cfg.inference_batch_size * num_transitions
            num_extend_cycles = math.ceil(num_samples / samples_per_cycle)
        else:
            num_extend_cycles = math.ceil(num_samples / self.cfg.inference_batch_size)
        
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
            if need_transition or need_full_path:
                trajectory = self.integrator.run(
                    initial_data=data_0,
                    center_every_step=self.potential.is_mean_free,
                    zero_last_step_noise=False,
                    return_trajectory=True,
                    progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
                )
                timesteps = self.integrator.timesteps.to(data_0.device)
                if need_transition:
                    for k in range(len(trajectory) - 1):
                        data_k = trajectory[k]
                        data_k1 = trajectory[k + 1]
                        t_k = timesteps[k]
                        t_k1 = timesteps[k + 1]
                        dt = t_k1 - t_k
                        sigma2_k = self.base_sde.diffsquare_integral(t_k1) - self.base_sde.diffsquare_integral(t_k)
                        bsz = data_k.shape[0]
                        populate_data = {
                            "data_k": data_k.detach().cpu(),
                            "data_k1": data_k1.detach().cpu(),
                            "time_k1": torch.full((bsz,), float(t_k1), dtype=torch.float),
                            "delta_t": torch.full((bsz,), float(dt), dtype=torch.float),
                            "sigma2_k": torch.full((bsz,), float(sigma2_k), dtype=torch.float),
                        }
                        if need_score:
                            populate_data["data_0"] = data_0.detach().cpu()
                            populate_data["data_1"] = trajectory[-1].detach().cpu()
                        buffer.extend(**populate_data)

                if need_full_path:
                    path_states = torch.stack(trajectory, dim=1)  # (B, K+1, ...)
                    batch_size = path_states.shape[0]
                    populate_data = {
                        "path_states": path_states.detach().cpu(),
                    }
                    if need_consistency_var:
                        data_dim = path_states[:, 0].reshape(batch_size, -1).shape[1]
                        log_pu_path = torch.zeros(batch_size, device=data_0.device)
                        for k in range(path_states.shape[1] - 1):
                            x_k = path_states[:, k]
                            x_k1 = path_states[:, k + 1]
                            t_k = timesteps[k]
                            t_k1 = timesteps[k + 1]
                            dt = t_k1 - t_k
                            sigma2_k = self.base_sde.diffsquare_integral(t_k1) - self.base_sde.diffsquare_integral(t_k)
                            sigma_k = sigma2_k.clamp(min=1e-20).sqrt()
                            if sigma_k.numel() == 1:
                                sigma_expand = torch.full((batch_size, data_dim), sigma_k.item(), device=data_0.device)
                            else:
                                sigma_expand = sigma_k
                            time_k = torch.full((batch_size,), float(t_k), device=data_0.device, dtype=torch.float)
                            drift_u = self.integrator.sde.drift(time_k, x_k)
                            mean_u = x_k + drift_u * dt
                            log_pu_step = gaussian_log_prob(
                                x_k1.reshape(batch_size, data_dim),
                                mean_u.reshape(batch_size, data_dim),
                                sigma_expand,
                                data_dim,
                            )
                            log_pu_path += log_pu_step

                        boundary_term = self.source.log_prob(path_states[:, 0]) - self.potential.log_prob(path_states[:, -1])
                        dt_steps = (timesteps[1:] - timesteps[:-1]).to(dtype=torch.float)
                        sigma2_steps = (
                            self.base_sde.diffsquare_integral(timesteps[1:])
                            - self.base_sde.diffsquare_integral(timesteps[:-1])
                        ).to(dtype=torch.float)
                        t_k1_steps = timesteps[1:].to(dtype=torch.float)
                        populate_data.update({
                            "log_pu_path": log_pu_path.detach().cpu(),
                            "boundary_term": boundary_term.detach().cpu(),
                            "delta_t_steps": dt_steps.unsqueeze(0).repeat(batch_size, 1).detach().cpu(),
                            "sigma2_steps": sigma2_steps.unsqueeze(0).repeat(batch_size, 1).detach().cpu(),
                            "time_k1_steps": t_k1_steps.unsqueeze(0).repeat(batch_size, 1).detach().cpu(),
                        })
                    buffer.extend(**populate_data)
            else:
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
                if self.backward_use_adjoint:
                    time_0 = self.fabric.to_device(torch.zeros(self.cfg.inference_batch_size, dtype=torch.float))
                    control_0 = self.controller(time_0, data_0)
                    with torch.enable_grad():
                        prior_score_0 = self.source.score(data_0)
                    sigma_0 = self.base_sde.diffusion(time_0)
                    sigma_0 = self._expand_scalar_to_data(sigma_0, prior_score_0)
                    sigma_prior_score_0 = sigma_0 * prior_score_0
                    populate_data["control_0"] = control_0.detach().cpu()
                    populate_data["sigma_prior_score_0"] = sigma_prior_score_0.detach().cpu()
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

    def _expand_scalar_to_data(self, scalar: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        while scalar.dim() < data.dim():
            scalar = scalar.unsqueeze(-1)
        return scalar

    def _compute_forward_adjoint_loss_components(self, batch, iw: torch.Tensor) -> dict[str, torch.Tensor]:
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]  # ∇E + h
        control_1 = batch["control_1"]
        batch_size = self.cfg.train_batch_size
        target = self.gamma * (terminal_cost + control_1) - control_1

        time_1 = self.ones_tensor
        if time_1.ndim == 0:
            time_1 = time_1.expand(batch_size)

        # t ~ U(1e-3, 1-1e-3)
        time = self.fabric.to_device(
            torch.rand(batch_size, dtype=torch.float)
        ) * (1 - 2e-3) + 1e-3

        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        # 3つを dim0 でまとめる
        time_all = torch.cat([time_1, time], dim=0)          # (2B,)
        data_all = torch.cat([data_1, data_t], dim=0)        # (2B, ...)

        # target, iw も 2回ぶんに拡張（target は data と同じ batch 次元を持つ想定）
        target_all = torch.cat([target, target], dim=0)      # (2B, ...)
        iw_all = torch.cat([iw, iw], dim=0)                      # (2B,)

        # forward は1回
        control_all = self.controller(time_all, data_all)            # (2B, ...)

        # 元の mse_with_iw を “2B版” にしてから、(2, B) に戻して各 loss を取る
        per_sample = (
            (control_all + target_all).pow(2)
            .reshape(2 * batch_size, -1)
            .mean(-1)                                                # (2B,)
            .mul(iw_all)                                             # (2B,)
        )                                                            # ここまではサンプル毎

        per_sample = per_sample.view(2, batch_size)
        loss_1 = per_sample[0].mean()
        loss_t = per_sample[1].mean()
        loss = loss_t + loss_1
        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_1": loss_1,
        }
        
    def adjoint_matching_step(self, batch) -> torch.Tensor:
        iw = self._compute_iw(batch, self.gamma)  # (B,) を想定
        adjoint_losses = self._compute_forward_adjoint_loss_components(batch, iw)
        total_loss = adjoint_losses["loss"]

        return {
            "loss": total_loss,
            "loss_t": adjoint_losses["loss_t"].detach().cpu(),
            "loss_1": adjoint_losses["loss_1"].detach().cpu(),
            "loss_adjoint": adjoint_losses["loss"].detach().cpu(),
        }

    def corrector_matching_step(self, batch) -> torch.Tensor:
        if self.backward_use_adjoint:
            return self.corrector_adjoint_matching_step(batch)
        loss = torch.zeros((), device=self.fabric.device)
        logs = {}

        if self.score_matching_weight > 0.0:
            score_losses = self.corrector_score_matching_step(batch)
            loss = loss + self.score_matching_weight * score_losses["loss"]
            logs["loss_score"] = score_losses["loss"].detach().cpu()

        if self.consistency_loss_weight > 0.0:
            if "data_k" in batch:
                consistency_losses = self.corrector_consistency_loss_step(batch)
            else:
                consistency_losses = self.corrector_consistency_loss_full_path_step(batch)
            loss = loss + self.consistency_loss_weight * consistency_losses["loss"]
            logs["loss_consistency"] = consistency_losses["loss"].detach().cpu()

        if self.consistency_full_path_weight > 0.0:
            consistency_full_losses = self.corrector_consistency_loss_full_path_step(batch)
            loss = loss + self.consistency_full_path_weight * consistency_full_losses["loss"]
            logs["loss_consistency_full_path"] = consistency_full_losses["loss"].detach().cpu()

        if self.consistency_var_full_path_weight > 0.0:
            lv_losses = self.corrector_consistency_log_variance_step(batch)
            loss = loss + self.consistency_var_full_path_weight * lv_losses["loss"]
            logs["loss_consistency_var_full_path"] = lv_losses["loss"].detach().cpu()

        logs["loss"] = loss
        return logs

    def corrector_score_matching_step(self, batch) -> torch.Tensor:
        """Score-matching loss for backward control."""
        if "data_0" in batch and "data_1" in batch:
            data_0 = batch["data_0"]
            data_1 = batch["data_1"]
        elif "path_states" in batch:
            path_states = self.fabric.to_device(batch["path_states"])
            data_0 = path_states[:, 0]
            data_1 = path_states[:, -1]
        else:
            raise ValueError("score_matching requires either (data_0, data_1) or path_states in batch.")

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

    def corrector_adjoint_matching_step(self, batch) -> torch.Tensor:
        """Adjoint-matching loss for backward control."""
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        control_0 = self.fabric.to_device(batch["control_0"])
        sigma_prior_score_0 = self.fabric.to_device(batch["sigma_prior_score_0"])

        batch_size = self.cfg.train_batch_size
        time = self.fabric.to_device(torch.rand(batch_size) * (1 - 2e-3) + 1e-3)
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        v_t = self.corrector(time, data_t)
        target = sigma_prior_score_0 - control_0
        loss_t = (
            (v_t - target)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)
            .mean()
        )
        return dict(loss=loss_t, loss_t=loss_t.detach().cpu())

    def corrector_consistency_loss_step(self, batch) -> torch.Tensor:
        """Discrete transition residual loss with Mahalanobis weighting."""
        data_k = batch["data_k"]
        data_k1 = batch["data_k1"]
        time_k1 = self.fabric.to_device(batch["time_k1"])
        delta_t = self.fabric.to_device(batch["delta_t"])
        sigma2_k = self.fabric.to_device(batch["sigma2_k"]).clamp(min=1e-20)

        v_k1 = self.corrector(time_k1, data_k1)
        delta_t_expand = self._expand_scalar_to_data(delta_t, v_k1)
        residual = data_k - data_k1 - v_k1 * delta_t_expand

        loss = (
            residual
            .reshape(self.cfg.train_batch_size, -1)
            .pow(2)
            .sum(dim=-1)
            .div(sigma2_k)
            .mean()
        )
        return dict(loss=loss, loss_t=loss.detach().cpu())

    def _compute_consistency_path_quantities(self, batch):
        path_states = self.fabric.to_device(batch["path_states"])
        delta_t_steps = self.fabric.to_device(batch["delta_t_steps"])
        sigma2_steps = self.fabric.to_device(batch["sigma2_steps"]).clamp(min=1e-20)
        time_k1_steps = self.fabric.to_device(batch["time_k1_steps"])

        batch_size = path_states.shape[0]
        steps = path_states.shape[1] - 1
        x_k = path_states[:, :-1]
        x_k1 = path_states[:, 1:]

        x_k1_flat = x_k1.reshape(batch_size * steps, *x_k1.shape[2:])
        time_flat = time_k1_steps.reshape(batch_size * steps).to(dtype=torch.float)
        v_flat = self.corrector(time_flat, x_k1_flat)
        v_steps = v_flat.reshape_as(x_k1)

        dt_expand = delta_t_steps.view(batch_size, steps, *([1] * (x_k.dim() - 2)))
        residual = x_k - x_k1 - v_steps * dt_expand

        sigma2_expand = sigma2_steps.view(batch_size, steps, *([1] * (x_k.dim() - 2)))
        sigma_expand = sigma2_expand.sqrt().expand_as(x_k)

        data_dim = x_k.reshape(batch_size * steps, -1).shape[1]
        mean_v = x_k1 + v_steps * dt_expand
        log_pv_step = gaussian_log_prob(
            x_k.reshape(batch_size * steps, data_dim),
            mean_v.reshape(batch_size * steps, data_dim),
            sigma_expand.reshape(batch_size * steps, data_dim),
            data_dim,
        ).reshape(batch_size, steps)
        log_pv_path = log_pv_step.sum(dim=1)

        consistency_loss = (
            residual
            .pow(2)
            .reshape(batch_size, steps, -1)
            .sum(dim=-1)
            .div(sigma2_steps)
            .mean()
        )
        return {
            "log_pv_path": log_pv_path,
            "consistency_loss": consistency_loss,
        }

    def corrector_consistency_loss_full_path_step(self, batch) -> torch.Tensor:
        """Memory-heavier consistency loss variant using full-path buffer entries."""
        q = self._compute_consistency_path_quantities(batch)
        consistency_loss = q["consistency_loss"]
        return dict(loss=consistency_loss, loss_t=consistency_loss.detach().cpu())

    def corrector_consistency_log_variance_step(self, batch) -> torch.Tensor:
        """Pathwise log-variance loss using consistency-style backward kernel."""
        q = self._compute_consistency_path_quantities(batch)
        log_pv_path = q["log_pv_path"]
        consistency_loss = q["consistency_loss"]
        log_pu_path = self.fabric.to_device(batch["log_pu_path"]).squeeze()
        boundary_term = self.fabric.to_device(batch["boundary_term"]).squeeze()

        log_weight_path = log_pv_path - log_pu_path + boundary_term
        lv_loss = log_weight_path.var(unbiased=self.consistency_log_variance_unbiased)

        return {
            "loss": lv_loss,
            "loss_lv": lv_loss.detach().cpu(),
            "loss_consistency": consistency_loss.detach().cpu(),
        }

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
        log_ph_pref_eval = integral_results['log_Ph_Pref'][:self.cfg.eval_samples]
        log_pu_pref_eval = integral_results['log_Pu_Pref'][:self.cfg.eval_samples]
        log_iw_eval = normalize_log_weights(
            self.gamma * log_ph_pref_eval + (1 - self.gamma) * log_pu_pref_eval
        )
        iw_eval = torch.exp(log_iw_eval).clamp(max=self.iw_clip_value)
        log_iw_no_gamma_eval = normalize_log_weights(log_ph_pref_eval)
        iw_no_gamma_eval = torch.exp(log_iw_no_gamma_eval).clamp(max=self.iw_clip_value)
        results = {
            "data_0": data_0[:self.cfg.eval_samples].detach().cpu(),
            "data_1": integral_results['data_1'][:self.cfg.eval_samples].detach().cpu(),
            "log_Ph_Pref": log_ph_pref_eval.detach().cpu(),
            "log_Pu_Pref": log_pu_pref_eval.detach().cpu(),
            'log_RN[u_next|u]_eff': integral_results['log_Pref_Pu_det'][:self.cfg.eval_samples].detach().cpu(),
            'log_RN[u_next|u]': -log_pu_pref_eval.detach().cpu(),
            "iw_debug": iw_eval.detach().cpu(),
            "log_iw_debug": log_iw_eval.detach().cpu(),
            "iw_debug_no_gamma": iw_no_gamma_eval.detach().cpu(),
            "log_iw_debug_no_gamma": log_iw_no_gamma_eval.detach().cpu(),
            "iw_debug_gamma_value": float(self.gamma),
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
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        inference_batch_size = self.cfg.inference_batch_size
        num_iters = math.ceil(total_samples / inference_batch_size)
        
        all_log_weights_np = []

        # プログレスバー（必要に応じて）
        iterator = range(num_iters)
        if self.fabric.global_rank == 0 and self.cfg.progress_bar:
             iterator = tqdm(iterator, desc="Collecting weights", leave=False)

        for _ in iterator:
            # 念のためループごとにメモリ掃除
            torch.cuda.empty_cache()
            # データの準備
            data_0 = self.source.sample((inference_batch_size,))
            
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
            # shape: (batch_size,)
            log_Pu_Ph = -integral_results["log_Ph_Pref"]
            
            # 【重要】ここで即座に CPU NumPy 配列に変換してリストに追加
            # これにより GPU メモリと PyTorch の計算グラフ用メモリを即解放
            all_log_weights_np.append(log_Pu_Ph.detach().cpu().numpy().astype(np.float64))
            
            # 明示的にキャッシュクリア（念のため）
            del integral_results, log_Pu_Ph, data_0
        
        self.integrator.train()
        
        # 最後に結合して、指定サンプル数に切り詰める
        if len(all_log_weights_np) > 0:
            return np.concatenate(all_log_weights_np, axis=0)[:total_samples]
        else:
            return np.array([])

    def optimize_gamma(self, max_lm=1e6) -> float:
        # 1. データを収集 (NumPy配列)
        log_weights_np = self.collect_log_weights_for_optimization()
        
        if log_weights_np.size == 0:
            rank_zero_warn("No samples collected for gamma optimization. Skipping.")
            return self.gamma, 0.0

        # scipy.special.logsumexp を使用
        from scipy.special import logsumexp

        # 2. 高速計算用ローカル関数 (純粋な NumPy 計算)
        def calc_V(lm):
            alpha = -1.0 / (1.0 + lm)
            term = logsumexp(alpha * log_weights_np, axis=0) - math.log(log_weights_np.shape[0])
            V = (1.0 + lm) * (-term)
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
        os.environ["WANDB_MODE"] = str(cfg.wandb_mode)
        os.environ["WANDB_BASE_URL"] = str(cfg.wandb_host)
    torch.set_float32_matmul_precision("high")
    trainer = GeneralASBSModule(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
