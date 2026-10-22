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
from tr_asbs.utils.eval_logweight import compute_raw_path_ess, compute_reverse_ess, normalize_log_weights
from tr_asbs.utils.helper import Checkpointer, CheckpointConfig, JointMetricDivergenceStopper

import time

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
        self.cfg = cfg
        random_hash = uuid.uuid4().hex[:8]
        self.group_id = str(stable_group_id(cfg, exclude=("name"))) + f"{random_hash}"
        logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
        if "wandb" in logger_target and "group" in cfg.fabric.loggers:
            cfg.fabric.loggers.group = stable_group_id(cfg)
        
        wandb_logger = hydra.utils.instantiate(cfg.fabric.loggers)
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(full_cfg)

        # step size
        self.gamma = cfg.gamma
        assert 0 <= self.gamma <= 1
        self.gamma_schedule = cfg.gamma_schedule
        assert self.gamma_schedule in ["fixed", "trust_region"]
        self.trust_region_bound = cfg.trust_region_bound

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
        self.divergence_stopper = JointMetricDivergenceStopper(self.cfg.divergence_early_stop)
        self.early_stop_requested = False
        
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
                },
            )
            
        if wandb.run is not None:
            wandb.run.summary["group_id"] = self.group_id
            wandb.run.summary["checkpoint_directory"] = str(self.checkpoint_directory)
        
    @property
    def lm(self) -> float:
        """lagrangian multiplier expressed from gamma. Kept for compatibility."""
        return (1 - self.gamma) / self.gamma

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
                        extra_status={"eval_step": metrics["eval_step"], "epoch": metrics["epoch"]},
                    )
                self._update_divergence_early_stop(metrics)
                if self.early_stop_requested:
                    break
            
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
                        extra_status={"eval_step": metrics["eval_step"], "epoch": metrics["epoch"]},
                    )
                self._update_divergence_early_stop(metrics)
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

        for outer_step in range(self.max_outer_loops):
            if self.gamma_schedule == "trust_region":
                print("Optimizing gamma via trust region...")
                self.gamma = self.optimize_gamma()
            
            print("Starting inner loop for controller u...")
            self.inner_loop_u()
            if self.early_stop_requested:
                break
            
            print("Starting inner loop for corrector h...")
            self.inner_loop_h()
            if self.early_stop_requested:
                break

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
            integral_results = self.integrator.run_with_logweights(
                integrator=self.integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            populate_data = {
                "data_0": data_0, 
                "data_1": integral_results['data_1'],
                "log_Ph_Pref": integral_results['log_Ph_Pref'],
                "log_Pu_Pref": integral_results['log_Pu_Pref'],
            }
            
            if "log_Pref_Pu_det" in integral_results:
                populate_data["log_Pref_Pu_det"] = integral_results["log_Pref_Pu_det"].unsqueeze(-1).detach()
            
            terminal_cost_results = self.terminal_cost(
                data_1=integral_results['data_1'],
                is_init_stage=is_init_stage,
                return_energy=self.cfg.log_buffer_energy,
                return_grad_norm=self.cfg.log_buffer_grad_norm,
            )
            
            populate_data["terminal_cost"] = terminal_cost_results["terminal_cost"]
            time_1 = self.fabric.to_device(torch.ones(self.cfg.inference_batch_size, dtype=torch.float))
            populate_data["control_1"] = self.prev_integrator.sde.u_controller(time_1, integral_results['data_1']).detach()
            
            if self.cfg.log_buffer_energy:
                populate_data["energy"] = terminal_cost_results["energy"]
            if self.cfg.log_buffer_grad_norm:
                populate_data["grad_norm"] = terminal_cost_results["grad_norm"]

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
                "data_0": data_0,
                "data_1": data_1,
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
            buffer.extend(**{"log_weights": log_Pu_Ph})
        
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
            self.fabric.clip_gradients(
                model, optimizer, max_norm=self.cfg.gradient_clip_val
            )
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
            
    def adjoint_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the adjoint matching and return the loss.
        Equation 14 in the ASBS paper.
        """
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]  # ∇E + h
        control_1 = batch["control_1"]

        target = self.gamma * (terminal_cost + control_1) - control_1

        batch_size = self.cfg.train_batch_size
        device = self.fabric

        time_0 = self.zeros_tensor
        time_1 = self.ones_tensor
        
        def mse(pred):
            return (
                pred.pow(2)
                .reshape(batch_size, -1)
                .mean(-1)
                .mean()
            )
        

        loss_0 = mse(self.controller(time_0, data_0) + target)
        loss_1 = mse(self.controller(time_1, data_1) + target)

        time = self.fabric.to_device(torch.rand(self.cfg.train_batch_size, dtype=torch.float)) * (1 - 2e-3) + 1e-3
        data_t = self.base_sde.sample_posterior(
            time=time, data_0=data_0, data_1=data_1
        )

        control = self.controller(time, data_t)
        loss_t = mse(control + target)

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
        device = self.fabric

        time = device.to_device(
            torch.rand(batch_size) * (1 - 2e-3) + 1e-3
        )
        data_t = self.base_sde.sample_posterior(
            time=time, data_0=data_0, data_1=data_1
        )

        h_t = self.corrector(time, data_t)
        score_t = self.base_sde.cond_score_t0(
            time=time, data_0=data_0, data_t=data_t
        )

        coef_t = (
            1
            + self.base_sde.diffusion(time).pow(2)
            * (1 - time) / time
        )

        loss_t = (
            (h_t - score_t)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)
            .div(coef_t.squeeze())
            .mean()
        )

        time_1 = self.ones_tensor
        h_1 = self.corrector(time_1, data_1)

        score_1 = self.base_sde.cond_score_t0(
            time=time_1, data_0=data_0, data_t=data_1
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
        
    def log_details(self, step: int, visualize: bool = False, eval_samples:int = None) -> dict:
        visualize = visualize or self.cfg.visualize
        eval_samples = eval_samples if eval_samples is not None else self.cfg.eval_samples
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
            "data_0": data_0[:eval_samples],
            "data_1": integral_results['data_1'][:eval_samples],
            "log_Ph_Pref": integral_results['log_Ph_Pref'][:eval_samples],
            "log_Pu_Pref": integral_results['log_Pu_Pref'][:eval_samples],
            'log_RN[u_next|u]_eff': integral_results['log_Pref_Pu_det'][:eval_samples],
            'log_RN[u_next|u]': -integral_results['log_Pu_Pref'][:eval_samples],
            'predict_drift_fn': self.sde.drift,
        }
        
        if self.current_epoch > 0:
            dataloader = DataLoader(
                self.adjoint_buffer,
                batch_size=self.cfg.train_batch_size,
                collate_fn=self.adjoint_buffer.collate_fn,
                num_workers=self.cfg.num_workers,
                pin_memory=False,
                persistent_workers=False,
            )
            
            all_log_weights = []
            all_log_weights_smooth = []
            for batch in dataloader:
                log_weights = batch["log_Ph_Pref"]
                log_weights_smooth = self.gamma * batch["log_Ph_Pref"] + (1 - self.gamma) * batch["log_Pu_Pref"]
                all_log_weights.append(log_weights)
                all_log_weights_smooth.append(log_weights_smooth)
            # dim=0 で concatenate
            results['log_RN[h|u]'] = torch.cat(all_log_weights, dim=0)
            results['log_RN[u_next|u]_pred_buffer'] = torch.cat(all_log_weights_smooth, dim=0)
            results['lm'] = self.lm
            results['gamma'] = self.gamma
        
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
        logs["eval/raw_path_ess"] = compute_raw_path_ess(results["log_Ph_Pref"])
        
        logs["epoch"] = self.current_epoch
        logs["eval_step"] = step

        self.fabric.log_dict(logs, step=self.global_step)
        
        logs.update({"data_0": data_0, "data_1": integral_results['data_1']})
        return logs
    
    def log_final(self):
        final_logs = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch, visualize=True, eval_samples=self.cfg.eval_samples_final)
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
            self.save_final_samples(data_0, data_1, name=f"best_{name}")
            results = {
                "data_0": data_0,
                "data_1": data_1,
                "log_Ph_Pref": integral_results["log_Ph_Pref"],
            }
            
            logs = self.potential.log_details(results, visualize=True)
            logs["eval/raw_path_ess"] = compute_raw_path_ess(results["log_Ph_Pref"])
            
            if self.fabric.global_rank == 0 and wandb.run is not None:
                for key, value in logs.items():
                    wandb.run.summary[f"best_{name}/{key}"] = value
                wandb.run.summary[f"best_{name}/step"] = int(step)
                wandb.run.summary[f"best_{name}/epoch"] = int(info["status"].get("epoch", step * self.cfg.log_detail_interval_epoch))
                wandb.run.summary[f"best_{name}/value"] = float(best_val)
            
            # ===== checkpoint 削除 =====
            if self.fabric.global_rank == 0 and not self.cfg.keep_best_checkpoints:
                ckpt_path.unlink(missing_ok=True)
                
        wandb.finish()
        
    @torch.no_grad()
    def dual_loss(self, lm, dataloader):
        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        mean_log_weights_sum = torch.zeros((min(self.cfg.inference_batch_size, self.cfg.num_samples_for_dual_loss[0]), ))
        for j, batch in enumerate(dataloader):
            # the expectation wrt paths using self.batch_size_path samples
            mean_log_weights = torch.logsumexp((- 1 / (1 + lm)) * batch["log_weights"], dim=1) - math.log(batch_size_path)
            mean_log_weights_sum += mean_log_weights
        # the expectation wrt x_0 using self.batch_size samples
        V = (1 + lm) * (-mean_log_weights_sum / (j+1)).mean(0)
        return V.item()

    def optimize_gamma(self, max_lm=1e6) -> float:
        log_weights_buffer = BatchBuffer(max_size=self.cfg.num_samples_for_dual_loss[0])
        self.populate_buffer_log_weights(log_weights_buffer)
        dataloader = DataLoader(
            log_weights_buffer,
            batch_size=min(self.cfg.inference_batch_size, self.cfg.num_samples_for_dual_loss[0]),
            collate_fn=log_weights_buffer.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        neg_dual = lambda lm: -self.dual_loss(lm, dataloader) + lm * self.trust_region_bound
        res = minimize_scalar(neg_dual, bounds=(0, max_lm), method="Bounded")
        best_lambda: float = res.x
        best_gamma = 1 / (1 + best_lambda)
        del log_weights_buffer
        return best_gamma

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
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
