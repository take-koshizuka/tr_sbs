# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import inspect

import ase.io
import hydra
import torch

import os
import numpy as np
import wandb
from lightning.fabric import seed_everything
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import json
import hashlib
from omegaconf import OmegaConf
import uuid
from copy import deepcopy

from tr_asbs.process.sde import ControlledSDE, ControlledFBSDE
from tr_asbs.utils.training import EMA, TrainingCurriculum, copy_model
from tr_asbs.utils.helper import Checkpointer, CheckpointConfig

import time

def stable_group_id(cfg: dict, exclude=("seed", "name")):
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    d = {k: cfg[k] for k in sorted(cfg.keys()) if k not in exclude}
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def instantiate_logger_compat(logger_cfg: DictConfig):
    """Instantiate logger while dropping merged keys unsupported by target logger."""
    target = str(getattr(logger_cfg, "_target_", ""))
    logger_cls = hydra.utils.get_class(target)

    cfg_dict = OmegaConf.to_container(logger_cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"logger config must be a mapping, got {type(cfg_dict)}")

    cfg_dict.pop("_target_", None)
    cfg_dict.pop("_delete_", None)

    sig = inspect.signature(logger_cls.__init__)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return logger_cls(**cfg_dict)

    valid_keys = {name for name in sig.parameters if name != "self"}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return logger_cls(**filtered)

class ASBSModule:
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

        wandb_logger = instantiate_logger_compat(cfg.fabric.loggers)
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(full_cfg)
        
        if ("corrector_matching_mode" in cfg) and (cfg.corrector_matching_mode == "full"):
            self.use_full_corrector_matching = True
            self.corrector_matching_step = self.corrector_full_matching_step
        else:
            self.use_full_corrector_matching = False
            self.corrector_matching_step = self.corrector_last_matching_step
        
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
            
            if self.use_full_corrector_matching:
                self.sde = ControlledFBSDE(
                    base_sde=self.base_sde,
                    u_controller=self.controller, 
                    h_controller=self.corrector
                )
            else:
                self.sde = ControlledSDE(base_sde=self.base_sde, controller=self.controller)
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            # Potential and cost function
            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential, corrector=self.corrector
            )
            
            self.prev_integrator = deepcopy(self.integrator)

        self.max_outer_loops = cfg.max_outer_loops
        curriculum = []
        for _ in range(cfg.max_outer_loops):
            loop_curriculum = [
                {"adjoint_matching": cfg.adjoint_matching_epochs},
                {"corrector_matching": cfg.corrector_matching_epochs},
            ]
            curriculum.extend(loop_curriculum)
            
        self.curriculum = TrainingCurriculum(
            curriculum=curriculum, valid_tasks=self.TASKS
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
        
        if hasattr(cfg, "controller_scheduler"):
            self.controller_scheduler = hydra.utils.instantiate(
                cfg.controller_scheduler,
                curriculum=self.curriculum,
                optimizer=self.controller_optimizer,
            )
        if hasattr(cfg, "corrector_scheduler"):
            self.corrector_scheduler = hydra.utils.instantiate(
                cfg.corrector_scheduler,
                curriculum=self.curriculum,
                optimizer=self.corrector_optimizer,
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
        
        print(self.checkpoint_directory)
        if wandb.run is not None:
            wandb.run.summary["group_id"] = self.group_id
            wandb.run.summary["checkpoint_directory"] = str(self.checkpoint_directory)
            
    
    def remove_grad_nan(self, model):
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()
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
        
        prev_task = None
        while self.current_epoch < len(self.curriculum):
            if self.current_epoch % self.cfg.log_detail_interval_epoch == 0:
                metrics = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch)
                for name in self.cfg.monitor_values:
                    name_safe = name.replace("/", "_")
                    info = self.ckpts[name_safe].update(
                        value=metrics[name],
                        step=self.current_epoch // self.cfg.log_detail_interval_epoch,
                        tag_best=f"{name_safe}_best",          # ファイル名 "{tag}.pt" なので "best.pt" になる
                        extra_status={"eval_step": metrics["eval_step"]},
                    )
            
            current_task = self.curriculum[self.current_epoch]
            
            if current_task != prev_task and current_task == "adjoint_matching":
                self.prev_integrator = copy_model(self.prev_integrator, self.integrator)
                
            self.populate_buffer(current_task)
            dataloader = self.create_dataloader(current_task)
            for batch in tqdm(
                dataloader,
                desc=f"Epoch {self.current_epoch + 1}/{len(self.curriculum)}",
                leave=False,
                disable=self.fabric.global_rank != 0 or not self.cfg.progress_bar,
            ):
                if self.training_step(batch, current_task):
                    self.global_step += 1
            
            self.current_epoch += 1
                
            prev_task = current_task
        
        self.log_final()
        self.save_checkpoint()
        
    ############################################################################
    # Buffer and dataloader management
    ############################################################################

    @torch.no_grad()
    def populate_buffer(self, current_task):
        """Populate the buffer with data from the controlled SDE integration and log the
        buffer size if the next task is adjoint matching."""
        if current_task.name == "adjoint_matching":
            buffer = self.adjoint_buffer
        elif current_task.name == "corrector_matching":
            buffer = self.corrector_buffer

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
        controller_training_state = self.controller.training
        corrector_training_state = self.corrector.training
        self.controller.eval()
        self.corrector.eval()

        # Populate the buffer with data from the controlled SDE integration
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            # data_0 = self.annealer(data_0, self.current_epoch) # 違い
            
            if self.use_full_corrector_matching:
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
                data_1 = integral_results['data_1']
                populate_data = {
                    "data_0": data_0, 
                    "data_1": integral_results['data_1'],
                    "log_Ph_Pref": integral_results['log_Ph_Pref'],
                    "log_Pu_Pref": integral_results['log_Pu_Pref'],
                }
            else:
                data_1 = self.integrator.run(
                    initial_data=data_0,
                    center_every_step=self.potential.is_mean_free,
                    zero_last_step_noise=False,
                    return_trajectory=False,
                    progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
                )
                populate_data = {"data_0": data_0, "data_1": data_1}
            
            if hasattr(self.potential, "filter_samples"):
                ind = self.potential.filter_samples(data_1)
                data_0 = data_0[ind]
                data_1 = data_1[ind]

            # Add terminal cost if the next task is adjoint matching
            if current_task.name == "adjoint_matching":
                results = self.terminal_cost(
                    data_1=data_1,
                    is_init_stage=current_task.is_init_stage,
                    return_energy=self.cfg.log_buffer_energy,
                    return_grad_norm=self.cfg.log_buffer_grad_norm,
                )
                populate_data["terminal_cost"] = results["terminal_cost"]
                if self.cfg.log_buffer_energy:
                    populate_data["energy"] = results["energy"]
                if self.cfg.log_buffer_grad_norm:
                    populate_data["grad_norm"] = results["grad_norm"]
                
            buffer.extend(**populate_data)

        # Log the buffer statistics
        if current_task.name == "adjoint_matching":
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
                f"train/{current_task.name}_buffer_size", len(buffer), step=self.global_step
            )
            # Restore the training state of the models
            self.fabric.barrier()
        
        self.controller.train(controller_training_state)
        self.corrector.train(corrector_training_state)

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
        if current_task.name == "adjoint_matching":
            dataset = self.adjoint_buffer
        elif current_task.name == "corrector_matching":
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
        if current_task.name == "adjoint_matching":
            loss = self.adjoint_matching_step(batch)
            model = self.controller
            optimizer = self.controller_optimizer
            scheduler_step = hasattr(self, "controller_scheduler")
            if scheduler_step:
                scheduler = self.controller_scheduler
            
        elif current_task.name == "corrector_matching":
            loss = self.corrector_matching_step(batch)
            model = self.corrector
            optimizer = self.corrector_optimizer
            scheduler_step = hasattr(self, "corrector_scheduler")
            if scheduler_step:
                scheduler = self.corrector_scheduler
        
        if not torch.isfinite(loss):
            return False

        # Perform the optimizer step
        optimizer.zero_grad()
        self.fabric.backward(loss)
        
        try:
            self.fabric.clip_gradients(model, optimizer, max_norm=self.cfg.gradient_clip_val)
        except RuntimeError as e:
            if "non-finite" in str(e):
                optimizer.zero_grad(set_to_none=True)
                return False
            raise
    
        optimizer.step()
        # Update the EMA
        if self.cfg.ema_decay is not None:
            model.update_ema(self.global_step)
            
        if scheduler_step:
            if (
                not hasattr(self.cfg, "skip_scheduler_steps")
                or self.global_step >= self.cfg.skip_scheduler_steps
            ):
                scheduler.step()

        # Log the lr and loss for the current task
        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                f"train/{current_task.name}_lr",
                optimizer.param_groups[0]["lr"],
                step=self.global_step,
            )
            loss = self.fabric.all_reduce(loss.detach(), reduce_op="mean")
            self.fabric.log(
                f"train/{current_task.name}_loss", loss, step=self.global_step
            )
        return True

    def adjoint_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the adjoint matching and return the loss.
        Equation 14 in the ASBS paper."""
        # Unpack the batch: X_0, X_1 ~ p_{0,1}^u and terminal cost (∇E + h^(k-1))(X_1)
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]

        # Sample data at random time: t ~ U[0, 1], X_t ~ p_{t|0,1}^base
        time = self.fabric.to_device(torch.rand(self.cfg.train_batch_size, dtype=torch.float))
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        # Predict control at time t: u_t(X_t)/σ_t
        control = self.controller(time, data_t)

        # Compute the MSE loss, averaged over all atoms and dimensions
        # NOTE: This is a scaled version: ||u_t(X_t)/σ_t + (∇E + h^(k-1))||^2
        loss = (control + terminal_cost).pow(2).mean()
        return loss

    def corrector_last_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the corrector matching and return the loss.
        Equation 15 in the ASBS paper."""
        # Unpack the batch: X_0, X_1 ~ p_{0,1}^{u^(k)}
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]

        # Predict correction at time t=1: h(X_1)
        time_1 = self.fabric.to_device(torch.ones(self.cfg.train_batch_size, dtype=torch.float))
        correction = self.corrector(time_1, data_1)

        # Compute the target score: ∇_{X_1} log p^base(X_1 | X_0)
        score = self.base_sde.cond_score_t0(time=time_1, data_0=data_0, data_t=data_1)

        # Compute the MSE loss, averaged over all atoms and dimensions
        loss = (correction - score).pow(2).mean()
        return loss
    
    def corrector_full_matching_step(self, batch) -> torch.Tensor:
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
        return loss

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

    def save_checkpoint(self):
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
        
    def log_details(self, step, visualize=False, eval_samples=None):
        visualize = visualize or self.cfg.visualize
        eval_samples = eval_samples if eval_samples is not None else self.cfg.eval_samples
        data_0 = self.source.sample((eval_samples,))
        
        if self.use_full_corrector_matching:
            integral_results = self.integrator.run_with_logweights(
                integrator=self.integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=True,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            data_1 = integral_results['data_1']
            
            results = {
                "data_0": data_0,
                "data_1": data_1,
                "log_Ph_Pref": integral_results['log_Ph_Pref'],
                "log_Pu_Pref": integral_results['log_Pu_Pref'],
                'log_RN[u_next|u]_eff': -integral_results['log_Pref_Pu_det'],
                'log_RN[u_next|u]': -integral_results['log_Pu_Pref'],
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
                    log_weights_smooth = batch["log_Ph_Pref"]
                    all_log_weights.append(log_weights)
                    all_log_weights_smooth.append(log_weights_smooth)
                # dim=0 で concatenate
                results['log_RN[h|u]'] = torch.cat(all_log_weights, dim=0)
                results['log_RN[u_next|u]_pred_buffer'] = torch.cat(all_log_weights_smooth, dim=0)
        else:
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            results = {
                "data_0": data_0,
                "data_1": data_1,
                'predict_drift_fn': self.sde.drift,
            }
        
        if self.current_epoch == 0:
            logs = self.potential.log_details(results, visualize=visualize)
            if self.cfg.visualize:
                logs["figures/gt_data_0"] = self.potential.visualize_raw(data_0, title="gt_data_0")
                logs["figures/gt_data_1"] = self.potential.visualize(self.potential.sample(data_0.shape[0:1]), title="gt_data_1")
        else:
            logs = self.potential.log_details(results, visualize=visualize)
        
        logs["epoch"] = self.current_epoch
        logs["eval_step"] = step
        if eval_samples == self.cfg.eval_samples:
            self.fabric.log_dict(logs, step=self.global_step)
        logs.update({"data_0": data_0, "data_1": data_1})
        return logs
    
    def log_final(self):
        final_logs = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch, eval_samples=self.cfg.eval_samples_final)
        data_0, data_1 = final_logs.pop("data_0"), final_logs.pop("data_1")
        self.save_final_samples(data_0, data_1, name=f"last_epoch")
        
        # rank0だけ summary に確定値を書き込む
        if self.fabric.global_rank == 0 and wandb.run is not None:
            wandb.run.summary["seed"] = int(self.cfg.seed)
            for key, value in final_logs.items():
                wandb.run.summary[f"final/{key}_final"] = value
            wandb.run.summary["final/epoch"] = int(self.current_epoch)
            wandb.run.summary["final/global_step"] = int(self.global_step)
            
        final_logs = self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch, visualize=True, eval_samples=self.cfg.eval_samples)
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
            data_0 = self.source.sample((self.cfg.eval_samples_final,))
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
                "data_1": data_1[:self.cfg.eval_samples],
                'predict_drift_fn': self.sde.drift,
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

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
    use_wandb_logger = "wandb" in logger_target
    if use_wandb_logger:
        os.environ["WANDB_MODE"] = str(cfg.wandb_mode)
        os.environ["WANDB_BASE_URL"] = str(cfg.wandb_host)
        os.environ["WANDB_API_KEY"] = str(cfg.wandb_key)
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    # NOTE: changing matmul precision may result in mean drift
    torch.set_float32_matmul_precision("high")
    trainer = ASBSModule(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
