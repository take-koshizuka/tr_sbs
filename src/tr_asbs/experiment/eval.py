# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

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
import json
import hashlib
from omegaconf import OmegaConf
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from tr_asbs.potential.base_potential import SumPotential
from tr_asbs.potential.metadynamics import WellTemperedMetadynamicsBias
from tr_asbs.process.sde import ControlledSDE
from tr_asbs.utils.topology import save_data_to_pdb
from tr_asbs.utils.training import EMA, TrainingCurriculum
from tr_asbs.utils.md_utils import plot_marginal_group_AD, plot_energy_hist, plot_ramachandran_AD

import time

def stable_group_id(cfg: dict, exclude=("seed", "name")):
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    d = {k: cfg[k] for k in sorted(cfg.keys()) if k not in exclude}
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]

class ASBSModule:
    """Module for ASBS training."""

    TASKS: list[str] = ["adjoint_matching", "corrector_matching"]

    ############################################################################
    # Initialization
    ############################################################################

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        cfg.fabric.loggers.group = stable_group_id(cfg)
        # Lightning fabric setup
        self.fabric = hydra.utils.instantiate(cfg.fabric)
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
            self.sde = ControlledSDE(base_sde=self.base_sde, controller=self.controller)
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            # Potential and cost function
            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential, corrector=self.corrector
            )

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
        
        while self.current_epoch < len(self.curriculum):
            if self.current_epoch % self.cfg.log_detail_interval_epoch == 0:
                self.log_details(self.current_epoch // self.cfg.log_detail_interval_epoch)
            
            current_task = self.curriculum[self.current_epoch]
            
            self.populate_buffer(current_task)
            dataloader = self.create_dataloader(current_task)
            for batch in tqdm(
                dataloader,
                desc=f"Epoch {self.current_epoch + 1}/{len(self.curriculum)}",
                leave=False,
                disable=self.fabric.global_rank != 0 or not self.cfg.progress_bar,
            ):
                self.training_step(batch, current_task)
                self.global_step += 1
            
            self.current_epoch += 1
            
            if self.current_epoch % self.cfg.save_checkpoint_interval_epoch == 0:
                self.save_checkpoint()
        
        final_logs = self.log_details(self.max_outer_loops + 1)
        
        # rank0だけ summary に確定値を書き込む
        if self.fabric.global_rank == 0 and wandb.run is not None:
            for key, value in final_logs.items():
                wandb.run.summary[f"final/{key}"] = value

            # 併せて便利なメタ情報も
            wandb.run.summary["final/epoch"] = int(self.current_epoch)
            wandb.run.summary["final/global_step"] = int(self.global_step)
            # dimension/seed も入れると集計が楽
            wandb.run.summary["seed"] = int(self.cfg.seed)
            # wandb.run.summary["target_dim"] = int(self.cfg.target_dim)
        
            wandb.finish()
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
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.potential.is_mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            populate_data = {"data_0": data_0, "data_1": data_1}
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
            if hasattr(self.cfg, "topology_pdb_file"):  # Save as PDB instead
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.pdb"
                save_data_to_pdb(
                    data=data,
                    topology_file=self.cfg.topology_pdb_file,
                    output_file=str(filename),
                )
            elif isinstance(data, torch.Tensor):  # Save as XYZ if data is tensor
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.npy"
                np.save(filename, data.cpu().numpy())
            else:  # default to XYZ format
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.xyz"
                ase.io.write(str(filename), data.to_ase())

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

        # Perform the optimizer step
        optimizer.zero_grad()
        self.fabric.backward(loss)
        self.fabric.clip_gradients(
            model, optimizer, max_norm=self.cfg.gradient_clip_val
        )
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

    def corrector_matching_step(self, batch) -> torch.Tensor:
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
        checkpoint_directory = Path(self.cfg.checkpoint_directory)
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
        checkpoint_directory = Path(self.cfg.checkpoint_directory)
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
        
    def log_details(self, step):
        data_0 = self.source.sample((self.cfg.eval_samples,))
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
        }
        
        if self.current_epoch == 0:
            logs = self.potential.log_details(results, visualize=self.cfg.visualize)
            if self.cfg.visualize:
                logs["figures/gt_data_0"] = self.potential.visualize_raw(data_0, title="gt_data_0")
                logs["figures/gt_data_1"] = self.potential.visualize(self.potential.sample(data_0.shape[0:1]), title="gt_data_1")
        
        elif step == (self.max_outer_loops + 1):
            logs = self.potential.log_details(results, visualize=True)
        else:
            logs = self.potential.log_details(results, visualize=self.cfg.visualize)
        
        logs["epoch"] = self.current_epoch
        logs["eval_step"] = step
        self.fabric.log_dict(logs, step=self.global_step)
        return logs

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Potential (AlanineDipeptideEnergyIC) のインスタンス化
    # cfg.potential に AlanineDipeptideEnergyIC の設定が入っていることを想定
    print("Instantiating AlanineDipeptideEnergyIC...")
    potential = hydra.utils.instantiate(cfg.potential, device=device)

    # 3. Ground-Truth から 2000 サンプルサンプリング
    # sampleメソッドは return_tensor=True (デフォルト) で 
    # [N, 66] (Cartesian) または [N, dim] (Internal) のテンソルを返す
    num_samples = 2000
    print(f"Sampling {num_samples} samples from ground-truth...")
    
    # 4. visualize 関数の実行
    # 内部で plot_energy_hist と plot_ramachandran_AD を呼び出し、
    # wandb.Image や Figure を含む辞書を返します。
    #print("Running visualize (Energy histogram & Ramachandran plot)...")
    #vis_logs = potential.visualize(gt_samples_x, title="GT_Samples_Visualization")

    # 5. plot_marginal_group_AD の実行 (log_details を利用)
    # log_details は内部で x への変換と plot_marginal_group_AD を行います。
    print("Running log_details (Marginal distributions)...")
    # log_details は results["data_1"] を期待しているため、辞書を作成
    gt_samples_x = potential.sample(num_samples, return_tensor=True, coordinate="z")
    results = {"data_1": gt_samples_x}
    logs = potential.log_details(results, visualize=True)
    gt_samples_x = potential.sample(num_samples, return_tensor=True, coordinate="z")
    results = {"data_1": gt_samples_x}
    logs1 = potential.log_details(results, visualize=False)
    gt_samples_x = potential.sample(num_samples, return_tensor=True, coordinate="z")
    results = {"data_1": gt_samples_x}
    logs2 = potential.log_details(results, visualize=False)
    
    keys = [
        'loss/kld_phi',
        'loss/kld_gamma1',
        'loss/kld_psi',
        'loss/kld_gamma2',
        'loss/kld_gamma3',
        'loss/w2_phi_psi'
    ]
    for key in keys:
        val = [log[key] for log in [logs, logs1, logs2]]
        print(f"{key}: mean={np.mean(val):.4f}, std={np.std(val):.4f}")
    
    output_dir = Path("aldp_visualizations")
    output_dir.mkdir(exist_ok=True)

    #logs = plot_marginal_group_AD(gt_samples_x.cpu(), ref_samples)
    logs["figures/marginal"].image.save(output_dir / "marginal_distributions.png")
    # logs = plot_marginal_group_AD(gt_samples_z.cpu(), ref_samples)
    # logs["figures/marginal"][0].image.save(output_dir / "marginal_distributions_z.png")
    #gen_energies = potential.eval(gt_samples_z).detach().cpu()
    #ref_energies = potential.eval(ref_samples_z).detach().cpu()
    
    #logs = plot_energy_hist(gen_energies, ref_energies)
    logs["figures/energy_hist"].image.save(output_dir / "energy_histogram.png")
    
    #logs = plot_ramachandran_AD(ref_samples, potential.n_particles, potential.spatial_dim)
    logs["figures/ramachandran"].image.save(output_dir / "gt_ramachandran.png")
    
    print(f"\nAll visualizations are saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()