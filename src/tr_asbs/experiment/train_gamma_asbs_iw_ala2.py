# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gamma/TR-SBS trainer for the Ala2 (graph-batch / ThermoAtomicData) pipeline.

Subclasses GammaASBSModule (train_gamma_asbs_iw.py) rather than duplicating
it wholesale: the gamma dual-solve itself (optimize_gamma,
collect_log_weights_for_optimization's log-weight bookkeeping,
compute_V0_for_fixed_gamma) and the outer training-loop orchestration
(fit/inner_loop_u/inner_loop_h/training_step/_compute_iw/checkpointing/
create_dataloader/save_samples) already operate on data type-agnostic
quantities (scalar log-weights, state dicts, buffer statistics) and are
inherited unchanged -- carrying along every fix accumulated on the flat-
tensor trainer (r-sign fix, per_time_anchor_target, gamma_floor, the
cross-rank all_gather gamma sync, find_unused_parameters, resumable
checkpointing) automatically.

Only methods that directly manipulate ThermoAtomicData / assume flat (B, D)
tensors are overridden here:
  - __init__ (potential/source/base_sde instantiation differ: atomic
    potentials take no `device` kwarg and have no `is_mean_free` attribute)
  - populate_buffer_adjoint_matching / populate_buffer_corrector_matching /
    populate_buffer_log_weights (center_every_step source, buffer-stat
    aggregation for the (1,)-per-graph storage convention)
  - adjoint_matching_step / corrector_matching_step (torch.cat on
    ThermoAtomicData doesn't work -- replaced with separate per-anchor-time
    forward calls; per-graph loss reduction via .reshape(batch_size, -1)
    relies on Ala2 being a HOMOGENEOUS batch, i.e. every graph has the same
    atom count, true for this fixed single-molecule system)
  - log_details / log_final (no potential.log_details()/visualize() for
    atomic potentials -- replaced with a simpler energy/buffer-stats-only
    version, matching train_wt_asbs.py's ASBS-baseline trainer)
"""

import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mdtraj as md
import torch
import numpy as np
import wandb
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig
from tqdm import tqdm

from tr_asbs.eval.calculate_torsion_metrics import calculate_torsion_metrics
from tr_asbs.eval.eval_utils import make_ramachandran_figure, plot_marginal_group_AD
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
from tr_asbs.utils.training import EMA


class GammaASBSAtomicModule(GammaASBSModule):
    """Ala2 (ThermoAtomicData) counterpart of GammaASBSModule."""

    def __init__(self, cfg: DictConfig):
        if getattr(cfg, "reference_preset", None) is not None or str(cfg.base_sde._target_).split(".")[-1] not in ("VESDE", "EDMSDE", "PISCosineSDE") or str(cfg.integrator._target_).split(".")[-1] not in ("AtomicExponentialIntegrator", "AtomicEulerMaruyamaIntegrator"):
            raise ValueError("Atomic trainer losses support legacy references/solvers; use the flat GammaASBSModule for alanine internal coordinates")
        # Deliberately does NOT call GammaASBSModule.__init__ (or
        # ASBSModule.__init__ super().__init__ chains) -- the atomic
        # potential/source/base_sde instantiation calls differ from the
        # flat-tensor version in ways that would crash if reused verbatim
        # (see module docstring). This is a full reimplementation of
        # __init__, kept in lockstep with GammaASBSModule.__init__'s
        # structure/fields intentionally -- only the marked lines differ.
        self.cfg = cfg
        import json
        import hashlib
        import uuid
        from omegaconf import OmegaConf

        def stable_group_id(cfg, exclude=("seed", "name")):
            if not isinstance(cfg, dict):
                cfg = OmegaConf.to_container(cfg, resolve=True)
            d = {k: cfg[k] for k in sorted(cfg.keys()) if k not in exclude}
            s = json.dumps(d, sort_keys=True)
            return hashlib.sha1(s.encode()).hexdigest()[:10]

        explicit_group_id = getattr(cfg, "checkpoint_group_id", None)
        random_hash = uuid.uuid4().hex[:8]
        self.group_id = explicit_group_id or (str(stable_group_id(cfg, exclude=("name"))) + f"{random_hash}")
        logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
        if "wandb" in logger_target and "group" in cfg.fabric.loggers:
            cfg.fabric.loggers.group = stable_group_id(cfg)

        wandb_logger = hydra.utils.instantiate(cfg.fabric.loggers)
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(full_cfg)

        self.gamma = cfg.gamma
        self.V0 = 1e-9
        assert 0 <= self.gamma <= 1
        self.gamma_schedule = cfg.gamma_schedule
        assert self.gamma_schedule in ["fixed", "trust_region"]
        self.trust_region_bound = cfg.trust_region_bound
        self.iw_clip_value = cfg.iw_clip_value

        self.fc_loss_mode = getattr(cfg, "fc_loss_mode", "cross_entropy")
        assert self.fc_loss_mode == "cross_entropy", (
            "log_variance (adjoint_matching_log_variance_step) has not been "
            "ported to the atomic pipeline -- use cross_entropy."
        )
        self.adjoint_buffer_duplicates = int(getattr(cfg, "adjoint_buffer_duplicates", 1))
        self.per_time_anchor_target = bool(getattr(cfg, "per_time_anchor_target", False))
        self.gamma_floor = float(getattr(cfg, "gamma_floor", 0.0))
        self.logvar_amp_dtype = None

        self.weight_rollback_enabled = getattr(cfg, "weight_rollback_enabled", False)
        self.weight_rollback_ess_threshold = getattr(cfg, "weight_rollback_ess_threshold", 1e-3)
        self.weight_rollback_patience = getattr(cfg, "weight_rollback_patience", 1)
        self._weight_rollback_bad_count = 0
        self._weight_rollback_checkpoint = None
        self._weight_rollback_count = 0

        self.trust_region_adaptive = getattr(cfg, "trust_region_adaptive", False)
        self.trust_region_gamma_low = getattr(cfg, "trust_region_gamma_low", 0.05)
        self.trust_region_gamma_high = getattr(cfg, "trust_region_gamma_high", 0.9)
        self.trust_region_growth_factor = getattr(cfg, "trust_region_growth_factor", 3.0)
        self.trust_region_shrink_factor = getattr(cfg, "trust_region_shrink_factor", 0.5)
        self.trust_region_patience = getattr(cfg, "trust_region_patience", 1)
        self.trust_region_bound_min = getattr(cfg, "trust_region_bound_min", 1e-3)
        self.trust_region_bound_max = getattr(cfg, "trust_region_bound_max", 1000.0)
        self.trust_region_energy_blowup_factor = getattr(cfg, "trust_region_energy_blowup_factor", 3.0)
        self._trust_region_stuck_low_count = 0
        self._trust_region_stuck_high_count = 0
        self._trust_region_growth_cooldown = 0
        self._trust_region_reference_energy = None
        self._last_logged_energy = None
        self._trust_region_low_bound = None
        self._trust_region_high_bound = None

        # Lightning fabric setup (same DDP find_unused_parameters fix as the
        # flat-tensor trainer -- PaiNN/atomic controller-corrector also don't
        # necessarily touch every parameter in every forward pass).
        fabric_overrides = {"loggers": wandb_logger}
        if cfg.fabric.get("devices", 1) not in (1, "1") and cfg.fabric.get("strategy", "auto") == "auto":
            from lightning.fabric.strategies import DDPStrategy

            fabric_overrides["strategy"] = DDPStrategy(find_unused_parameters=True)
        self.fabric = hydra.utils.instantiate(cfg.fabric, **fabric_overrides)
        self.fabric.launch()
        from lightning.fabric import seed_everything

        self.rank_seed = cfg.seed ^ self.fabric.global_rank
        seed_everything(self.rank_seed, workers=False)

        with self.fabric.init_module():
            # ATOMIC DIFFERENCE: no `device=` kwarg -- SumPotential/
            # OpenMMPotential/BasePotential don't take one (unlike BaseEnergy).
            self.potential = hydra.utils.instantiate(cfg.potential)
            controller = hydra.utils.instantiate(cfg.controller)
            corrector = hydra.utils.instantiate(cfg.corrector)
            if cfg.ema_decay is not None:
                self.controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)
                self.corrector = EMA(corrector, cfg.ema_decay, cfg.ema_start_step)
            else:
                self.controller = controller
                self.corrector = corrector

            self.adjoint_buffer = hydra.utils.instantiate(cfg.adjoint_buffer)
            self.corrector_buffer = hydra.utils.instantiate(cfg.corrector_buffer)

            # ATOMIC DIFFERENCE: no `device=` kwarg on source; is_mean_free
            # comes from cfg.base_sde itself (ala2.yaml sets
            # base_sde.is_mean_free: ${mean_free}), not from
            # self.potential.is_mean_free (SumPotential has no such
            # attribute).
            from tr_asbs.process.sde import ControlledFBSDE

            self.source = hydra.utils.instantiate(cfg.source)
            self.base_sde = hydra.utils.instantiate(cfg.base_sde)
            self.sde = ControlledFBSDE(
                base_sde=self.base_sde,
                u_controller=self.controller,
                h_controller=self.corrector,
            )
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential, corrector=self.corrector
            )

            from copy import deepcopy

            self.prev_integrator = deepcopy(self.integrator)
            self.prev_corrector = deepcopy(self.corrector)

        self.corrector_gamma_mode = getattr(cfg, "corrector_gamma_mode", "none") or "none"
        assert self.corrector_gamma_mode in (
            "none", "reuse_u_gamma", "independent", "joint_reuse", "joint_independent",
        )
        # "independent"/"joint_independent" route through the inherited
        # compute_corrector_gamma(), which assumes flat tensors
        # (data_0.shape[0], a CPU self.ones_tensor indexed by a CUDA
        # data.batch) and hasn't been adapted for ThermoAtomicData -- fail
        # fast here instead of a confusing crash mid-training.
        if self.corrector_gamma_mode in ("independent", "joint_independent"):
            raise NotImplementedError(
                f"corrector_gamma_mode={self.corrector_gamma_mode!r} is not yet "
                "supported by GammaASBSAtomicModule (compute_corrector_gamma "
                "assumes flat-tensor data) -- use 'none' or 'reuse_u_gamma'/"
                "'joint_reuse' (which don't call compute_corrector_gamma)."
            )
        self.corrector_trust_region_bound = getattr(cfg, "corrector_trust_region_bound", None)
        if self.corrector_trust_region_bound is None:
            self.corrector_trust_region_bound = cfg.trust_region_bound
        self.gamma_h = 1.0

        self.clear_adjoint_buffer_per_outer_loop = getattr(
            cfg, "clear_adjoint_buffer_per_outer_loop", False
        )

        self.max_outer_loops = cfg.max_outer_loops
        self.adjoint_matching_epochs = cfg.adjoint_matching_epochs
        self.corrector_matching_epochs = cfg.corrector_matching_epochs
        self.total_epochs = self.max_outer_loops * (
            self.adjoint_matching_epochs + self.corrector_matching_epochs
        )

        self.controller_optimizer = hydra.utils.instantiate(
            cfg.controller_optimizer, params=self.controller.parameters(),
        )
        self.corrector_optimizer = hydra.utils.instantiate(
            cfg.corrector_optimizer, params=self.corrector.parameters(),
        )
        self.controller, self.controller_optimizer = self.fabric.setup(
            self.controller, self.controller_optimizer
        )
        self.corrector, self.corrector_optimizer = self.fabric.setup(
            self.corrector, self.corrector_optimizer
        )

        # ATOMIC DIFFERENCE: no flat zeros/ones tensor of shape
        # (train_batch_size,) needed up front -- adjoint_matching_step/
        # corrector_matching_step below build per-call time tensors sized by
        # data_0.num_graphs directly.
        self.zeros_tensor = torch.zeros(())
        self.ones_tensor = torch.ones(())

        from tr_asbs.utils.helper import Checkpointer, CheckpointConfig

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
                    mode="min",
                ),
                make_payload=lambda: {"fbsde": self.sde.state_dict()},
            )

        if wandb.run is not None:
            wandb.run.summary["group_id"] = self.group_id
            wandb.run.summary["checkpoint_directory"] = str(self.checkpoint_directory)

        self.topology = None
        self.eval_traj = None
        self._load_eval_traj()

    def _load_eval_traj(self) -> None:
        """Load a reference trajectory for torsion-metric/Ramachandran
        evaluation in log_details, if configured. Mirrors the reference
        wt_diffusion_sampler trainer's _load_eval_traj: 1) md_eval_data
        (.npy) + topology_pdb_file, or 2) md_eval_pdb_file (trajectory PDB),
        optionally with topology_pdb_file for atom names.
        """
        topology_path = getattr(self.cfg, "topology_pdb_file", None)
        md_eval_data = getattr(self.cfg, "md_eval_data", None)
        md_eval_pdb_file = getattr(self.cfg, "md_eval_pdb_file", None)

        if topology_path is not None and os.path.exists(topology_path):
            try:
                self.topology = md.load_topology(topology_path)
            except Exception as exc:
                rank_zero_warn(f"Failed to load topology_pdb_file '{topology_path}': {exc}")
                self.topology = None

        if md_eval_data is not None and os.path.exists(md_eval_data):
            if self.topology is None:
                rank_zero_warn(
                    "md_eval_data is set but topology_pdb_file is missing/invalid; "
                    "skipping torsion-metric/Ramachandran logging."
                )
                return
            try:
                xyz = np.load(md_eval_data)
                self.eval_traj = md.Trajectory(xyz, self.topology)
                return
            except Exception as exc:
                rank_zero_warn(f"Could not load md_eval_data '{md_eval_data}': {exc}")

        if md_eval_pdb_file is not None and os.path.exists(md_eval_pdb_file):
            try:
                if self.topology is not None:
                    self.eval_traj = md.load(md_eval_pdb_file, top=topology_path)
                else:
                    self.eval_traj = md.load(md_eval_pdb_file)
            except Exception as exc:
                rank_zero_warn(f"Could not load md_eval_pdb_file '{md_eval_pdb_file}': {exc}")

    ############################################################################
    # Buffer and dataloader management (ThermoAtomicData-aware)
    ############################################################################

    @torch.no_grad()
    def populate_buffer_adjoint_matching(self):
        buffer = self.adjoint_buffer
        is_init_stage = self.current_epoch < self.cfg.adjoint_matching_epochs
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

        self.prev_integrator.eval()

        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            integral_results = self.prev_integrator.run_with_logweights(
                integrator=self.prev_integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,  # ATOMIC: was self.potential.is_mean_free
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )

            raw_log_Ph_Pref_mean = integral_results['log_Ph_Pref'].mean().item()
            r = min(abs(self.V0) / max(abs(raw_log_Ph_Pref_mean), 1e-12), 1.0)

            # IMPORTANT: unlike plain tensors, ThermoAtomicData.cpu()/.to()
            # mutate the object IN PLACE and return the same (now
            # CPU-resident) object -- so every GPU computation that still
            # needs data_0/data_1 (terminal_cost, control_1/control_0 below)
            # must happen BEFORE any .cpu() call on them, not after.
            data_1 = integral_results['data_1']
            terminal_cost_results = self.terminal_cost(
                data_1=data_1,
                is_init_stage=is_init_stage,
                return_energy=self.cfg.log_buffer_energy,
                return_grad_norm=self.cfg.log_buffer_grad_norm,
            )
            time_1 = self.fabric.to_device(
                torch.ones(data_0.num_graphs, dtype=torch.float)
            )
            control_1 = self.prev_integrator.sde.u_controller(time_1, data_1).detach().cpu()
            if self.per_time_anchor_target:
                time_0 = self.fabric.to_device(
                    torch.zeros(data_0.num_graphs, dtype=torch.float)
                )
                control_0 = self.prev_integrator.sde.u_controller(time_0, data_0).detach().cpu()

            populate_data = {
                "data_0": data_0.cpu(),  # ThermoAtomicData has no .detach()
                "data_1": data_1.cpu(),
                "log_Ph_Pref": (r * integral_results['log_Ph_Pref']).detach().cpu(),
                "log_Ph_Pref_raw": integral_results['log_Ph_Pref'].detach().cpu(),
                "log_Pu_Pref": integral_results['log_Pu_Pref'].detach().cpu(),
                "terminal_cost": terminal_cost_results["terminal_cost"].detach().cpu(),
                "control_1": control_1,
            }
            if self.per_time_anchor_target:
                populate_data["control_0"] = control_0

            if self.cfg.log_buffer_energy:
                populate_data["energy"] = terminal_cost_results["energy"].detach().cpu()
            if self.cfg.log_buffer_grad_norm:
                populate_data["grad_norm"] = terminal_cost_results["grad_norm"].detach().cpu()

            if self.adjoint_buffer_duplicates > 1:
                # ATOMIC: repeating a ThermoAtomicData batch isn't a plain
                # tensor .repeat -- duplicate via concatenating cloned copies
                # of the per-graph atomicdata list. Plain tensor entries
                # (terminal_cost/control_1/control_0/energy/grad_norm/log_*)
                # still use .repeat.
                dup = self.adjoint_buffer_duplicates
                populate_data = {
                    k: (
                        v.__class__.from_data_list(
                            [g.clone() for g in v.batch_to_atomicdata_list()] * dup
                        )
                        if hasattr(v, "batch_to_atomicdata_list")
                        else v.repeat(dup, *([1] * (v.dim() - 1)))
                    )
                    for k, v in populate_data.items()
                }
            buffer.extend(**populate_data)

        if self.global_step % self.cfg.log_loss_interval_step == 0:
            if self.cfg.log_buffer_energy:
                # ATOMIC: buffer.storage["energy"] holds (1,)-shaped
                # per-graph tensors (ThermoAtomicDataBuffer.extend splits
                # per-graph tensors via value[i].unsqueeze(0)), so use cat
                # (not the flat trainer's as_tensor-of-scalars / stack).
                median_energy = torch.cat(buffer.storage["energy"]).median()
                median_energy = self.fabric.all_reduce(
                    self.fabric.to_device(median_energy), reduce_op="mean"
                )
                self.fabric.log(
                    "train/adjoint_matching_buffer_median_energy",
                    median_energy,
                    step=self.global_step,
                )
                if self.cfg.save_samples and self.fabric.global_rank == 0:
                    if self.current_epoch % self.cfg.save_sample_interval_epoch == 0:
                        self.save_samples(integral_results['data_1'])
            if self.cfg.log_buffer_grad_norm:
                median_grad_norm = torch.cat(buffer.storage["grad_norm"]).median()
                median_grad_norm = self.fabric.all_reduce(
                    self.fabric.to_device(median_grad_norm), reduce_op="max"
                )
                self.fabric.log(
                    "train/adjoint_matching_buffer_median_grad_norm",
                    median_grad_norm,
                    step=self.global_step,
                )

            self.fabric.log(
                "train/adjoint_matching_buffer_size", len(buffer), step=self.global_step
            )
            self.fabric.log(
                "train/adjoint_matching_buffer_rescale_r", r, step=self.global_step
            )
            self.fabric.log(
                "train/adjoint_matching_buffer_raw_log_Ph_Pref_mean",
                raw_log_Ph_Pref_mean,
                step=self.global_step,
            )
            self.fabric.barrier()
        self.integrator.train()

    @torch.no_grad()
    def populate_buffer_corrector_matching(self):
        buffer = self.corrector_buffer
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

        self.integrator.eval()

        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,  # ATOMIC: was self.potential.is_mean_free
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            populate_data = {
                "data_0": data_0.cpu(),  # ThermoAtomicData has no .detach()
                "data_1": data_1.cpu(),
            }
            buffer.extend(**populate_data)

        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                "train/corrector_matching_buffer_size", len(buffer), step=self.global_step
            )
            self.fabric.barrier()

        self.integrator.train()

    def collect_log_weights_for_optimization(self):
        """ATOMIC override of GammaASBSModule.collect_log_weights_for_optimization
        -- identical except center_every_step comes from self.cfg.mean_free
        instead of self.potential.is_mean_free (SumPotential has no such
        attribute). The gamma dual-solve math itself (optimize_gamma,
        compute_V0_for_fixed_gamma) is untouched and inherited unchanged --
        it only consumes this method's plain-numpy return value.
        """
        self.integrator.eval()

        total_samples = self.cfg.num_samples_for_dual_loss[0]
        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        inference_batch_size = self.cfg.inference_batch_size

        paths_per_batch = inference_batch_size // batch_size_path
        num_iters = total_samples // paths_per_batch

        all_log_weights_np = []

        iterator = range(num_iters)
        if self.fabric.global_rank == 0 and self.cfg.progress_bar:
            iterator = tqdm(iterator, desc="Collecting weights", leave=False)

        for _ in iterator:
            torch.cuda.empty_cache()
            data_0 = self.source.replicate_sample((paths_per_batch,), batch_size_path)

            integral_results = self.integrator.run_with_logweights(
                integrator=self.integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,  # ATOMIC: was self.potential.is_mean_free
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=False,
            )

            log_Pu_Ph = -integral_results["log_Ph_Pref"].reshape(paths_per_batch, batch_size_path)
            all_log_weights_np.append(log_Pu_Ph.detach().cpu().numpy().astype(np.float64))

            del integral_results, log_Pu_Ph, data_0

        self.integrator.train()

        if len(all_log_weights_np) > 0:
            combined = np.concatenate(all_log_weights_np, axis=0)
        else:
            combined = np.array([])

        if self.fabric.world_size > 1 and combined.size > 0:
            local_tensor = self.fabric.to_device(torch.from_numpy(combined).float())
            gathered = self.fabric.all_gather(local_tensor)
            combined = gathered.reshape(-1, gathered.shape[-1]).detach().cpu().numpy().astype(np.float64)

        return combined

    def populate_buffer_log_weights(self, buffer):
        self.integrator.eval()
        batch_size_path = self.cfg.num_samples_for_dual_loss[1]
        num_iters = self.cfg.num_samples_for_dual_loss[0] // (self.cfg.inference_batch_size // self.cfg.num_samples_for_dual_loss[1])
        for j in range(num_iters):
            data_0 = self.source.replicate_sample((self.cfg.inference_batch_size // batch_size_path,), batch_size_path)
            integral_results = self.integrator.run_with_logweights(
                integrator=self.integrator,
                prior_log_prob_fn=self.source.log_prob,
                target_log_prob_fn=self.potential.log_prob,
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,  # ATOMIC: was self.potential.is_mean_free
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=self.cfg.progress_bar,
            )
            log_Pu_Ph = -integral_results["log_Ph_Pref"].reshape(self.cfg.inference_batch_size // batch_size_path, batch_size_path)
            buffer.extend(**{"log_weights": log_Pu_Ph.detach().cpu()})

        self.integrator.train()

    ############################################################################
    # Training steps: adjoint and corrector matching (ThermoAtomicData-aware)
    ############################################################################

    def adjoint_matching_step(self, batch) -> dict:
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]
        control_1 = batch["control_1"]

        batch_size = data_0.num_graphs

        iw = self._compute_iw(batch, self.gamma)  # (B,)

        time_0 = self.fabric.to_device(torch.zeros(batch_size, dtype=torch.float))
        time_1 = self.fabric.to_device(torch.ones(batch_size, dtype=torch.float))
        time = self.fabric.to_device(
            torch.rand(batch_size, dtype=torch.float)
        ) * (1 - 2e-3) + 1e-3

        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        if self.per_time_anchor_target:
            control_0 = batch["control_0"]
            with torch.no_grad():
                control_t = self.prev_integrator.sde.u_controller(time, data_t)
            target_0 = self.gamma * (terminal_cost + control_0) - control_0
            target_1 = self.gamma * (terminal_cost + control_1) - control_1
            target_t = self.gamma * (terminal_cost + control_t) - control_t
        else:
            target = self.gamma * (terminal_cost + control_1) - control_1
            target_0 = target_1 = target_t = target

        # ATOMIC: torch.cat on ThermoAtomicData batches isn't supported --
        # three separate forward calls (one per anchor time) instead of the
        # flat trainer's single fused (3B,...) forward. Mathematically
        # identical; costs 3 kernel launches instead of 1.
        control_pred_0 = self.controller(time_0, data_0)
        control_pred_1 = self.controller(time_1, data_1)
        control_pred_t = self.controller(time, data_t)

        def per_sample_mse(control, target):
            # Relies on Ala2 being a homogeneous batch (constant atoms per
            # graph), so reshape(batch_size, -1) correctly separates graphs.
            return (control + target).pow(2).reshape(batch_size, -1).mean(-1)

        loss_0 = (per_sample_mse(control_pred_0, target_0) * iw).mean()
        loss_1 = (per_sample_mse(control_pred_1, target_1) * iw).mean()
        loss_t = (per_sample_mse(control_pred_t, target_t) * iw).mean()

        loss = loss_0 + loss_t + loss_1
        return dict(
            loss=loss,
            loss_0=loss_0.detach().cpu(),
            loss_t=loss_t.detach().cpu(),
            loss_1=loss_1.detach().cpu(),
        )

    def corrector_matching_step(self, batch) -> dict:
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]

        batch_size = data_0.num_graphs

        time = self.fabric.to_device(torch.rand(batch_size) * (1 - 2e-3) + 1e-3)
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)
        time_1 = self.fabric.to_device(torch.ones(batch_size, dtype=torch.float))

        # ATOMIC: separate forward calls instead of torch.cat-then-split.
        h_t = self.corrector(time, data_t)
        h_1 = self.corrector(time_1, data_1)

        score_t = self.base_sde.cond_score_t0(time=time, data_0=data_0, data_t=data_t)
        score_1 = self.base_sde.cond_score_t0(time=time_1, data_0=data_0, data_t=data_1)

        if self.corrector_gamma_mode != "none":
            reference = (
                self.controller
                if self.corrector_gamma_mode in ("joint_reuse", "joint_independent")
                else self.prev_corrector
            )
            with torch.no_grad():
                ref_t = reference(time, data_t)
                ref_1 = reference(time_1, data_1)
            target_t = self.gamma_h * score_t + (1 - self.gamma_h) * ref_t
            target_1 = self.gamma_h * score_1 + (1 - self.gamma_h) * ref_1
        else:
            target_t = score_t
            target_1 = score_1

        # coef_t is computed directly from `time` (shape (B,), one value per
        # graph), so it already aligns with the per-graph-reduced loss below
        # -- no per-atom broadcasting needed.
        coef_t = 1 + self.base_sde.diffusion(time).pow(2) * (1 - time) / time

        loss_t = (
            (h_t - target_t)
            .pow(2)
            .reshape(batch_size, -1)
            .mean(-1)
            .div(coef_t)
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

    ############################################################################
    # Sample saving (already generic in the base class -- save_samples/
    # save_final_samples both branch on isinstance(data, torch.Tensor) vs
    # else (ase/xyz), so ThermoAtomicData is already handled correctly
    # without an override; see GammaASBSModule.save_samples/save_final_samples.)
    ############################################################################

    ############################################################################
    # Evaluation/logging (simplified -- no potential.log_details()/visualize()
    # for atomic potentials; matches train_wt_asbs.py's simpler ASBS-baseline
    # buffer-stats-only approach instead of the flat trainer's rich
    # Eq-EMD2/W_2 point-cloud metrics)
    ############################################################################

    @torch.no_grad()
    def log_details(self, step: int, visualize: bool = False, eval_samples: int = None) -> dict:
        visualize = visualize or self.cfg.visualize
        eval_samples = eval_samples if eval_samples is not None else self.cfg.eval_samples
        self.integrator.eval()
        data_0 = self.source.sample((eval_samples,))
        integral_results = self.integrator.run_with_logweights(
            integrator=self.prev_integrator,
            prior_log_prob_fn=self.source.log_prob,
            target_log_prob_fn=self.potential.log_prob,
            initial_data=data_0,
            center_every_step=self.cfg.mean_free,
            zero_last_step_noise=False,
            return_trajectory=False,
            progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
        )
        data_1 = integral_results["data_1"]
        potential_result = self.potential(data_1)
        mean_energy = potential_result["energy"].mean()

        logs = {
            "loss/neg_avg_energy": mean_energy.detach().cpu().item(),
            "loss/log_Ph_Pref_mean": integral_results["log_Ph_Pref"].mean().detach().cpu().item(),
            "loss/log_Pu_Pref_mean": integral_results["log_Pu_Pref"].mean().detach().cpu().item(),
        }

        if self.eval_traj is not None and self.topology is not None:
            try:
                # Reference repo convention: internal positions are in
                # Angstrom, mdtraj.Trajectory expects nanometers.
                xyz = data_1.pos.reshape(data_1.num_graphs, -1, 3).detach().cpu().numpy() / 10.0
                traj = md.Trajectory(xyz, self.topology)
                kld_logs = plot_marginal_group_AD(gen_traj=traj, ref_traj=self.eval_traj)
                if not visualize and "Vis/marginal" in kld_logs:
                    kld_logs.pop("Vis/marginal")
                logs.update(kld_logs)
                logs.update(
                    calculate_torsion_metrics(
                        self.eval_traj, traj,
                        temp=float(getattr(self.cfg, "temperature", 300.0)),
                    )
                )
                if visualize:
                    rama_fig = make_ramachandran_figure(traj)
                    if rama_fig is not None:
                        logs["Vis/ramachandran"] = wandb.Image(rama_fig)
                        plt.close(rama_fig)
            except Exception as exc:
                rank_zero_warn(f"Torsion/Ramachandran logging skipped due to evaluation error: {exc}")

        if self.current_epoch > 0 and len(self.adjoint_buffer) > 0:
            log_Ph_Pref = torch.cat(self.adjoint_buffer.storage["log_Ph_Pref"])
            log_Pu_Pref = torch.cat(self.adjoint_buffer.storage["log_Pu_Pref"])
            log_pred = self.gamma * log_Ph_Pref + (1 - self.gamma) * log_Pu_Pref
            ess = (log_Ph_Pref.exp().sum() ** 2 / (log_Ph_Pref.exp() ** 2).sum() / len(log_Ph_Pref)).item()
            logs["RN[h|u]/ess"] = ess
            logs["RN[h|u]/mean"] = log_Ph_Pref.mean().item()
            logs["lm"] = self.lm
            logs["gamma"] = self.gamma
            if self.corrector_gamma_mode != "none":
                logs["gamma_h"] = self.gamma_h
            del log_pred

        logs["epoch"] = self.current_epoch
        logs["eval_step"] = step

        self.fabric.log_dict(
            {k: v for k, v in logs.items() if k not in ("epoch", "eval_step")},
            step=self.global_step,
        )
        self._last_logged_energy = logs["loss/neg_avg_energy"]

        if self.cfg.save_samples and self.fabric.global_rank == 0 and visualize:
            self.save_samples(data_1)

        logs.update({"data_0": data_0.cpu(), "data_1": data_1.cpu()})  # ThermoAtomicData has no .detach()
        del integral_results
        return logs

    def log_final(self):
        final_logs = self.log_details(
            self.current_epoch // self.cfg.log_detail_interval_epoch,
            visualize=True,
            eval_samples=self.cfg.eval_samples * 2,
        )
        data_0, data_1 = final_logs.pop("data_0"), final_logs.pop("data_1")
        self.save_final_samples(data_0, data_1, name="last_epoch")
        if self.fabric.global_rank == 0 and wandb.run is not None:
            wandb.run.summary["seed"] = int(self.cfg.seed)
            for key, value in final_logs.items():
                if key in ("epoch", "eval_step"):
                    continue
                wandb.run.summary[f"final/{key}"] = value
            wandb.run.summary["final/epoch"] = int(self.current_epoch)
            wandb.run.summary["final/global_step"] = int(self.global_step)

        for name in self.ckpts.keys():
            ckpt_path = Path(self.checkpoint_directory) / f"{name}_best.pt"
            if not ckpt_path.exists():
                continue
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
                center_every_step=self.cfg.mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
            )
            self.save_final_samples(data_0, data_1, name=f"best_{name}")

            if self.fabric.global_rank == 0 and wandb.run is not None:
                wandb.run.summary[f"best_{name}/step"] = int(step)
                wandb.run.summary[f"best_{name}/value"] = float(best_val)

            if self.fabric.global_rank == 0:
                ckpt_path.unlink(missing_ok=True)

        wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    import os

    logger_target = str(getattr(cfg.fabric.loggers, "_target_", "")).lower()
    use_wandb_logger = "wandb" in logger_target
    if use_wandb_logger:
        os.environ["WANDB_BASE_URL"] = str(cfg.wandb_host)
    torch.set_float32_matmul_precision("high")
    trainer = GammaASBSAtomicModule(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
