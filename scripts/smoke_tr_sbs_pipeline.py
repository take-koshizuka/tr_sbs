#!/usr/bin/env python3
"""Strictly bounded CPU pipeline verification; no remote calls or training runs.

Four samples, four intervals, one FC and one BC optimizer step per case.
Missing molecular reference datasets are explicitly not_evaluated, never faked.
"""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
import traceback

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import torch

from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
from tr_asbs.data.buffer import BatchBuffer
from tr_asbs.utils.training import copy_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/tr_asbs/config"
CASES = {
    "legacy_ce": dict(fc_loss_mode="cross_entropy", reference_preset=None),
    "diffusion_logvar": dict(
        fc_loss_mode="log_variance", reference_preset="brownian_geometric"
    ),
    "diffusion_pathkl": dict(
        fc_loss_mode="path_kl", reference_preset="brownian_constant"
    ),
    "ou_pathkl": dict(fc_loss_mode="path_kl", reference_preset="ou"),
    "vp_logvar": dict(fc_loss_mode="log_variance", reference_preset="vp"),
}
PRIMARY_CASES = list(CASES)
CASES.update(
    {
        "euler_logvar": dict(
            fc_loss_mode="log_variance",
            reference_preset="brownian_geometric",
            reference_solver="euler",
        ),
        "tau_cosine_logvar": dict(
            fc_loss_mode="log_variance",
            reference_preset="brownian_cosine",
            reference_grid="uniform_tau",
        ),
        "edm_logvar": dict(
            fc_loss_mode="log_variance", reference_preset="brownian_edm"
        ),
        "midpoint_logvar": dict(
            fc_loss_mode="log_variance",
            reference_preset="ou",
            reference_control_evaluation="midpoint",
        ),
        "dual_time_logvar": dict(
            fc_loss_mode="log_variance",
            reference_preset="brownian_geometric",
            time_coordinates="both",
        ),
    }
)
TARGETS = {
    "gauss": "gauss",
    "gmm": "gmm_mode",
    "ring": "ring_gmm",
    "manywell": "mw5",
    "dw4": "dw4",
    "lj13": "lj13",
    "alanine_ic": "ad_ic_reference",
}


def configuration(target, case, output):
    with initialize_config_dir(str(CONFIG), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=" + TARGETS[target]])
    with open_dict(cfg):
        cfg.root = str(output)
        cfg.fabric = dict(
            _target_="lightning.fabric.Fabric",
            accelerator="cpu",
            devices=1,
            strategy="auto",
            num_nodes=1,
            loggers=dict(
                _target_="lightning.pytorch.loggers.CSVLogger",
                save_dir=str(output),
                name="csv",
            ),
        )
        cfg.checkpoint_directory = str(output / "checkpoints")
        cfg.checkpoint_group_id = target + "_" + case
        cfg.sample_directory = str(output / "samples")
        cfg.monitor_values, cfg.monitor_modes = [], {}
        cfg.seed, cfg.compile, cfg.resume = 7, False, False
        cfg.ema_decay = None
        cfg.progress_bar, cfg.visualize, cfg.save_samples = False, False, False
        cfg.log_buffer_energy, cfg.log_buffer_grad_norm = False, False
        cfg.train_batch_size = cfg.inference_batch_size = 4
        cfg.initial_buffer_samples = cfg.buffer_samples_per_epoch = 4
        cfg.num_samples_for_dual_loss = [2, 2]
        cfg.eval_samples = cfg.eval_samples_final = 4
        cfg.max_outer_loops = cfg.adjoint_matching_epochs = (
            cfg.corrector_matching_epochs
        ) = cfg.steps_per_epoch = 1
        cfg.log_loss_interval_step, cfg.log_detail_interval_epoch = 1, 100
        cfg.corrector_gamma_mode = "none"
        cfg.clear_adjoint_buffer_per_outer_loop = True
        cfg.reference_steps, cfg.reference_noise_knots = 4, 16
        cfg.reference_variance = 0.25 if target == "alanine_ic" else 1.0
        cfg.reference_grid = "uniform_t"
        cfg.bc_loss_weight = "reference_variance" if case != "legacy_ce" else "legacy"
        for name, value in CASES[case].items():
            if name != "time_coordinates":
                cfg[name] = value
        cfg.source.scale = 1.0
        if target in ("dw4", "lj13"):
            cfg.potential.ref_samples_path = None
            for name in ("controller", "corrector"):
                cfg[name].hidden_nf, cfg[name].n_layers = 16, 2
        else:
            for name in ("controller", "corrector"):
                cfg[name].channels, cfg[name].num_layers = 16, 2
                cfg[name].time_coordinates = CASES[case].get(
                    "time_coordinates",
                    "physical" if case == "legacy_ce" else "variance",
                )
        if cfg.reference_preset is None:
            cfg.integrator.timesteps.steps = 4
        if target == "alanine_ic":
            cfg.potential.transform_max_frames = cfg.potential.reference_max_frames = 16
            cfg.potential.transform_offset, cfg.potential.reference_offset = 0, 128
            cfg.potential.transform_stride = cfg.potential.reference_stride = 1
        cfg.adjoint_buffer.max_size = cfg.corrector_buffer.max_size = 8
    return cfg


def check_case(target, case, output):
    started = time.perf_counter()
    cfg = configuration(target, case, output)
    module = GammaASBSModule(cfg)
    module.global_step, module.current_epoch = 0, 0
    copy_model(module.prev_integrator, module.integrator)
    copy_model(module.prev_corrector, module.corrector)
    module.gamma, module.V0 = module.optimize_gamma()
    assert 0 <= module.gamma <= 1 and torch.isfinite(torch.tensor(module.V0))
    module.populate_buffer_adjoint_matching()
    batch = BatchBuffer.collate_fn([module.adjoint_buffer[i] for i in range(4)])
    assert module.training_step(batch, "adjoint_matching"), "Nonfinite FC loss/gradient"
    fc = module.last_training_losses.copy()
    module.global_step += 1
    module.populate_buffer_corrector_matching()
    batch_h = BatchBuffer.collate_fn([module.corrector_buffer[i] for i in range(4)])
    assert module.training_step(batch_h, "corrector_matching"), (
        "Nonfinite BC loss/gradient"
    )
    bc = module.last_training_losses.copy()
    module.current_epoch = 2
    evaluation = module.log_details(step=1, visualize=False, eval_samples=4)
    assert torch.isfinite(torch.tensor(evaluation["trust_region/achieved_forward_kl"]))
    x0 = module.source.sample((4,))
    result = module.integrator.run_with_logweights(
        module.prev_integrator,
        module.source.log_prob,
        module.potential.log_prob,
        x0,
        center_every_step=module.potential.is_mean_free,
        return_trajectory=True,
    )
    for name in ["data_1", "log_Ph_Pref", "log_Pu_Pref", "log_Pref_Pu_det"]:
        assert torch.isfinite(result[name]).all(), name + " is nonfinite"
    backwards = module.integrator.run_backward(
        result["data_1"],
        center_every_step=module.potential.is_mean_free,
        return_trajectory=False,
        progress_bar=False,
    )
    assert torch.isfinite(backwards).all()
    saved_step = module.global_step
    module.current_epoch = 2
    frozen_before = {
        k: v.clone() for k, v in module.prev_integrator.state_dict().items()
    }
    module.save_checkpoint(periodic=True)
    for parameter in module.prev_integrator.parameters():
        parameter.data.add_(0.01)
    module.global_step = -1
    module.load_checkpoint()
    assert module.global_step == saved_step and module.current_epoch == 2
    for key, value in frozen_before.items():
        torch.testing.assert_close(module.prev_integrator.state_dict()[key], value)
    # A changed grid must fail before loading weights into models.
    original_times = module.integrator.timesteps.clone()
    module.integrator.timesteps[1] += 0.001
    try:
        module.load_checkpoint()
    except ValueError:
        checkpoint_guard = True
    else:
        raise AssertionError("Changed-grid checkpoint was accepted")
    module.integrator.timesteps.copy_(original_times)
    extra = {}
    if target == "alanine_ic":
        native = module.potential.sample(2, coordinate="native")
        x = module.potential.native_to_cartesian(native)
        native_back = module.potential.cartesian_to_native(x)
        assert native.shape == (2, 60) and x.shape == (2, 66)
        roundtrip = (native - native_back).abs().max().item()
        assert roundtrip < 0.02
        z = module.potential.aldp_potential.x_to_z(x)
        complete = module.potential.aldp_potential.p.coordinate_transform.transform
        ic = complete.ic_transform
        circular_index = (
            int(torch.nonzero(complete.permute[9:] == ic.dih_indices[0])[0]) + 3
        )
        shifted = z[:1].clone()
        shifted[:, circular_index] += 2 * torch.pi / ic.std_dih[0]
        alias_x = module.potential.aldp_potential.z_to_x(shifted)
        alias_error = (alias_x - x[:1]).abs().max().item()
        alias_energy_error = (
            (
                module.potential.aldp_potential._log_prob(shifted)
                - module.potential.aldp_potential._log_prob(z[:1])
            )
            .abs()
            .max()
            .item()
        )
        mapped, logjac = module.potential.chart(native)
        torch.testing.assert_close(
            module.potential.eval(native),
            -module.potential.aldp_potential._log_prob(mapped) - logjac,
        )
        try:
            module.potential.chart.inverse(shifted)
        except ValueError:
            alias_rejected = True
        else:
            raise AssertionError("IC chart accepted a duplicated periodic coordinate")
        extra = dict(
            native_dim=60,
            cartesian_dim=66,
            roundtrip_max_error=roundtrip,
            target_gradient_finite=bool(
                torch.isfinite(module.potential.grad_E(native)).all()
            ),
            data=module.potential.data_metadata,
            periodic_alias_cartesian_error=alias_error,
            periodic_alias_energy_error=alias_energy_error,
            ic_chart_jacobian_verified=True,
            periodic_alias_rejected=alias_rejected,
            periodic_geometry_status="single_chart_pipeline_verified",
            openmm_configurations_evaluated=module.potential.aldp_potential.openmm_configurations_evaluated,
            openmm_batch_calls=module.potential.aldp_potential.openmm_batch_calls,
        )
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    row = dict(
        target=target,
        case=case,
        status="passed",
        samples=4,
        steps=4,
        fc_optimizer_steps=1,
        bc_optimizer_steps=1,
        fc=fc,
        bc=bc,
        gamma=module.gamma,
        dual_V0=module.V0,
        achieved_forward_kl_on_current_paths=float(result["log_Pref_Pu_det"].mean()),
        checkpoint_roundtrip=True,
        frozen_snapshot_roundtrip=True,
        evaluation_logging=True,
        checkpoint_mismatch_rejected=checkpoint_guard,
        quality_status="not_evaluated",
        seconds=time.perf_counter() - started,
        kernel=module.integrator.kernel_metadata(),
        **extra,
    )
    del module
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGETS,
        default=["gauss", "gmm", "ring", "manywell", "dw4", "lj13"],
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=PRIMARY_CASES)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/tr_sbs_pipeline_smoke"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(
        protocol="pipeline_smoke_only",
        expensive_jobs_submitted=False,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        pytorch=torch.__version__,
        device="cpu",
        rows=[],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        (args.output / f"report_previous_{time.time_ns()}.json").write_text(
            (args.output / "report.json").read_text()
        )
    failures = 0
    for target in args.targets:
        for case in args.cases:
            output = args.output / target / case
            output.mkdir(parents=True, exist_ok=True)
            try:
                row = check_case(target, case, output)
            except Exception as exc:
                failures += 1
                row = dict(
                    target=target,
                    case=case,
                    status="failed",
                    error=str(exc),
                    traceback=traceback.format_exc(),
                    quality_status="not_evaluated",
                )
            report["rows"].append(row)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
            print(
                json.dumps(
                    {k: row[k] for k in ("target", "case", "status")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
