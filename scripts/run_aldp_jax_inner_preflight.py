#!/usr/bin/env python3
"""Paired GPU verification/timing of JAX u/h updates in the shared ALDP runner.

This is a backend preflight, not an ALDP quality or shortened-training experiment.
The declared reference retains u2000/h1000 epochs and 100 updates per epoch.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import jax
import numpy as np
import torch

from run_aldp_chart_campaign import ROOT, receipt, sha256, verify_source, write_json
from run_aldp_component_campaign import configuration as component_configuration
from run_aldp_mechanism_campaign import diagnostic_rng, paired_physical_source_audit
from tr_asbs.experiment.jax_inner import (
    JaxInnerBackend, bc_loss, export_fourier, from_torch, lv_loss, prepare_bc_batch,
    prepare_lv_batch,
)
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
from tr_asbs.utils.training import copy_model


def check_files(files):
    for path, expected in files.items():
        if sha256(path) != expected:
            raise RuntimeError("Frozen input SHA mismatch: " + path)


def configuration(plan, campaign):
    parent = json.loads(Path(plan["parent_plan_path"]).read_text())
    cfg, _ = component_configuration(parent, campaign, "mechanism_pilot", "ou", 0)
    from omegaconf import open_dict
    output = campaign / "backend_preflight" / "jax_seed0"
    with open_dict(cfg):
        cfg.name, cfg.root = output.name, str(output)
        cfg.fabric.loggers.save_dir = str(output)
        cfg.checkpoint_directory = str(output / "checkpoints")
        cfg.sample_directory = str(output / "samples")
        cfg.max_outer_loops = 1
        cfg.adjoint_matching_epochs, cfg.corrector_matching_epochs = 2000, 1000
        cfg.initial_adjoint_matching_epochs = None
        cfg.steps_per_epoch = 100
        cfg.log_detail_interval_epoch = 250
    return cfg, output


def initialize_backend_probe(module):
    # These are normally set by fit(). This preflight deliberately exercises
    # the shared buffer/step APIs without running the training curriculum.
    module.current_epoch = 0
    module.global_step = 0


def torch_prepared_bc(module, prepared):
    def tensor(name):
        return torch.utils.dlpack.from_dlpack(prepared[name]).to(module.fabric.device)
    time, data_t, data_1 = tensor("time"), tensor("data_t"), tensor("data_1")
    size = len(time)
    values = module.corrector(torch.cat((time, torch.ones_like(time))),
                              torch.cat((data_t, data_1)))
    loss_t = ((values[:size] - tensor("score_t"))**2).mean(-1).div(tensor("coef_t")).mean()
    return loss_t + ((values[size:] - tensor("score_1"))**2).mean()


def gradient_comparison(module, phase, batch, tolerances):
    network = module.controller if phase == "u" else module.corrector
    model, params, coeff, spec = export_fourier(network)
    network.zero_grad(set_to_none=True)
    with diagnostic_rng(module.fabric.device):
        if phase == "u":
            prepared = prepare_lv_batch(module, batch)
            _, old, _, _ = export_fourier(module.prev_integrator.sde.u_controller)
            def objective(p):
                return lv_loss(p, old, coeff, spec, prepared, module.gamma)[0]
            expected = module.adjoint_matching_log_variance_step(batch)["loss"]
        else:
            prepared = prepare_bc_batch(module, batch)
            def objective(p):
                return bc_loss(p, coeff, spec, prepared)[0]
            expected = torch_prepared_bc(module, prepared)
        (actual, gradients) = jax.jit(jax.value_and_grad(objective))(params)
        jax.block_until_ready((actual, gradients))
        expected.backward()
        diff_norm, baseline_norm, actual_norm, max_error = 0., 0., 0., 0.
        details = {}
        for name, p in model.named_parameters():
            reference = p.grad.detach().double().cpu().numpy()
            value = np.asarray(gradients[name], dtype=np.float64)
            error = value-reference
            diff_norm += float(np.square(error).sum())
            baseline_norm += float(np.square(reference).sum())
            actual_norm += float(np.square(value).sum())
            max_error = max(max_error, float(np.abs(error).max()))
            details[name] = dict(max_abs_error=float(np.abs(error).max()),
                                 reference_max_abs=float(np.abs(reference).max()))
        relative = diff_norm**.5 / max(1e-12, baseline_norm**.5)
        expected_value, actual_value = float(expected.detach()), float(actual)
        loss_error = abs(expected_value-actual_value) / max(1., abs(expected_value))
        passed = (all(np.isfinite(x) for x in (relative, loss_error, max_error))
            and relative <= tolerances["gradient_relative_l2"]
            and loss_error <= tolerances["loss_relative"])
        network.zero_grad(set_to_none=True)
    return dict(status="passed" if passed else "failed", phase=phase,
        torch_loss=expected_value, jax_loss=actual_value,
        loss_scaled_absolute_error=loss_error, gradient_relative_l2_error=relative,
        torch_gradient_norm_float64=baseline_norm**.5, jax_gradient_norm_float64=actual_norm**.5,
        gradient_max_absolute_error=max_error, parameters=details)


def synchronize():
    torch.cuda.synchronize()


def benchmark_updates(module, phase, batch, iterations, warmup, tolerances):
    """Shared actual training_step, including transfers/logging/acceptance.

    Both sides start with the same actual trained parameters, Adam state and
    identical Torch RNG. Healthy probes use fresh Adam; the rejected-batch probe
    retains its saved 8208-update moments and count. Timing includes JAX staging and a final Torch
    parameter/Adam synchronization. Kernel-only timing is not labelled end-to-end.
    """
    current_task = "adjoint_matching" if phase == "u" else "corrector_matching"
    models = [export_fourier(module.controller)[0], export_fourier(module.corrector)[0]]
    initial_weights = [deepcopy(m.state_dict()) for m in models]
    optimizers = [module.controller_optimizer, module.corrector_optimizer]
    initial_optimizers = [deepcopy(o.state_dict()) for o in optimizers]
    original_step = module.training_step

    def adam_count(state):
        values = {int(value["step"]) for value in state["state"].values()}
        if len(values) > 1:
            raise RuntimeError("Different Adam counters in a paired benchmark")
        return next(iter(values)) if values else 0

    selected = 0 if phase == "u" else 1
    initial_count = adam_count(initial_optimizers[selected])

    def reset():
        for model, weights in zip(models, initial_weights):
            model.load_state_dict(weights)
        for optimizer, state in zip(optimizers, initial_optimizers):
            # load_state_dict can reuse same-device state tensors. Its in-place
            # Adam updates would otherwise mutate this saved restart baseline.
            optimizer.load_state_dict(deepcopy(state))
        module.global_step = 0
        torch.manual_seed(23547)
        torch.cuda.manual_seed_all(23547)

    def run(step, count):
        rejects = []
        for index in range(count):
            if step(batch, current_task):
                module.global_step += 1
            else:
                rejects.append(index)
        return rejects

    with diagnostic_rng(module.fabric.device):
        reset()
        run(original_step, warmup)
        synchronize()
        reset()
        synchronize()
        torch_start_count = adam_count(optimizers[selected].state_dict())
        start = time.perf_counter()
        torch_rejects = run(original_step, iterations)
        synchronize()
        torch_seconds = time.perf_counter()-start
        torch_end_count = adam_count(optimizers[selected].state_dict())
        torch_values = [{k: v.detach().cpu().numpy().copy() for k, v in model.named_parameters()}
                        for model in models]
        torch_cpu_rng, torch_cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        reset()
        backend = JaxInnerBackend(module)
        initial_jax_params = dict(backend.params)
        initial_jax_states = dict(backend.states)

        def reset_jax():
            reset()
            backend.params = dict(initial_jax_params)
            backend.states = dict(initial_jax_states)
            backend.old_params = None

        synchronize()
        cold_start = time.perf_counter()
        backend.training_step(batch, current_task)
        synchronize()
        cold_seconds = time.perf_counter()-cold_start
        reset_jax()
        run(backend.training_step, warmup)
        synchronize()
        reset_jax()
        synchronize()
        jax_start_count = int(backend.states[phase].count)
        start = time.perf_counter()
        jax_rejects = run(backend.training_step, iterations)
        backend.flush(optimizers=True)
        synchronize()
        jax_seconds = time.perf_counter()-start
        jax_end_count = int(backend.states[phase].count)
        rng_matches = torch.equal(torch_cpu_rng, torch.get_rng_state()) and torch.equal(
            torch_cuda_rng, torch.cuda.get_rng_state())
        errors = {}
        diff_norm, update_norm = 0., 0.
        for index, model in enumerate(models):
            for name, value in model.named_parameters():
                reference = torch_values[index][name]
                actual = value.detach().cpu().numpy()
                initial = initial_weights[index][name].detach().cpu().numpy()
                difference = actual.astype(np.float64)-reference
                errors[f"{index}:{name}"] = float(np.abs(difference).max())
                diff_norm += float(np.square(difference).sum())
                update_norm += float(np.square(reference.astype(np.float64)-initial).sum())
        maximum = max(errors.values())
        relative = diff_norm**.5/max(1e-12, update_norm**.5)
        # Restore reference models; these are test updates, not a training pilot.
        reset()
    counters_match = (initial_count == torch_start_count == jax_start_count
                      and torch_end_count == jax_end_count == initial_count+iterations)
    passed = (not torch_rejects and not jax_rejects and rng_matches and counters_match
        and np.isfinite(relative) and maximum <= tolerances["parameter_max_absolute"]
        and relative <= tolerances["update_relative_l2"])
    return dict(status="passed" if passed else "failed", phase=phase,
        paired_updates=iterations, warmup_updates=warmup,
        torch_seconds=torch_seconds, jax_seconds=jax_seconds,
        jax_first_call_seconds=cold_seconds,
        torch_per_update_seconds=torch_seconds/iterations,
        jax_per_update_seconds=jax_seconds/iterations,
        warm_speed_ratio=torch_seconds/jax_seconds,
        torch_rejected_updates=torch_rejects, jax_rejected_updates=jax_rejects,
        rng_matches=rng_matches, parameter_max_absolute_error=maximum,
        adam_restart_counters=dict(initial=initial_count, torch_start=torch_start_count,
            jax_start=jax_start_count, torch_end=torch_end_count, jax_end=jax_end_count),
        adam_restart_counters_match=counters_match,
        update_relative_l2_error=relative, parameter_errors=errors,
        timing_scope="actual neural training_step + JAX staging + one final model/Adam sync; excludes buffer generation",
        first_call_includes_compilation_and_execution=True,
        estimates_are_not_full_campaign_speedup=True)


def numerical_repair_preflight(module, plan):
    """Use the saved rejected state/batch, including its8208-step Adam state."""
    from dataclasses import replace
    path=Path(plan['numerical_repair_evidence_path'])
    evidence=torch.load(path,map_location='cpu',weights_only=False)
    if (evidence['schema']!='aldp_numerical_rejection_batch_v1'
            or evidence['counters']['global_step']!=8208 or evidence['adam']['count']!=8208):
        raise RuntimeError('Registered rejected batch/state required')
    controller=export_fourier(module.controller)[0]
    teacher=export_fourier(module.prev_integrator.sde.u_controller)[0]
    previous=dict(controller=deepcopy(controller.state_dict()),teacher=deepcopy(teacher.state_dict()),
        optimizer=deepcopy(module.controller_optimizer.state_dict()),gamma=module.gamma)
    try:
        with torch.no_grad():
            for model,key in ((controller,'params'),(teacher,'old_params')):
                for name,p in model.named_parameters():p.copy_(evidence[key][name].to(p))
                model.timestep_embed.timestep_coeff.copy_(evidence['time_coeff'].to(
                    model.timestep_embed.timestep_coeff))
        # Fabric's actual optimizer state is used by the shared Torch step and
        # exported by JaxInnerBackend; preserve count and both moments.
        raw=module.controller_optimizer
        while hasattr(raw,'optimizer') and raw.optimizer is not raw:raw=raw.optimizer
        for name,p in controller.named_parameters():
            raw.state[p]=dict(step=torch.tensor(float(evidence['adam']['count'])),
                exp_avg=evidence['adam']['m'][name].to(p),
                exp_avg_sq=evidence['adam']['v'][name].to(p))
        module.gamma=np.float64(evidence['counters']['gamma'])
        batch={name:evidence['prepared'][name].to(module.fabric.device)
               for name in ('data_traj','log_Ph_Pref_raw')}
        prepared=prepare_lv_batch(module,batch)
        for name,value in prepared.items():
            actual=torch.utils.dlpack.from_dlpack(value).cpu()
            torch.testing.assert_close(actual,evidence['prepared'][name],rtol=0,atol=0)
        model,params,coeff,spec=export_fourier(controller)
        dim=prepared['data_traj'].shape[-1]
        data=prepared['data_traj'][:,:-1].reshape((-1,dim))
        times=jax.numpy.broadcast_to(prepared['times'],
            (len(batch['data_traj']),prepared['data_traj'].shape[1]-1)).reshape(-1)
        from tr_asbs.experiment.jax_inner import fourier_mlp
        unclipped=jax.jit(lambda p:fourier_mlp(p,coeff,replace(spec,clip_norm=None),times,data))(params)
        bounded=jax.jit(lambda p:fourier_mlp(p,coeff,spec,times,data))(params)
        unclipped=np.asarray(unclipped);bounded=np.asarray(bounded)
        intended=np.clip(unclipped,-spec.clip_norm,spec.clip_norm)
        legacy=unclipped+(intended-unclipped)
        clipping=dict(forward_exact_bound=bool(np.array_equal(bounded,intended)),
            maximum_absolute=float(np.abs(bounded).max()),
            unclipped_maximum_absolute=float(np.abs(unclipped).max()),
            legacy_wrong_forward_fraction=float(np.mean(legacy!=intended)),
            derivative='straight_through identity, unchanged')
        gradient=gradient_comparison(module,'u',batch,plan['tolerances'])
        updates=benchmark_updates(module,'u',batch,plan['paired_updates'],plan['warmup_updates'],plan['tolerances'])
        passed=(gradient['status']=='passed' and updates['status']=='passed'
                and clipping['forward_exact_bound'])
        return dict(status='passed' if passed else 'failed_equivalence',
            evidence_sha256=sha256(path),prior_adam_updates=8208,
            clipping=clipping,gradient=gradient,updates=updates,
            target_recomputed=False,openmm_configurations=0,
            tolerance_relaxation=False,normal_reference_completed=False)
    finally:
        controller.load_state_dict(previous['controller']);teacher.load_state_dict(previous['teacher'])
        module.controller_optimizer.load_state_dict(previous['optimizer']);module.gamma=previous['gamma']


def run(plan, campaign, source_sha):
    from omegaconf import OmegaConf
    cfg, output = configuration(plan, campaign)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output already exists; retain it and use a new attempt")
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    if cfg.adjoint_matching_epochs != 2000 or cfg.corrector_matching_epochs != 1000 or cfg.steps_per_epoch != 100:
        raise RuntimeError("Normal inner training contract changed")
    check_files(plan["frozen_inputs"])
    parent_receipt = json.loads(Path(plan["parent_receipt_path"]).read_text())
    parent_directory = Path(plan["parent_receipt_path"]).parent
    check_files({str(parent_directory / k): v for k, v in parent_receipt["files"].items()})
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Both Torch and JAX must see a GPU; no CPU fallback")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    module = GammaASBSModule(cfg)
    initialize_backend_probe(module)
    counters = module.potential.aldp_potential
    module.fabric.load(plan["parent_checkpoint_path"],
        dict(controller=module.controller, corrector=module.corrector))
    module.prev_integrator = copy_model(module.prev_integrator, module.integrator)
    module.prev_integrator.requires_grad_(False)
    timings = {}
    with diagnostic_rng(module.fabric.device):
        torch.manual_seed(12455)
        y = module.source.sample((64,))
        paired_source = paired_physical_source_audit(module, y,
            module.potential.native_to_cartesian(y), "low_density")
    synchronize()
    before = counters.openmm_configurations_evaluated
    start = time.perf_counter()
    module.gamma, module.V0 = module.optimize_gamma()
    synchronize()
    timings["dual"] = dict(seconds=time.perf_counter()-start,
                           openmm_configurations=counters.openmm_configurations_evaluated-before)
    frozen = {}
    for phase, name, task in (("u", "populate_buffer_adjoint_matching", "adjoint_matching"),
                             ("h", "populate_buffer_corrector_matching", "corrector_matching")):
        before = counters.openmm_configurations_evaluated
        synchronize()
        start = time.perf_counter()
        getattr(module, name)()
        frozen[phase] = next(iter(module.create_dataloader(task)))
        synchronize()
        timings[phase+"_buffer"] = dict(seconds=time.perf_counter()-start,
            openmm_configurations=counters.openmm_configurations_evaluated-before)
    torch.save({phase: {k: v.detach().cpu() for k, v in batch.items()}
                for phase, batch in frozen.items()}, output / "paired_batches.pt")
    gradients, updates = [], []
    for phase in ("u", "h"):
        gradients.append(gradient_comparison(module, phase, frozen[phase], plan["tolerances"]))
        write_json(output / "progress.json", dict(gradients=gradients, updates=updates))
        updates.append(benchmark_updates(module, phase, frozen[phase],
            plan["paired_updates"], plan["warmup_updates"], plan["tolerances"]))
        write_json(output / "progress.json", dict(gradients=gradients, updates=updates))
    numerical_repair=(numerical_repair_preflight(module,plan)
        if plan.get('numerical_repair_evidence_path') else None)
    if counters.openmm_configurations_evaluated > plan["max_openmm_configurations"]:
        raise RuntimeError("Backend preflight OpenMM budget exceeded")
    passed = (all(row["status"] == "passed" for row in gradients+updates)
              and (numerical_repair is None or numerical_repair['status']=='passed'))
    result = dict(schema="aldp_jax_inner_preflight_v1", status="passed" if passed else "failed_equivalence",
        scientific_status="not_evaluated", no_quality_gate=True,
        source_sha256=source_sha, plan_sha256=sha256(campaign / "plan.json"),
        config_sha256=sha256(output / "config.yaml"), paired_source_audit=paired_source,
        gradients=gradients, updates=updates, timings=timings,numerical_repair=numerical_repair,
        gamma=float(module.gamma), physical_energy_mode=module.potential.aldp_potential.p.energy_mode,
        cost=dict(total_openmm_configurations=counters.openmm_configurations_evaluated,
            openmm_batch_calls=counters.openmm_batch_calls, wall_seconds=time.perf_counter()-started),
        environment=dict(torch=torch.__version__, jax=jax.__version__,
            gpu=torch.cuda.get_device_name(), jax_devices=[str(x) for x in jax.devices()]),
        reference_training_contract=dict(u_epochs=2000, h_epochs=1000, steps_per_epoch=100,
            outer_loops=1, u_updates=200000, h_updates=100000,
            training_batch_size=int(cfg.train_batch_size), integration_steps=200,
            reference_training_submitted=False),
        sampling_metadata=module.sampling_metadata())
    write_json(output / "result.json", result)
    receipt(output, dict(stage="backend_preflight", candidate="jax", seed=0),
        dict(source_sha256=source_sha, plan_sha256=result["plan_sha256"], frozen_inputs=plan["frozen_inputs"]))
    if not passed:
        raise RuntimeError("Paired JAX/Torch equivalence gate failed; details and receipt retained")
    print("postflight_verified=true", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / "plan.json").read_text())
    try:
        source_sha = verify_source()
        run(plan, campaign, source_sha)
    except Exception:
        output = campaign / "backend_preflight" / "jax_seed0"
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", dict(status="failed", traceback=traceback.format_exc(),
                                                scientific_status="not_evaluated"))
        raise


if __name__ == "__main__":
    main()
