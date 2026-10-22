#!/usr/bin/env python3
"""Bounded replay of a failed u phase; preserve the offending batch and state.

This is numerical diagnosis, not a replacement normal-duration reference.
The shared loop, frozen policy, buffer generation, sampler, Adam and RNG stay
unchanged.  Only a phase-start snapshot may be used by this diagnostic.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import torch

from run_aldp_chart_campaign import receipt, sha256, verify_source, write_json
from run_aldp_jax_inner_preflight import check_files
from tr_asbs.experiment.inner_observation import restore_inner_snapshot
from tr_asbs.experiment.jax_inner import JaxInnerBackend, export_fourier, fourier_mlp, lv_loss


def finite_number(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_summary(value):
    array = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    finite = np.isfinite(array)
    values = array[finite].astype(np.float64)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
        nonfinite_elements=int((~finite).sum()),
        maximum_absolute=float(np.abs(values).max()) if values.size else None,
        quantiles=np.quantile(values, [0, .5, .99, 1]).tolist() if values.size else [])


def gradient_summary(gradients):
    details = {name:tensor_summary(value) for name,value in gradients.items()}
    squared = sum(float(np.square(np.asarray(
        value.detach().cpu() if isinstance(value, torch.Tensor) else value,
        dtype=np.float64)).sum()) for value in gradients.values())
    return dict(norm_float64=finite_number(math.sqrt(squared)),
        nonfinite_elements=sum(item['nonfinite_elements'] for item in details.values()),
        parameters=details)


def lv_loss_float64_reduction(params, old, coeff, spec, batch, gamma):
    """Diagnostic: original float32 network/kernel, float64 weight variance."""
    trajectory = batch['data_traj']
    size, states, dim = trajectory.shape
    steps = states - 1
    x, xn = trajectory[:, :-1].reshape((-1, dim)), trajectory[:, 1:].reshape((-1, dim))
    t = jnp.broadcast_to(batch['times'], (size, steps)).reshape(-1)
    phi, k, q = (jnp.broadcast_to(batch[name], (size, steps)).reshape((-1, 1))
                 for name in ('phi', 'k', 'q'))
    new = phi*x + k*fourier_mlp(params, coeff, spec, t, x)
    previous = phi*x + k*jax.lax.stop_gradient(fourier_mlp(old, coeff, spec, t, x))
    def log_density(mean):
        return -.5*(jnp.sum((xn-mean)**2/q, axis=-1)+dim*jnp.log(2*jnp.pi*q[:, 0]))
    ratio = (log_density(new)-log_density(previous)).reshape((size, steps)).sum(-1)
    weights = ratio.astype(jnp.float64)-jnp.asarray(gamma, jnp.float64)*batch[
        'log_Ph_Pref_raw'].reshape(-1).astype(jnp.float64)
    return jnp.var(weights, ddof=1)


def diagnose_rejection(module, backend, batch, attempted, output):
    params, adam, prepared, old, gamma, values = attempted
    backend.flush(optimizers=True)
    _, _, coeff, spec = export_fourier(backend.models['u'])
    def objective(p):
        return lv_loss(p, old, coeff, spec, prepared, gamma)[0]
    jax_loss, gradients = jax.jit(jax.value_and_grad(objective))(params)
    loss64, gradients64 = jax.jit(jax.value_and_grad(
        lambda p:lv_loss_float64_reduction(p, old, coeff, spec, prepared, gamma)))(params)
    module.controller.zero_grad(set_to_none=True)
    torch_loss = module.adjoint_matching_log_variance_step(batch)['loss']
    torch_loss.backward()
    model = backend.models['u']
    torch_gradients = {name:p.grad for name,p in model.named_parameters()}
    reference_gradients = gradient_summary(torch_gradients)
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), module.cfg.gradient_clip_val,
                                              error_if_nonfinite=True)
        torch_clip = dict(accepted=bool(torch.isfinite(torch_loss)), norm=finite_number(norm))
    except RuntimeError as error:
        torch_clip = dict(accepted=False, error=str(error))
    def host(value):
        return torch.from_numpy(np.asarray(value).copy())
    torch.save(dict(schema='aldp_numerical_rejection_batch_v1', exact_mid_epoch_resume=False,
        counters=dict(epoch=int(module.current_epoch), global_step=int(module.global_step),
                      phase_epoch=int(module.phase_epoch), gamma=float(gamma)),
        prepared={name:host(value) for name,value in prepared.items()},
        params={name:host(value) for name,value in params.items()},
        old_params={name:host(value) for name,value in old.items()},
        adam=dict(m={name:host(value) for name,value in adam.m.items()},
                  v={name:host(value) for name,value in adam.v.items()}, count=int(adam.count)),
        time_coeff=host(coeff), model_spec=vars(spec), adam_spec=vars(backend.specs['u'])),
        output/'rejected_batch.pt')
    return dict(jax_original=dict(loss=finite_number(jax_loss),
            returned_metrics={name:finite_number(value) for name,value in values.items()},
            gradients=gradient_summary(gradients)),
        torch_same_jax_weights_batch=dict(loss=finite_number(torch_loss),
            gradients=reference_gradients, clipping=torch_clip),
        jax_float64_variance_diagnostic=dict(loss=finite_number(loss64),
            gradients=gradient_summary(gradients64), changes_production_math=False),
        batch={name:tensor_summary(value) for name,value in prepared.items()},
        params={name:tensor_summary(value) for name,value in params.items()})


class ReplayLimitReached(Exception):
    pass


class ReplayRejected(Exception):
    pass


def run_bounded_u_replay(module, snapshot, provenance, epoch_limit, on_rejection, progress):
    """Direct shared u loop from its empty, pre-population epoch-zero boundary."""
    payload = torch.load(snapshot, map_location='cpu', weights_only=False)
    counters = payload['counters']
    if (counters['phase'], counters['phase_epoch'], counters['global_step'],
            counters['outer_step']) != ('u', 0, 0, 0):
        raise ValueError('Numerical replay requires the exact first u0 boundary')
    if payload['buffers']['adjoint_buffer']['storage']:
        raise ValueError('u0 replay requires an empty adjoint buffer')
    restore_inner_snapshot(module, snapshot, expected_provenance=provenance)
    original_step, original_populate = module.training_step, module.populate_buffer_adjoint_matching
    def populate():
        if module.phase_epoch >= epoch_limit:
            raise ReplayLimitReached()
        original_populate()
    def step(batch, task):
        accepted = original_step(batch, task)
        if not accepted:
            on_rejection(batch)
            raise ReplayRejected()
        if module.global_step % module.cfg.steps_per_epoch == 0:
            progress(module)
        return accepted
    module.populate_buffer_adjoint_matching, module.training_step = populate, step
    try:
        module.inner_loop_u()
    except ReplayLimitReached:
        return 'no_rejection_within_registered_replay_limit'
    except ReplayRejected:
        return 'numerical_rejection_captured'
    finally:
        module.populate_buffer_adjoint_matching, module.training_step = original_populate, original_step
    raise RuntimeError('Shared phase ended before the diagnostic replay limit')


def run(plan, campaign, source_sha):
    from omegaconf import OmegaConf, open_dict
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    output = campaign/'numerical_diagnostic/seed0'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Preserve previous diagnostic output; use a fresh attempt')
    output.mkdir(parents=True)
    cfg = OmegaConf.load(plan['failed_config_path'])
    if ((cfg.adjoint_matching_epochs,cfg.corrector_matching_epochs,cfg.steps_per_epoch)
            != (2000,1000,100) or not 0<plan['replay_epoch_limit']<cfg.log_detail_interval_epoch):
        raise ValueError('Replay must preserve normal inner settings and stop before online250')
    with open_dict(cfg):
        cfg.root, cfg.name = str(output), output.name
        cfg.fabric.loggers.save_dir = str(output)
        cfg.checkpoint_directory, cfg.sample_directory = str(output/'checkpoints'), str(output/'samples')
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    module = GammaASBSModule(cfg)
    backend = JaxInnerBackend(module).attach()
    # Capture diagnostics without changing the compiled update or accepted state.
    original_update, attempted = backend.updates['u'], {}
    def update(*args):
        result = original_update(*args)
        values = {name:np.asarray(value).item() for name,value in result[2].items()}
        if not values['accepted']:
            attempted['rejected'] = (*args, values)
        return result
    backend.updates['u'] = update
    start = time.perf_counter()
    before = module.potential.aldp_potential.openmm_configurations_evaluated
    diagnosis = {}
    with (output/'progress.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['global_step','phase_epoch','loss','gradient_norm'])
        writer.writeheader();stream.flush()
        def progress(current):
            values = current.last_training_losses
            writer.writerow(dict(global_step=int(current.global_step)+1,
                phase_epoch=int(current.phase_epoch), loss=values['loss'],
                gradient_norm=values['gradient_norm']))
            stream.flush()
        def reject(batch):
            diagnosis.update(diagnose_rejection(module, backend, batch, attempted['rejected'], output))
        status = run_bounded_u_replay(module, plan['u0_snapshot_path'], plan['u0_provenance'],
                                     plan['replay_epoch_limit'], reject, progress)
    delta = module.potential.aldp_potential.openmm_configurations_evaluated-before
    if delta > plan['openmm_cap']:
        raise RuntimeError('Registered numerical diagnostic OpenMM cap exceeded')
    result = dict(status=status, scientific_status='not_evaluated', source_sha256=source_sha,
        plan_sha256=sha256(campaign/'plan.json'), config_sha256=sha256(output/'config.yaml'),
        failed_job_id=plan['failed_job_id'], global_step=int(module.global_step),
        epoch=int(module.current_epoch), phase_epoch=int(module.phase_epoch),
        openmm_configurations=delta, neural_updates=int(backend.states['u'].count),
        wall_seconds=time.perf_counter()-start, diagnosis=diagnosis,
        expected_failure_counters=plan['failure_counters'],
        replay_counter_match=(int(module.global_step)==plan['failure_counters']['global_step']
            and int(module.phase_epoch)==plan['failure_counters']['phase_epoch']),
        same_gpu_model_as_original=torch.cuda.get_device_name()==plan['failed_gpu_name'],
        exact_mid_epoch_resume_available=False, normal_reference_completed=False)
    write_json(output/'result.json', result)
    receipt(output, dict(stage='jax_numerical_diagnostic',chart='low_density',seed=0),
            dict(source_sha256=source_sha, plan_sha256=sha256(campaign/'plan.json')))
    print(json.dumps({k:v for k,v in result.items() if k!='diagnosis'}, allow_nan=False), flush=True)
    print('postflight_verified=true', flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    args=parser.parse_args();campaign=args.campaign.resolve()
    plan=json.loads((campaign/'plan.json').read_text());source_sha=verify_source()
    if source_sha!=plan['source_manifest_sha256']:
        raise RuntimeError('Diagnostic source manifest changed')
    check_files(plan['frozen_inputs'])
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Numerical replay requires the remote GPU')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    torch.set_float32_matmul_precision('highest')
    try:
        run(plan,campaign,source_sha)
    except Exception as error:
        write_json(campaign/'numerical_diagnostic/seed0/failure.json', dict(
            error=repr(error),traceback=traceback.format_exc(),scientific_status='not_evaluated'))
        raise


if __name__=='__main__':
    main()
