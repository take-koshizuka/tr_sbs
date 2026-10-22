"""Bounded saved-batch LV/Adam comparison; never generates samples or forces."""
import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_aldp_first_nonfinite import safe
from run_aldp_scaled_backward_gate import compare, manual_adam
from tr_asbs.utils.scaled_backward_diagnostic import power_of_two_scale, unscale_and_clip


def reference_adam_step(parameters, gradients, optimizer_state):
    """Independent FP64 Adam, retaining moments for the next reference step."""
    state = copy.deepcopy(optimizer_state)
    groups = state['param_groups']
    assert len(groups) == 1
    group = groups[0]
    assert not group.get('amsgrad', False) and group['weight_decay'] == 0
    assert not group.get('maximize', False)
    assert len(parameters) == len(group['params'])
    b1, b2 = group['betas']
    updated = manual_adam(parameters, gradients, optimizer_state)
    for name, index in zip(parameters, group['params']):
        g = gradients[name].double()
        previous = optimizer_state['state'].get(index, {})
        # All moments use the same unchanged beta/epsilon/LR as production.
        m = previous.get('exp_avg', torch.zeros_like(g)).to(g)
        v = previous.get('exp_avg_sq', torch.zeros_like(g)).to(g)
        state['state'][index] = dict(
            step=int(previous.get('step', 0)) + 1,
            exp_avg=m*b1 + g*(1-b1),
            exp_avg_sq=v*b2 + g.square()*(1-b2))
    return updated, state


def counters(optimizer_state):
    return sorted({int(v['step']) for v in optimizer_state['state'].values()})


def state_parameters(model):
    """Use the same key namespace as Fabric's delegated state_dict loader."""
    original = getattr(model, '_original_module', model)
    parameters = dict(original.named_parameters())
    assert set(parameters) == set(model.state_dict()), 'Registered controller state must contain parameters only'
    return parameters


def validate_plan(plan):
    assert plan['updates_per_case'] == 20
    assert plan['cases'] == ['normal', 'witness']
    assert plan['optimizer_updates'] == 40
    assert plan['reference_optimizer_updates'] == 40
    assert plan['maximum_backward_passes'] == 120
    assert plan['expected_OpenMM'] == plan['OpenMM_cap'] == 0
    assert not plan['runtime']['fresh_sampling']
    assert plan['runtime']['GPU_required']
    assert plan['runtime']['production_dtype'] == 'FP32'
    assert plan['runtime']['matmul_precision'] == 'highest'
    assert not plan['production_changed'] and not plan['full_training_submitted']
    assert plan['optimizer_updates_discarded']


def row_gates(row, tolerances):
    return dict(
        finite=row['all_gradients_and_parameters_finite'],
        loss=row['loss_relative_error'] <= tolerances['loss_relative'],
        clipped_gradient=row['clipped_gradient_error']['relative_l2'] <= tolerances['gradient_relative_l2'],
        parameter=row['parameter_error']['maximum_absolute'] <= tolerances['parameter_max_absolute'],
        update=row['update_error']['relative_l2'] <= tolerances['update_relative_l2'],
        cumulative_parameter=row['cumulative_parameter_error']['maximum_absolute'] <= tolerances['parameter_max_absolute'],
        cumulative_update=row['cumulative_update_error']['relative_l2'] <= tolerances['update_relative_l2'],
        Adam_counter=row['Adam_counters'] == row['reference_Adam_counters'] == row['expected_Adam_counters'],
    )


def run_case(label, config_path, payload, batch, output, plan, save_progress):
    from omegaconf import OmegaConf, open_dict
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.root = str(output/label)
        cfg.fabric.loggers.save_dir = cfg.root
        cfg.checkpoint_directory = str(output/'unused_checkpoints')
        cfg.sample_directory = str(output/'unused_samples')
    module = GammaASBSModule(cfg)
    assert module.fabric.world_size == 1 and module.fabric.device.type == 'cuda'
    assert module.logvar_amp_dtype is None
    assert payload['sampling_metadata'] == module.sampling_metadata()
    OmegaConf.save(cfg, output/(label+'_config.yaml'), resolve=True)
    device = module.fabric.device
    original_optimizer = copy.deepcopy(payload['optimizers']['controller_optimizer'])
    original_rng = payload.get('rng', payload.get('pre_forward_rng'))
    assert original_rng is not None
    for name, state in payload['models'].items():
        getattr(module, name).load_state_dict(state)
    module.prev_integrator.requires_grad_(False)
    module.prev_corrector.requires_grad_(False)
    for name, value in payload['counters'].items():
        setattr(module, name, value)
    for name, state in payload['buffers'].items():
        buffer = getattr(module, name)
        assert buffer.max_size == state['max_size']
        buffer.clear()
        buffer.storage = defaultdict(list, state['storage'])
        buffer.metadata = copy.deepcopy(state['metadata'])
    module.controller_optimizer.load_state_dict(copy.deepcopy(original_optimizer))
    parameters = state_parameters(module.controller)
    batch = {n: v.to(device) for n, v in batch.items()}
    batch64 = {n: v.double() for n, v in batch.items()}

    def restore_rng():
        torch.set_rng_state(original_rng['torch_cpu'])
        torch.cuda.set_rng_state_all(original_rng['torch_cuda'])
        np.random.set_state(original_rng['numpy'])
        random.setstate(original_rng['python'])

    def load_weights(weights, dtype):
        module.integrator.to(dtype=dtype)
        module.prev_integrator.to(dtype=dtype)
        module.controller.to(dtype=dtype)
        module.controller.load_state_dict(weights)
        # Never round the reference teacher through a mutated FP32 tensor.
        module.prev_integrator.load_state_dict(payload['models']['prev_integrator'])
        module.prev_integrator.requires_grad_(False)
        module.controller_optimizer.zero_grad(set_to_none=True)

    def gradient64(weights):
        load_weights(weights, torch.float64)
        restore_rng()
        loss = module.adjoint_matching_log_variance_step(batch64)['loss']
        assert torch.isfinite(loss), 'Nonfinite independent FP64 loss'
        loss.backward()
        gradient = {n: p.grad.detach().double().clone() for n, p in parameters.items()}
        assert all(torch.isfinite(g).all() for g in gradient.values()), 'Nonfinite independent FP64 gradient'
        norm = torch.sqrt(sum(g.square().sum() for g in gradient.values()))
        coefficient = min(1., float(cfg.gradient_clip_val/(norm+1e-6)))
        return loss.detach().clone(), {n: g*coefficient for n, g in gradient.items()}, float(norm)

    actual = {n: p.detach().clone() for n, p in parameters.items()}
    initial = {n: v.clone() for n, v in actual.items()}
    actual_state = copy.deepcopy(module.controller_optimizer.state_dict())
    reference = {n: v.double().clone() for n, v in actual.items()}
    reference_state = copy.deepcopy(actual_state)
    initial_counters = counters(original_optimizer) or [0]
    rows = []
    for number in range(1, plan['updates_per_case']+1):
        # Each local comparison starts from identical actual weights/moments.
        reference_loss, reference_gradient, reference_norm = gradient64(actual)
        expected = manual_adam(actual, reference_gradient, actual_state)
        # Also advance an independent full-FP64 trajectory to bound cumulative drift.
        if number == 1:
            trajectory_gradient = reference_gradient
        else:
            _, trajectory_gradient, _ = gradient64(reference)
        reference, reference_state = reference_adam_step(reference, trajectory_gradient, reference_state)
        previous = {n: v.clone() for n, v in actual.items()}
        load_weights(actual, torch.float32)
        # Deepcopy prevents Adam from mutating the reference baseline by aliasing.
        module.controller_optimizer.load_state_dict(copy.deepcopy(actual_state))
        restore_rng()
        loss = module.adjoint_matching_log_variance_step(batch)['loss']
        scale = power_of_two_scale(loss)
        module.fabric.backward(loss*scale)
        norm = unscale_and_clip(module.controller.parameters(), scale, cfg.gradient_clip_val)
        gradient = {n: p.grad.detach().clone() for n, p in parameters.items()}
        module.controller_optimizer.step()
        actual = {n: p.detach().clone() for n, p in parameters.items()}
        actual_state = copy.deepcopy(module.controller_optimizer.state_dict())
        row = dict(case=label, update=number, scale=scale, norm64=float(norm),
            reference_norm64=reference_norm, loss=float(loss.detach()), reference_loss=float(reference_loss),
            loss_relative_error=float((loss.detach().double()-reference_loss).abs()/reference_loss.abs().clamp_min(1e-300)),
            clipped_gradient_error=compare(gradient, reference_gradient),
            parameter_error=compare(actual, expected),
            update_error=compare({n: actual[n].double()-previous[n] for n in actual},
                                 {n: expected[n]-previous[n] for n in actual}),
            cumulative_parameter_error=compare(actual, reference),
            cumulative_update_error=compare({n: actual[n].double()-initial[n] for n in actual},
                                            {n: reference[n]-initial[n] for n in actual}),
            Adam_counters=counters(actual_state), reference_Adam_counters=counters(reference_state),
            expected_Adam_counters=[v+number for v in initial_counters],
            all_gradients_and_parameters_finite=all(bool(torch.isfinite(v).all()) for v in [*gradient.values(), *actual.values(), *reference.values()]),
            restored_counters=payload['counters'], frozen_RNG_and_buffers_restored=True,
            sampling_metadata_verified=True, matmul_precision=torch.get_float32_matmul_precision(), OpenMM=0)
        row['gates'] = row_gates(row, plan['tolerances'])
        row['status'] = 'passed' if all(row['gates'].values()) else 'failed'
        assert module.potential.aldp_potential.openmm_configurations_evaluated == 0
        rows.append(row)
        save_progress(rows)
    # The independent reference uses double moments; only actual clipped updates
    # retain original FP32 dtype. Both paths are discarded after this diagnosis.
    torch.save(dict(actual_parameters=actual, actual_optimizer=actual_state,
                    reference_parameters=reference, reference_optimizer=reference_state,
                    initial_Adam_counters=initial_counters, frozen_counters=payload['counters']),
               output/(label+'_final_diagnostic_state.pt'))
    return rows


def main():
    from run_aldp_chart_campaign import verify_source, sha256, write_json, receipt
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    campaign = parser.parse_args().campaign
    plan = json.loads((campaign/'plan.json').read_text())
    validate_plan(plan)
    source = verify_source()
    assert source == plan['source_manifest_sha256']
    for name, expected in plan['frozen_inputs'].items():
        assert sha256(name) == expected, name
    assert plan['tolerances'] == json.loads(Path(plan['tolerance_origin']).read_text())['tolerances']
    parent = json.loads(Path(plan['verified_single_step_gate']).read_text())
    assert parent['status'] == 'passed' and parent['tolerances'] == plan['tolerances']
    assert parent['optimizer_updates'] == 2 and parent['OpenMM'] == 0
    assert not parent['production_changed']
    assert all(all(row['gates'].values()) for row in parent['rows'])
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    output = campaign/'diagnostic'
    assert not output.exists()
    output.mkdir()
    normal = torch.load(plan['normal_snapshot'], map_location='cpu', weights_only=False)
    witness = torch.load(plan['witness'], map_location='cpu', weights_only=False)
    storage = normal['buffers']['adjoint_buffer']['storage']
    normal_batch = {n: torch.stack(v[:64]) for n, v in storage.items()}
    rows = []
    for label, path, payload, batch in [('normal', plan['normal_config'], normal, normal_batch),
                                      ('witness', plan['witness_config'], witness, witness['batch'])]:
        base_rows = list(rows)
        def save_progress(case_rows):
            write_json(output/'gate.json', safe(dict(rows=base_rows+case_rows, status='partial', OpenMM=0)))
        rows.extend(run_case(label, path, payload, batch, output, plan, save_progress))
    passed = all(row['status'] == 'passed' for row in rows)
    write_json(output/'gate.json', safe(dict(rows=rows, status='passed' if passed else 'failed',
        source_sha256=source, plan_sha256=sha256(campaign/'plan.json'), OpenMM=0,
        optimizer_updates=len(rows), reference_optimizer_updates=len(rows),
        updates_per_case=plan['updates_per_case'], optimizer_updates_discarded=True,
        tolerances=plan['tolerances'], matmul_precision=torch.get_float32_matmul_precision(),
        production_changed=False, full_training_submitted=False,
        limits='Two frozen batches repeated for 20 updates each; same-state and independent FP64 trajectory comparisons. No fresh sampling, phase changes, or quality/admission claim.')))
    # Retain a complete receipt even when an original numerical gate fails.
    receipt(output, dict(stage='scaled_backward_multistep', seed=0),
            dict(source_sha256=source, plan_sha256=sha256(campaign/'plan.json')))
    assert passed, 'Original multistep clipped-gradient/Adam tolerances not met'
    print('scaled_backward_multistep_passed=true', flush=True)


if __name__ == '__main__':
    main()
