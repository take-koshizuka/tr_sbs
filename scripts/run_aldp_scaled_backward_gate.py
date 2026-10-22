"""Same-batch clipped-gradient and resumed-Adam gate, with no OpenMM calls."""
import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_aldp_first_nonfinite import safe
from tr_asbs.utils.scaled_backward_diagnostic import power_of_two_scale, unscale_and_clip


def compare(a, b):
    numerator = sum((a[n].double() - b[n].double()).square().sum() for n in a)
    denominator = sum(b[n].double().square().sum() for n in a)
    return dict(relative_l2=float((numerator / denominator.clamp_min(1e-300)).sqrt()),
                maximum_absolute=max(float((a[n].double() - b[n].double()).abs().max()) for n in a))


def manual_adam(parameters, gradients, optimizer_state):
    """Independent FP64 arithmetic using the unchanged Adam hyperparameters."""
    result = {}
    names = list(parameters)
    groups = optimizer_state['param_groups']
    assert len(groups) == 1
    group = groups[0]
    assert not group.get('amsgrad', False) and group['weight_decay'] == 0 and not group.get('maximize', False)
    assert len(names) == len(group['params'])
    b1, b2 = group['betas']
    for name, index in zip(names, group['params']):
        state = optimizer_state['state'].get(index, {})
        step = int(state.get('step', 0)) + 1
        g = gradients[name].double()
        m = state.get('exp_avg', torch.zeros_like(g)).double() * b1 + g * (1-b1)
        v = state.get('exp_avg_sq', torch.zeros_like(g)).double() * b2 + g.square() * (1-b2)
        denominator = v.sqrt() / (1-b2**step)**.5 + group['eps']
        result[name] = parameters[name].double() - (group['lr'] / (1-b1**step)) * m / denominator
    return result


def run_case(label, config_path, payload, batch, output, tolerances):
    from omegaconf import OmegaConf, open_dict
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    from tr_asbs.utils.gradient_clip import clip_gradients_with_overflow_fallback
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.root = str(output/label)
        cfg.fabric.loggers.save_dir = cfg.root
        cfg.checkpoint_directory = str(output/'unused_checkpoints')
        cfg.sample_directory = str(output/'unused_samples')
    module = GammaASBSModule(cfg)
    assert module.fabric.world_size == 1 and module.fabric.device.type == 'cuda'
    assert module.logvar_amp_dtype is None, 'This gate requires the original FP32 path'
    assert payload['sampling_metadata'] == module.sampling_metadata(), 'Frozen configuration/state mismatch'
    cfg_path = output / (label+'_config.yaml')
    OmegaConf.save(cfg, cfg_path, resolve=True)
    device = module.fabric.device
    original_models = payload['models']
    original_optimizer = copy.deepcopy(payload['optimizers']['controller_optimizer'])
    batch = {n: value.to(device) for n, value in batch.items()}
    original_rng = payload.get('rng', payload.get('pre_forward_rng'))
    assert original_rng is not None

    def reset(dtype):
        module.integrator.to(dtype=dtype)
        module.prev_integrator.to(dtype=dtype)
        module.controller.to(dtype=dtype)
        for n, state in original_models.items():
            getattr(module, n).load_state_dict(state)
        module.prev_integrator.requires_grad_(False)
        module.prev_corrector.requires_grad_(False)
        for n, value in payload['counters'].items():
            setattr(module, n, value)
        for n, state in payload['buffers'].items():
            buffer = getattr(module, n)
            assert buffer.max_size == state['max_size']
            buffer.clear()
            buffer.storage = defaultdict(list, state['storage'])
            buffer.metadata = copy.deepcopy(state['metadata'])
        module.controller_optimizer.load_state_dict(copy.deepcopy(original_optimizer))
        module.controller_optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(original_rng['torch_cpu'])
        torch.cuda.set_rng_state_all(original_rng['torch_cuda'])
        np.random.set_state(original_rng['numpy'])
        random.setstate(original_rng['python'])

    reset(torch.float64)
    reference_batch = {n: value.double() for n, value in batch.items()}
    reference_loss = module.adjoint_matching_log_variance_step(reference_batch)['loss']
    reference_loss.backward()
    ref = {n: p.grad.detach().double().clone() for n,p in module.controller.named_parameters()}
    assert all(torch.isfinite(g).all() for g in ref.values()), 'FP64 reference gradient nonfinite'
    reference_norm = torch.sqrt(sum(g.square().sum() for g in ref.values()))
    factor = (cfg.gradient_clip_val/(reference_norm+1e-6)).clamp(max=1.)
    ref = {n: g*factor for n,g in ref.items()}
    reset(torch.float32)
    initial = {n:p.detach().clone() for n,p in module.controller.named_parameters()}
    # Original momentum tensors stay FP32; reference arithmetic is independent FP64.
    expected_parameters = manual_adam(initial, ref, module.controller_optimizer.state_dict())
    loss = module.adjoint_matching_log_variance_step(batch)['loss']
    scale = power_of_two_scale(loss)
    module.fabric.backward(loss * scale)
    norm = unscale_and_clip(module.controller.parameters(), scale, cfg.gradient_clip_val)
    actual_gradients = {n:p.grad.detach().clone() for n,p in module.controller.named_parameters()}
    gradient_error = compare(actual_gradients, ref)
    module.controller_optimizer.step()
    actual_parameters = {n:p.detach().clone() for n,p in module.controller.named_parameters()}
    parameter_error = compare(actual_parameters, expected_parameters)
    update_error = compare({n:actual_parameters[n].double()-initial[n] for n in initial},
                           {n:expected_parameters[n]-initial[n] for n in initial})
    expected_counters = sorted(set(int(s['step'])+1 for s in original_optimizer['state'].values())) or [1]
    actual_counters = sorted(set(int(s['step']) for s in module.controller_optimizer.state.values()))
    loss_relative = float((loss.double()-reference_loss.detach()).abs()/reference_loss.detach().abs().clamp_min(1e-300))
    row = dict(case=label,scale=scale,norm64=float(norm),reference_norm64=float(reference_norm),
               loss=float(loss),reference_loss=float(reference_loss.detach()),loss_relative_error=loss_relative,
               clipped_gradient_error=gradient_error,parameter_error=parameter_error,update_error=update_error,
               Adam_counters=actual_counters,expected_Adam_counters=expected_counters,
               restored_counters=payload['counters'],sampling_metadata_verified=True,
               frozen_RNG_and_buffers_restored=True,matmul_precision=torch.get_float32_matmul_precision(),
               native_comparison=None)
    if label == 'normal':
        reset(torch.float32)
        native = module.adjoint_matching_log_variance_step(batch)['loss']
        module.fabric.backward(native)
        clip_gradients_with_overflow_fallback(module.fabric,module.controller,module.controller_optimizer,cfg.gradient_clip_val)
        native_grads = {n:p.grad.detach().clone() for n,p in module.controller.named_parameters()}
        row['native_comparison'] = compare(actual_gradients, native_grads)
    gates = dict(loss=loss_relative<=tolerances['loss_relative'],
        clipped_gradient=gradient_error['relative_l2']<=tolerances['gradient_relative_l2'],
        parameter=parameter_error['maximum_absolute']<=tolerances['parameter_max_absolute'],
        update=update_error['relative_l2']<=tolerances['update_relative_l2'],
        Adam_counter=actual_counters==expected_counters,
        native_normal=label!='normal' or row['native_comparison']['relative_l2']<=tolerances['gradient_relative_l2'])
    assert module.potential.aldp_potential.openmm_configurations_evaluated == 0
    row.update(gates=gates,status='passed' if all(gates.values()) else 'failed',OpenMM=0)
    return row


def main():
    from run_aldp_chart_campaign import verify_source, sha256, write_json, receipt
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    campaign = parser.parse_args().campaign
    plan = json.loads((campaign/'plan.json').read_text())
    source = verify_source()
    assert source == plan['source_manifest_sha256']
    for name, expected in plan['frozen_inputs'].items():
        assert sha256(name)==expected,name
    assert plan['tolerances'] == json.loads(Path(plan['tolerance_origin']).read_text())['tolerances']
    assert plan['expected_OpenMM'] == plan['OpenMM_cap'] == 0
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    out = campaign/'diagnostic'
    assert not out.exists()
    out.mkdir()
    witness = torch.load(plan['witness'],map_location='cpu',weights_only=False)
    normal = torch.load(plan['normal_snapshot'],map_location='cpu',weights_only=False)
    storage = normal['buffers']['adjoint_buffer']['storage']
    normal_batch = {n:torch.stack(v[:64]) for n,v in storage.items()}
    rows = []
    for label,path,payload,batch in [('normal',plan['normal_config'],normal,normal_batch),
                                    ('witness',plan['witness_config'],witness,witness['batch'])]:
        rows.append(run_case(label,path,payload,batch,out,plan['tolerances']))
        write_json(out/'gate.json',safe(dict(rows=rows,status='partial',OpenMM=0)))
    passed = all(r['status']=='passed' for r in rows)
    write_json(out/'gate.json',safe(dict(rows=rows,status='passed' if passed else 'failed',
        source_sha256=source,plan_sha256=sha256(campaign/'plan.json'),OpenMM=0,
        optimizer_updates=2,optimizer_updates_discarded=True,tolerances=plan['tolerances'],
        matmul_precision=torch.get_float32_matmul_precision(),
        production_changed=False,full_training_submitted=False,
        limits='Two frozen batches/one resumed Adam step each; diagnostic FP64 reference changes arithmetic precision only, no quality/admission claim')))
    assert passed, 'Original clipped-gradient/Adam tolerances not met'
    receipt(out,dict(stage='scaled_backward_gate',seed=0),dict(source_sha256=source,plan_sha256=sha256(campaign/'plan.json')))
    print('scaled_backward_gate_passed=true',flush=True)


if __name__=='__main__':
    main()
