#!/usr/bin/env python3
"""Normal-duration u/h learning curves in the shared physical ALDP pipeline.

This runner does not select a smaller duration from training loss alone. Fixed
prefixes screen candidates; a shortened u/h pair needs a separate continuation
comparison, since shortening u changes h's data and the next outer update.
"""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from run_aldp_chart_campaign import receipt, sha256, train_case, verify_source, write_json
from run_aldp_jax_inner_preflight import check_files, configuration as probe_configuration
from run_aldp_stability_campaign import StabilityObserver


def configuration(plan, campaign):
    from omegaconf import open_dict
    cfg, _ = probe_configuration(plan, campaign)
    contract = plan['curve_contract']
    if (contract['u_epochs'], contract['h_epochs'], contract['steps_per_epoch'],
            contract['outer_loops']) != (2000, 1000, 100, 1):
        raise ValueError('The first curve reference must retain normal u/h duration and outer1')
    output = campaign/'normal_inner_curve/seed0'
    with open_dict(cfg):
        cfg.name, cfg.root = output.name, str(output)
        cfg.fabric.loggers.save_dir = str(output)
        cfg.checkpoint_directory = str(output/'checkpoints')
        cfg.sample_directory = str(output/'samples')
        cfg.log_detail_interval_epoch = contract['online_interval_epochs']
        cfg.eval_samples_final = contract['quality_samples']
    return cfg, output


def expected_cost(cfg, contract):
    """Asymmetric normal phases; BC buffers/validation use no OpenMM calls."""
    u, h = cfg.adjoint_matching_epochs, cfg.corrector_matching_epochs
    count, phase = contract['quality_samples'], contract['phase_samples']
    prefixes = sum(len(contract['milestones'][p]) for p in ('u', 'h'))
    online_calls = 1+(u+h)//cfg.log_detail_interval_epoch
    train = (u*cfg.buffer_samples_per_epoch
        +cfg.num_samples_for_dual_loss[0]*cfg.num_samples_for_dual_loss[1]
        +online_calls*cfg.eval_samples)
    diagnostics = (3*count+2*min(128, count)  # source + untrained policy
        +2*(2*phase+min(128, phase))         # shared phase-end probes
        +prefixes*(2*count+min(128, count))  # selected completed prefixes
        +contract['validation_samples']     # frozen u trajectories
        +min(128, count))                   # final gradient diagnosis
    return dict(training_and_online_openmm_configurations=train,
        diagnostic_openmm_configurations=diagnostics,
        final_evaluation_openmm_configurations=2*count,
        total_openmm_configurations=train+diagnostics+2*count,
        u_optimizer_steps=u*cfg.steps_per_epoch, h_optimizer_steps=h*cfg.steps_per_epoch,
        optimizer_steps=(u+h)*cfg.steps_per_epoch)


def validate_contract(cfg, contract):
    if (cfg.train_batch_size, cfg.inference_batch_size, cfg.integrator.timesteps.steps,
            list(cfg.num_samples_for_dual_loss)) != (64, 64, 200, [32, 8]):
        raise ValueError('Inherited B64/inference64/N200/dual32x8 contract changed')
    for phase, end in (('u', 2000), ('h', 1000)):
        grid = contract['milestones'][phase]
        if sorted(set(grid)) != grid or not grid or min(grid) <= 0 or grid[-1] != end:
            raise ValueError('Milestones must be sorted unique positive epochs including phase end')
    if contract['selection_bank'] != 'a' or contract['test_bank'] != 'closed':
        raise ValueError('Learning-curve selection must keep the test bank closed')


class InnerCurveObserver(StabilityObserver):
    def __init__(self, bundle, output, cfg, plan, source_sha):
        contract = plan['curve_contract']
        super().__init__(bundle, output, contract['quality_samples'],
            contract['phase_samples'], cfg.potential.chart_key)
        self.contract, self.cfg = contract, cfg
        self.provenance = dict(source_sha256=source_sha,
            plan_sha256=sha256(output.parent.parent/'plan.json'),
            calibration_sha256=plan['frozen_inputs'][cfg.potential.calibration_bundle_path],
            config_sha256=sha256(output/'config.yaml'))
        self.report.update(inner_boundaries=[], curve_contract=contract,
            reference_initialized_from='fresh seed0, not the trained backend-probe checkpoint',
            scientific_status='screening_only_not_minimum_or_geometry_admission')
        self.started = time.perf_counter()
        self.validation = {}

    @torch.no_grad()
    def policy(self, module, label):
        from tr_asbs.utils.eval_logweight import compute_raw_path_ess
        torch.manual_seed(12345)
        x0 = module.source.sample((self.count,))
        result = module.integrator.run_with_logweights(module.integrator,
            module.source.log_prob, module.potential.log_prob, x0,
            center_every_step=False, return_trajectory=False, progress_bar=False)
        y = result['data_1']
        cartesian = module.potential.native_to_cartesian(y)
        energy = module.potential.eval_x(cartesian).detach()
        for name, value in (('cartesian', cartesian), ('physical_energy', energy), ('native', y)):
            np.save(self.output/f'{label}_{name}.npy', value.cpu().numpy())
        state = self.state(module, y, cartesian, energy)
        z, _ = module.potential.chart(y)
        indices = self.bundle['phi_psi_native_indices']
        delta = (z[:,indices]*module.potential.chart.scales[indices]
                 -module.potential.chart.offsets[indices]).cpu().numpy()
        state.update(raw_path_ess_fraction=float(compute_raw_path_ess(result['log_Ph_Pref'])),
            raw_log_path_ratio_variance=float(result['log_Ph_Pref'].double().var(unbiased=False)),
            achieved_forward_kl=float(result['log_Pref_Pu_det'].double().mean()),
            near_cut_fraction_phi_psi=np.mean(np.pi-np.abs(delta)<.15, axis=0).tolist())
        return state

    @torch.no_grad()
    def _validation_start(self, module, phase):
        count = self.contract['validation_samples']
        torch.manual_seed(24680 if phase == 'u' else 24681)
        x0 = module.source.sample((count,))
        if phase == 'u':
            data = module.prev_integrator.run_with_logweights(module.prev_integrator,
                module.source.log_prob, module.potential.log_prob, x0,
                center_every_step=False, return_trajectory=True, progress_bar=False)
            batch = dict(data_traj=torch.stack(data['trajectory'], dim=1).detach(),
                log_Ph_Pref_raw=data['log_Ph_Pref'].detach())
        else:
            from tr_asbs.experiment.jax_inner import prepare_bc_batch
            x1 = module.integrator.run(initial_data=x0, center_every_step=False,
                return_trajectory=False, progress_bar=False)
            prepared = prepare_bc_batch(module, dict(data_0=x0, data_1=x1))
            batch = {name: torch.utils.dlpack.from_dlpack(value).to(module.fabric.device).clone()
                     for name, value in prepared.items()}
        if any(not torch.isfinite(value).all() for value in batch.values()):
            raise RuntimeError('Nonfinite frozen held-out '+phase+' validation batch')
        self.validation[phase] = batch
        torch.save({name:value.cpu() for name,value in batch.items()},
                   self.output/f'heldout_{phase}.pt')

    @torch.no_grad()
    def _validation_loss(self, module, phase):
        batch = self.validation[phase]
        if phase == 'u':
            values = module.adjoint_matching_log_variance_step(batch)
            return dict(heldout_loss=float(values['loss']),
                heldout_log_weight_mean=float(values['log_weight_mean']),
                sample_count=len(batch['data_traj']), gamma=float(module.gamma),
                bank='fixed frozen-policy paths, independent of training buffer')
        size = len(batch['time'])
        values = module.corrector(torch.cat((batch['time'], torch.ones_like(batch['time']))),
            torch.cat((batch['data_t'], batch['data_1'])))
        loss_t = ((values[:size]-batch['score_t']).square().mean(-1)
                  /batch['coef_t'].reshape(-1)).mean()
        loss_1 = (values[size:]-batch['score_1']).square().mean()
        return dict(heldout_loss=float(loss_t+loss_1), loss_t=float(loss_t),
            loss_1=float(loss_1), sample_count=size,
            bank='fixed endpoints from completed u and fixed bridge draws')

    @torch.no_grad()
    def __call__(self, module, event, payload=None):
        if event != 'inner_boundary':
            result = super().__call__(module, event, payload)
            if event == 'final':
                self.report['backend_adam_update_counts'] = {
                    phase:int(state.count) for phase,state in module.jax_inner_backend.states.items()}
                return self.report
            return result
        # The shared caller checks RNG immediately after this callback. The
        # boundary hook's outer fork exits later, so isolation here is also
        # required before returning to that caller.
        from tr_asbs.experiment.inner_observation import isolated_inner_diagnostic
        with isolated_inner_diagnostic(module):
            return self._observe_boundary(module, payload)

    @torch.no_grad()
    def _observe_boundary(self, module, payload):
        from tr_asbs.experiment.inner_observation import save_inner_snapshot
        # The boundary hook already isolates Torch/CUDA/NumPy/Python RNG and
        # supplies the pre-diagnostic training RNG for the exact snapshot.
        phase = payload['phase']
        if payload['phase_epoch'] == 0:
            self._validation_start(module, phase)
        label = f"outer{payload['outer_loop']:03d}_{phase}_{payload['phase_epoch']:04d}"
        save_inner_snapshot(module, self.output/'inner_snapshots'/f'{label}.pt', self.provenance)
        validation = self._validation_loss(module, phase)
        if not all(math.isfinite(value) for value in validation.values()
                   if isinstance(value, (int, float))):
            raise RuntimeError('Nonfinite held-out inner objective')
        item = dict(payload, label=label, validation=validation,
            snapshot=f'inner_snapshots/{label}.pt')
        if payload['phase_epoch']:
            item['policy'] = self.policy(module, label)
            item['metrics_role'] = ('u prefix screening with h frozen; shortening u changes subsequent h data'
                if phase == 'u' else 'h adequacy from BC/RN/ESS; forward endpoints fixed at completed u')
        item.update(wall_seconds=time.perf_counter()-self.started,
            openmm_configurations=module.potential.aldp_potential.openmm_configurations_evaluated)
        self.report['inner_boundaries'].append(item)
        write_json(self.output/'learning_curve_progress.json', dict(status='running',
            scientific_status='not_evaluated', boundaries=self.report['inner_boundaries'],
            current_openmm_configurations=module.potential.aldp_potential.openmm_configurations_evaluated))

def setup_backend(module, observer):
    from tr_asbs.experiment.inner_observation import observe_inner_boundaries
    from tr_asbs.experiment.jax_inner import JaxInnerBackend
    backend = JaxInnerBackend(module).attach()
    original = module.training_step

    def checked(batch, task):
        success = original(batch, task)
        if not success:
            write_json(observer.output/'numerical_failure.json', dict(task=task,
                epoch=int(module.current_epoch), global_step=int(module.global_step),
                phase=module.phase, phase_epoch=int(module.phase_epoch),
                openmm_configurations=module.potential.aldp_potential.openmm_configurations_evaluated,
                exact_mid_epoch_resume_available=False))
            raise RuntimeError('Nonfinite JAX neural update; preserve the last complete boundary')
        return success
    module.training_step = checked
    observe_inner_boundaries(module,
        lambda payload:module.campaign_observe('inner_boundary', payload),
        observer.contract['milestones'])
    return backend


def run(plan, campaign, source_sha):
    from tr_asbs.process.chart_source import load_calibration_bundle
    cfg, output = configuration(plan, campaign)
    validate_contract(cfg, plan['curve_contract'])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Preserve prior attempt; use a fresh snapshot/output for any retry')
    # Observer provenance reads the exact configuration saved by the shared runner.
    output.mkdir(parents=True)
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    observer = InnerCurveObserver(load_calibration_bundle(cfg.potential.calibration_bundle_path),
        output, cfg, plan, source_sha)
    costs = expected_cost(cfg, plan['curve_contract'])
    result_dir = train_case(dict(plan, max_openmm_per_geometry_run=plan['openmm_cap']),
        campaign, 'normal_inner_curve', cfg.potential.chart_key, 0, source_sha,
        config_and_output=(cfg, output), observer=observer, provenance=observer.provenance,
        module_setup=lambda module:setup_backend(module, observer))
    result = json.loads((result_dir/'result.json').read_text())
    if result['global_step'] != costs['optimizer_steps']:
        raise RuntimeError('Normal duration reference did not complete all300000 neural updates')
    for name, expected in costs.items():
        if name.endswith('configurations') and result['cost'][name] != expected:
            raise RuntimeError('Normal curve cost contract mismatch: '+name)
    counts = result['diagnostics']['backend_adam_update_counts']
    if counts != {'u':costs['u_optimizer_steps'], 'h':costs['h_optimizer_steps']}:
        raise RuntimeError('JAX Adam phase counts disagree with normal u/h contract')
    result.update(curve_expected_cost=costs, minimum_inner_duration='not_established',
        geometry_admission='not_evaluated', prefix_only_selection_forbidden=True)
    write_json(result_dir/'result.json', result)
    receipt(result_dir, dict(stage='normal_inner_curve', chart=cfg.potential.chart_key, seed=0),
        dict(source_sha256=source_sha, provenance=observer.provenance))
    return result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign/'plan.json').read_text())
    source_sha = verify_source()
    if source_sha != plan['source_manifest_sha256']:
        raise RuntimeError('Normal curve source manifest differs from plan')
    check_files(plan['frozen_inputs'])
    proof = json.loads(Path(plan['backend_verification_path']).read_text())
    if proof.get('status') != 'backend_verified' or proof.get('receipt_files_verified', 0) < 1:
        raise RuntimeError('Verified GPU preflight/accounting/receipt proof required')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Normal ALDP curve requires the configured remote GPU')
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    torch.set_float32_matmul_precision('highest')
    try:
        run(plan, campaign, source_sha)
        print('postflight_verified=true', flush=True)
    except Exception as error:
        write_json(campaign/'normal_inner_curve/seed0/failure.json', dict(
            status='failed', error=repr(error), traceback=traceback.format_exc(),
            source_sha256=source_sha, scientific_status='not_evaluated'))
        raise


if __name__ == '__main__':
    main()
