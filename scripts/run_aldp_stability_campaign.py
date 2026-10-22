#!/usr/bin/env python3
"""FC/BC and dual diagnostics through the shared ALDP training pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from run_aldp_chart_campaign import receipt, sha256, train_case, verify_source, write_json
from run_aldp_mechanism_campaign import MechanismObserver, configuration as mechanism_configuration


def configuration(plan, campaign, stage, candidate, seed):
    from omegaconf import open_dict
    cfg, output = mechanism_configuration(plan, campaign, stage, candidate, seed)
    variant = plan['candidates'][candidate]
    smoke = stage == 'mechanism_smoke'
    with open_dict(cfg):
        cfg.initial_adjoint_matching_epochs = variant.get(
            'smoke_initial_adjoint_matching_epochs' if smoke else 'initial_adjoint_matching_epochs')
        for key in ('per_time_anchor_target', 'bc_loss_weight', 'fc_loss_mode'):
            if key in variant:
                cfg[key] = variant[key]
        dual = variant.get('smoke_dual_samples' if smoke else 'dual_samples')
        if dual is not None:
            cfg.num_samples_for_dual_loss = dual
    return cfg, output


def expected_cost(plan, stage, candidate):
    variant = plan['candidates'][candidate]
    smoke = stage == 'mechanism_smoke'
    b = plan['budgets']['smoke' if smoke else 'geometry']
    loops = b['outer_loops'] if smoke else variant['outer_loops']
    first = variant.get('smoke_initial_adjoint_matching_epochs' if smoke
                        else 'initial_adjoint_matching_epochs')
    first = b['epochs_per_phase'] if first is None else first
    fc_epochs = first+(loops-1)*b['epochs_per_phase']
    dual = variant.get('smoke_dual_samples' if smoke else 'dual_samples', b['dual_samples'])
    online_calls = 1+(fc_epochs+loops*b['epochs_per_phase'])//b['epochs_per_phase']
    mode = variant.get('fc_loss_mode', 'cross_entropy')
    if mode not in ('cross_entropy', 'log_variance'):
        raise ValueError('No registered OpenMM accounting contract for FC mode: '+mode)
    # Two fresh batches per FC epoch. CE also evaluates the terminal force;
    # log-variance reuses exact path densities and makes no terminal-force call.
    fc_calls = 6 if mode == 'cross_entropy' else 2
    training = fc_epochs*fc_calls*b['batch_size']+loops*dual[0]*dual[1]+online_calls*b['online_samples']
    count, phase_count = b['final_samples'], b['phase_samples']
    diagnostic = 3*count+3*min(128, count)+2*loops*(2*phase_count+min(128, phase_count))
    final = 2*count
    return dict(training_and_online_openmm_configurations=training,
        final_evaluation_openmm_configurations=final, diagnostic_openmm_configurations=diagnostic,
        total_openmm_configurations=training+final+diagnostic,
        optimizer_steps=(fc_epochs+loops*b['epochs_per_phase'])*b['steps_per_epoch'],
        u_optimizer_steps=fc_epochs*b['steps_per_epoch'],
        h_optimizer_steps=loops*b['epochs_per_phase']*b['steps_per_epoch'])


def dual_diagnostics(log_weights, gamma, v0, bound, bootstrap_count=32):
    """Finite-sample dual sensitivity, with no new paths or global RNG changes."""
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Dual paths must be a finite, nonempty group-by-path matrix')

    def solve(data):
        def objective(lm):
            term = logsumexp(-data / (1 + lm), axis=1) - math.log(data.shape[1])
            return (1 + lm) * float(np.mean(term)) + lm * bound
        result = minimize_scalar(objective, bounds=(0., 1e6), method='bounded')
        return 1 / (1 + result.x), result, objective

    estimate, result, objective = solve(values)
    rng = np.random.default_rng(21455)
    draws = [solve(values[rng.integers(len(values), size=len(values))])[0]
             for _ in range(bootstrap_count)]
    conditional = -float(gamma) * values
    probabilities = np.exp(conditional - logsumexp(conditional, axis=1, keepdims=True))
    ess = 1 / np.square(probabilities).sum(axis=1) / values.shape[1]
    return dict(groups=values.shape[0], paths_per_group=values.shape[1],
        gamma=float(gamma), v0=float(v0), independent_solve_gamma=float(estimate),
        optimizer_success=bool(result.success), lambda_estimate=float(result.x),
        lambda_near_upper_bound=bool(result.x > .99e6),
        objective_at_solution=float(objective(result.x)),
        bootstrap_count=bootstrap_count, bootstrap_gamma_q05_q50_q95=np.quantile(
            draws, [.05, .5, .95]).tolist(),
        conditional_weight_ess_fraction_q05_q50_q95=np.quantile(ess, [.05, .5, .95]).tolist(),
        raw_log_weights_q01_q50_q99=np.quantile(values, [.01, .5, .99]).tolist(),
        interpretation='conditional finite-bank sensitivity, not unbiasedness or convergence',
        additional_openmm_configurations=0)


class StabilityObserver(MechanismObserver):
    def __init__(self, bundle, output, count, phase_count, chart_key, bootstrap_count=32):
        super().__init__(bundle, output, count, False, chart_key)
        self.phase_count, self.bootstrap_count = phase_count, bootstrap_count
        self.report.update(phase_end=[], dual=[], diagnostic_schema='aldp_stability_v1',
                           phase_sample_count=phase_count, dual_bootstrap_count=bootstrap_count)

    def _probes(self, module):
        from diagnose_aldp_geometry import energy_summary
        results = {}
        for time in (0., .5, 1.):
            t = torch.full((len(self.probe),), time, device=self.probe.device)
            u = module.controller(t, self.probe).double()
            h = module.corrector(t, self.probe).double()
            results[str(time)] = {label: energy_summary(value.norm(dim=-1).cpu().numpy())
                                  for label, value in [('u', u), ('h', h), ('h_minus_u', h-u)]}
        return results

    def _instrument(self, module):
        original_collect = module.collect_log_weights_for_optimization
        original_solve = module.optimize_gamma

        def collect():
            data = original_collect()
            self.dual_paths = np.asarray(data).copy()
            return data

        def solve():
            gamma, v0 = original_solve()
            module.campaign_observe('dual', dict(gamma=gamma, v0=v0,
                paths=self.dual_paths, outer_loop=int(module.outer_step)+1))
            return gamma, v0

        module.collect_log_weights_for_optimization = collect
        module.optimize_gamma = solve
        for name, phase in [('inner_loop_u', 'u'), ('inner_loop_h', 'h')]:
            original = getattr(module, name)

            def wrapped(original=original, phase=phase):
                original()
                module.campaign_observe('phase_end', dict(phase=phase,
                    outer_loop=int(module.outer_step)+1, epoch=int(module.current_epoch),
                    global_step=int(module.global_step), phase_epoch=int(module.phase_epoch)))
            setattr(module, name, wrapped)

    @torch.no_grad()
    def __call__(self, module, event, payload=None):
        from run_aldp_mechanism_campaign import diagnostic_rng
        with diagnostic_rng(module.fabric.device):
            if event == 'dual':
                label = f"dual_outer{payload['outer_loop']:03d}"
                np.save(self.output / f'{label}_logweights.npy', payload['paths'])
                diagnostic = dual_diagnostics(payload['paths'], payload['gamma'], payload['v0'],
                    module.trust_region_bound, self.bootstrap_count)
                diagnostic.update(outer_loop=payload['outer_loop'], global_step=int(module.global_step),
                                  file=f'{label}_logweights.npy')
                self.report['dual'].append(diagnostic)
                return
            if event == 'phase_end':
                label = f"phase_outer{payload['outer_loop']:03d}_{payload['phase']}"
                # MechanismObserver.policy uses self.count only for sampling; restore it
                # after a small phase probe so final/source diagnostics retain1024 draws.
                original_count = self.count
                try:
                    self.count = self.phase_count
                    state = self.policy(module, label)
                finally:
                    self.count = original_count
                state.update(payload, label=label, gamma=float(module.gamma),
                    gamma_h=float(module.gamma_h), probes=self._probes(module),
                    metrics_role='small_phase_diagnosis_only_not_final_quality_gate')
                self.report['phase_end'].append(state)
                return
            result = super().__call__(module, event, payload)
            if event == 'initial':
                self.report['training_contract'] = dict(
                    initial_adjoint_matching_epochs=module.initial_adjoint_matching_epochs,
                    regular_adjoint_matching_epochs=module.adjoint_matching_epochs,
                    corrector_matching_epochs=module.corrector_matching_epochs,
                    per_time_anchor_target=module.per_time_anchor_target,
                    bc_loss_weight=module.bc_loss_weight,
                    fc_loss_mode=module.fc_loss_mode,
                    dual_samples=list(module.cfg.num_samples_for_dual_loss),
                    outer_loops=module.max_outer_loops,
                    initial_probes=self._probes(module))
                self._instrument(module)
            if event == 'online':
                self.report['online'][-1]['probes'] = self._probes(module)
            return result


def verify_stability_report(report, plan, stage, candidate, files):
    variant = plan['candidates'][candidate]
    smoke = stage == 'mechanism_smoke'
    budget = plan['budgets']['smoke' if smoke else 'geometry']
    loops = budget['outer_loops'] if smoke else variant['outer_loops']
    first = variant.get('smoke_initial_adjoint_matching_epochs' if smoke
                        else 'initial_adjoint_matching_epochs', budget['epochs_per_phase'])
    if first is None:
        first = budget['epochs_per_phase']
    expected = [(outer, phase) for outer in range(1, loops+1) for phase in ('u', 'h')]
    diagnostic = report['diagnostics']
    contract = diagnostic.get('training_contract', {})
    dual_samples = variant.get('smoke_dual_samples' if smoke else 'dual_samples', budget['dual_samples'])
    if (contract.get('initial_adjoint_matching_epochs') != first
            or contract.get('regular_adjoint_matching_epochs') != budget['epochs_per_phase']
            or contract.get('corrector_matching_epochs') != budget['epochs_per_phase']
            or contract.get('outer_loops') != loops
            or contract.get('dual_samples') != dual_samples
            or contract.get('per_time_anchor_target') != variant.get('per_time_anchor_target', False)
            or contract.get('bc_loss_weight') != variant.get('bc_loss_weight', 'legacy')):
        raise RuntimeError('Resolved FC/BC/dual contract differs from plan')
    # Historical stability receipts predate this field and used CE exclusively.
    if contract.get('fc_loss_mode', 'cross_entropy') != variant.get('fc_loss_mode', 'cross_entropy'):
        raise RuntimeError('Resolved FC loss differs from plan')
    phases = diagnostic.get('phase_end', [])
    if [(r['outer_loop'], r['phase']) for r in phases] != expected:
        raise RuntimeError('Missing/reordered FC/BC phase diagnostics')
    phase_count = budget['phase_samples']
    epoch = 0
    for item in phases:
        phase_epochs = first if item['outer_loop'] == 1 and item['phase'] == 'u' else budget['epochs_per_phase']
        epoch += phase_epochs
        if item['epoch'] != epoch or item['global_step'] != epoch*budget['steps_per_epoch']:
            raise RuntimeError('Phase diagnostic has wrong epoch/update count')
        if item['phase_epoch'] != phase_epochs or item['physical_energy']['count'] != phase_count:
            raise RuntimeError('Incomplete phase or wrong diagnosis sample count')
        if (not math.isfinite(item['raw_path_ess_fraction'])
                or not 0 <= item['raw_path_ess_fraction'] <= 1
                or not math.isfinite(item['raw_log_path_ratio_variance'])):
            raise RuntimeError('Nonfinite/invalid phase path-ratio diagnosis')
        for suffix in ('cartesian', 'physical_energy'):
            if f"{item['label']}_{suffix}.npy" not in files:
                raise RuntimeError('Phase array absent from receipt')
    dual = diagnostic.get('dual', [])
    if [r['outer_loop'] for r in dual] != list(range(1, loops+1)):
        raise RuntimeError('Missing dual diagnostics')
    for item in dual:
        if item['file'] not in files or not item['optimizer_success']:
            raise RuntimeError('Dual solver/receipt verification failed')
        if not all(math.isfinite(x) for x in item['bootstrap_gamma_q05_q50_q95']):
            raise RuntimeError('Nonfinite bootstrap gamma')
    if not diagnostic.get('diagnostic_rng_isolated'):
        raise RuntimeError('Diagnostic RNG isolation missing')
    for key, value in expected_cost(plan, stage, candidate).items():
        if key.endswith('configurations') and report['cost'][key] != value:
            raise RuntimeError('Stability cost boundary differs from plan: '+key)


def run_case(plan, campaign, stage, candidate, seed, source_sha):
    from tr_asbs.process.chart_source import load_calibration_bundle
    cfg, output = configuration(plan, campaign, stage, candidate, seed)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Retain existing attempt; use a fresh output')
    bundle = load_calibration_bundle(cfg.potential.calibration_bundle_path)
    budget = plan['budgets']['smoke' if stage == 'mechanism_smoke' else 'geometry']
    observer = StabilityObserver(bundle, output, cfg.eval_samples_final,
        budget['phase_samples'], cfg.potential.chart_key, plan['stability_diagnostics']['bootstrap_count'])
    return train_case(dict(plan, max_openmm_per_geometry_run=plan['candidates'][candidate]['openmm_cap']),
        campaign, stage, cfg.potential.chart_key, seed, source_sha, config_and_output=(cfg, output),
        observer=observer, provenance=dict(candidate=candidate, plan_sha256=sha256(campaign/'plan.json'),
            upstream_source_sha256=plan['upstream_source_sha256'],
            diagnostic_report_sha256=plan['diagnostic_report_sha256'], cached_baseline=plan['cached_baseline']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'mechanism_pilot'), required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign/'plan.json').read_text())
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the configured remote GPU')
    source_sha = verify_source()
    if args.stage == 'preflight':
        checks = []
        for candidate in plan['candidates']:
            output = run_case(plan, campaign, 'mechanism_smoke', candidate, 0, source_sha)
            report = json.loads((output/'result.json').read_text())
            document = json.loads((output/'receipt.json').read_text())
            verify_stability_report(report, plan, 'mechanism_smoke', candidate, document['files'])
            checks.append(dict(candidate=candidate, result_sha256=sha256(output/'result.json'),
                               receipt_sha256=sha256(output/'receipt.json')))
        write_json(campaign/'preflight/preflight_result.json',
                   dict(status='passed', checks=checks, source_sha256=source_sha))
        receipt(campaign/'preflight', dict(stage='preflight'),
                dict(source_sha256=source_sha, plan_sha256=sha256(campaign/'plan.json'),
                     calibration_sha256=plan['calibration_sha256']))
    else:
        run_case(plan, campaign, args.stage, args.candidate, args.seed, source_sha)
    print('postflight_verified=true', flush=True)


if __name__ == '__main__':
    main()
