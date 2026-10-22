#!/usr/bin/env python3
"""Staged loss/solver/noise/reference comparisons in the shared ALDP pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from run_aldp_chart_campaign import receipt, sha256, train_case, verify_source, write_json
from run_aldp_stability_campaign import (StabilityObserver, configuration as stability_configuration,
    expected_cost, verify_stability_report)

SOLVERS = dict(euler='EulerMaruyamaIntegrator', legacy='ExponentialIntegrator',
    diffusion_euler='DiffusionTimeEulerIntegrator', linear_exponential='LinearExponentialIntegrator')
FAMILIES = ('loss_bridge', 'solver', 'noise', 'reference')


def validate_plan(plan):
    family = plan['study_family']
    variants = list(plan['candidates'].values())
    if family not in FAMILIES or not variants:
        raise ValueError('Registered nonempty component family required')
    defaults = dict(chart_key='current', source_scale=1., terminal_gradient_clip=1000.,
        initial_adjoint_matching_epochs=None, smoke_initial_adjoint_matching_epochs=None,
        per_time_anchor_target=False, bc_loss_weight='legacy',
        dual_samples=plan['budgets']['geometry']['dual_samples'],
        smoke_dual_samples=plan['budgets']['smoke']['dual_samples'])
    keys = (*defaults, 'outer_loops', 'openmm_cap')
    def nuisance(value):
        return {key: value.get(key, defaults.get(key)) for key in keys}
    if any(nuisance(v) != nuisance(variants[0]) for v in variants[1:]):
        raise ValueError('Component comparison changes training/source/chart nuisance settings')
    for variant in variants:
        if variant.get('fc_loss_mode') != 'log_variance':
            raise ValueError('Component pilots require the common log-variance loss')
        if family in ('loss_bridge', 'solver') and variant.get('noise_shape', 'geometric') != 'geometric':
            raise ValueError('Loss/solver comparison fixes geometric noise')
        if family == 'loss_bridge' and variant.get('solver', 'legacy') != 'legacy':
            raise ValueError('Loss bridge fixes the cached legacy solver')
        if family != 'reference' and 'reference_kind' in variant:
            raise ValueError('Reference changes belong in their own comparison')
    if family == 'noise' and len({v.get('solver', 'legacy') for v in variants}) != 1:
        raise ValueError('Noise comparison fixes one solver')
    if family == 'reference':
        rate = plan['component_diagnostics']['ou_rate']
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('OU comparison rate must be finite and positive')


def configuration(plan, campaign, stage, candidate, seed):
    from omegaconf import open_dict
    validate_plan(plan)
    cfg, output = stability_configuration(plan, campaign, stage, candidate, seed)
    variant = plan['candidates'][candidate]
    family = plan['study_family']
    solver = variant.get('solver', 'legacy')
    if solver not in SOLVERS:
        raise ValueError('Unregistered component solver')
    variance = float(plan['component_diagnostics']['integrated_diffusion_variance'])
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError('Invalid integrated diffusion variance')
    with open_dict(cfg):
        cfg.integrator._target_ = 'tr_asbs.process.integrator.'+SOLVERS[solver]
        cfg.integrator.grid = 'uniform_t'
        # Noise knots remain fixed even for N4 solver smoke tests.
        shape = variant.get('noise_shape', 'geometric')
        if shape not in ('geometric', 'constant', 'cosine'):
            raise ValueError('Unregistered noise shape')
        if shape == 'constant':
            cfg.base_sde = dict(_target_='tr_asbs.process.reference.BrownianReference',
                noise_schedule=dict(_target_='tr_asbs.process.noise_schedule.AnalyticNoiseSchedule',
                    kind='constant', g_min=.001, g_max=.5, total_variance=variance))
        elif shape == 'cosine':
            cfg.base_sde = dict(_target_='tr_asbs.process.reference.BrownianReference',
                noise_schedule=dict(_target_='tr_asbs.process.noise_schedule.LegacyNoiseSchedule',
                    total_variance=variance, process=dict(_target_='tr_asbs.process.sde.PISCosineSDE',
                        sigma_min=.001, sigma_max=.5,
                        num_steps=plan['component_diagnostics']['noise_knots'])))
        # Geometric retains the existing VE object, exactly preserving the
        # cached solver control's reference implementation and coefficients.
        if family == 'reference':
            kind = variant.get('reference_kind')
            if kind not in ('brownian', 'ou') or solver != 'linear_exponential':
                raise ValueError('Reference comparison uses Brownian-rate0/OU and one exact linear solver; VP excluded')
            if shape != 'constant':
                raise ValueError('Reference comparison fixes constant diffusion')
            cfg.base_sde = dict(_target_='tr_asbs.process.reference.LinearGaussianReference',
                kind='ou', rate=0. if kind == 'brownian' else float(
                    plan['component_diagnostics']['ou_rate']), diffusion_scale=variance**.5)
        elif solver == 'linear_exponential':
            raise ValueError('Exact linear solver is reserved for the linear reference comparison')
        if solver == 'linear_exponential':
            cfg.integrator.control_evaluation = 'left'
    return cfg, output


class UnitControl(nn.Module):
    def forward(self, time, x):
        return torch.ones_like(x)


@torch.no_grad()
def kernel_contract(integrator):
    """Exact coefficients of the configured finite kernels; no paths or energy calls."""
    from tr_asbs.process.sde import ControlledFBSDE
    from tr_asbs.process.reference import LinearGaussianReference
    base = integrator.reference
    times = integrator.timesteps
    options = dict(timesteps=times, grid='uniform_t')
    if hasattr(integrator, 'control_evaluation'):
        options['control_evaluation'] = integrator.control_evaluation
    uncontrolled = type(integrator)(base, **options)
    unit = type(integrator)(ControlledFBSDE(base, UnitControl(), UnitControl()), **options)
    x = torch.ones(1, 1, device=times.device, dtype=torch.double)
    zero = torch.zeros_like(x)
    forward, reverse = [], []
    for t, tn in zip(times[:-1], times[1:]):
        dt = tn-t
        for output, time, backward in ((forward, t, False), (reverse, tn, True)):
            kernel = uncontrolled.transition(x, time, dt, backward=backward)
            control = unit.transition(zero, time, dt, backward=backward)
            values = [float(v.double().reshape(-1)[0]) for v in (
                kernel.mean, control.mean, kernel.variance)]
            if not all(math.isfinite(v) for v in values) or values[2] <= 0:
                raise RuntimeError('Nonfinite or singular component kernel')
            output.append(values)

    def compose(rows):
        phi, k, q = 1., 0., 0.
        for p, c, v in rows:
            phi, k, q = p*phi, p*k+c, p*p*q+v
        return dict(phi=phi, unit_frozen_control_k=k, conditional_noise_variance=q)

    total = float(base.diffsquare_integral(times.new_tensor(1.)).double())
    if isinstance(base, LinearGaussianReference):
        continuous = [float(v) for v in base.transition_coefficients(0., 1.)]
        continuous_reverse = [float(v) for v in base.transition_coefficients(0., 1., backward=True)]
    else:
        continuous = continuous_reverse = [1., total, total]
    return dict(schema='aldp_component_kernel_v1', solver=type(integrator).__name__,
        reference=type(base).__name__, grid=integrator.grid, timesteps=times.cpu().tolist(),
        integrated_diffusion_variance=total,
        forward_composed=compose(forward), backward_composed=compose(reversed(reverse)),
        continuous_reference_phi_k_q=continuous,
        continuous_reverse_phi_k_q=continuous_reverse,
        interval_forward_phi_k_q=forward, interval_reverse_phi_k_q=reverse,
        control_units='score', conditional_noise_only=True,
        source_note='mapped nonGaussian source; marginal terminal covariance additionally propagates source covariance',
        additional_openmm_configurations=0, additional_sampled_paths=0)


@torch.no_grad()
def constant_reference_equivalence(brownian, linear):
    """Audit cached constant Brownian versus rate0 linear kernels and BC scores."""
    from tr_asbs.process.reference import LinearGaussianReference
    from tr_asbs.process.integrator import LinearExponentialIntegrator
    right = linear.reference
    if (not isinstance(right, LinearGaussianReference) or right.kind != 'ou'
            or float(right.rate) != 0. or not isinstance(linear, LinearExponentialIntegrator)):
        raise ValueError('Require the zero-rate exact linear control')
    times = torch.tensor([.2, .5, .8], device=brownian.timesteps.device, dtype=torch.double)
    amplitudes = brownian.reference.diffusion(times)
    if not torch.allclose(amplitudes, amplitudes[0].expand_as(amplitudes), rtol=2e-6, atol=2e-8):
        raise ValueError('Cached reference control must have constant diffusion')
    left_contract, right_contract = kernel_contract(brownian), kernel_contract(linear)
    if left_contract['timesteps'] != right_contract['timesteps']:
        raise ValueError('Reference control grids differ')
    errors = {}
    for key in ('interval_forward_phi_k_q', 'interval_reverse_phi_k_q'):
        left = torch.tensor(left_contract[key], dtype=torch.double)
        other = torch.tensor(right_contract[key], dtype=torch.double)
        if not torch.allclose(left, other, rtol=2e-6, atol=2e-8):
            raise ValueError('Cached Brownian/rate0 finite kernels differ')
        errors[key] = float((left-other).abs().max())
    x0 = torch.arange(6, device=times.device, dtype=torch.double).reshape(3, 2)/3
    x1 = x0*.4-.2
    state = x0*.6+.3
    for name in ('cond_score_t0', 'cond_score_1t'):
        left = getattr(brownian.reference, name)(times, x0 if name=='cond_score_t0' else x1, state)
        other = getattr(right, name)(times, x0 if name=='cond_score_t0' else x1, state)
        if not torch.allclose(left, other, rtol=2e-6, atol=2e-8):
            raise ValueError('Cached Brownian/rate0 BC score contract differs')
        errors[name] = float((left-other).abs().max())
    v = brownian.reference.diffsquare_integral(times)
    total = brownian.reference.diffsquare_integral(times.new_tensor(1.))
    mean = x0+(v/total)[:, None]*(x1-x0)
    variance = v-v.square()/total
    p0, _, q0 = right.transition_coefficients(0., times)
    p1, _, q1 = right.transition_coefficients(times, 1.)
    q01 = p1.square()*q0+q1
    gain = q0*p1/q01
    other_mean = p0[:, None]*x0+gain[:, None]*(x1-p1[:, None]*p0[:, None]*x0)
    other_variance = q0-gain.square()*q01
    for name, left, other in (('bridge_mean', mean, other_mean), ('bridge_variance', variance, other_variance)):
        if not torch.allclose(left, other, rtol=2e-6, atol=2e-8):
            raise ValueError('Cached Brownian/rate0 BC bridge moments differ')
        errors[name] = float((left-other).abs().max())
    return dict(status='passed',absolute_errors=errors,relative_tolerance=2e-6,absolute_tolerance=2e-8,
        brownian_kernel=left_contract,linear_rate0_kernel=right_contract,
        additional_sampled_paths=0,additional_openmm_configurations=0,
        interpretation='coefficient and conditional-score equivalence; no bit-identical trained trajectory or Gaussian-source claim')


class ComponentObserver(StabilityObserver):
    @torch.no_grad()
    def __call__(self, module, event, payload=None):
        result = super().__call__(module, event, payload)
        if event == 'initial':
            from omegaconf import OmegaConf
            self.report['component_contract'] = dict(kernel=kernel_contract(module.integrator),
                sampling_metadata=module.sampling_metadata(), fc_loss_mode=module.fc_loss_mode,
                base_sde_config=OmegaConf.to_container(module.cfg.base_sde, resolve=True),
                integrator_config=OmegaConf.to_container(module.cfg.integrator, resolve=True),
                corrector_matching_mode=str(module.cfg.corrector_matching_mode),
                terminal_force_clip_used_by_fc=module.fc_loss_mode == 'cross_entropy',
                per_time_anchor_used_by_fc=module.fc_loss_mode == 'cross_entropy',
                bc_bridge='analytic reference bridge conditional on both endpoints',
                reference_quality='finite correlated MD banks; convergence not certified')
        return result


def verify_component_report(report, plan, stage, candidate):
    from omegaconf import OmegaConf
    roundtrip = report.get('native_cartesian_roundtrip_diagnostic', {})
    if (roundtrip.get('schema') != 'aldp_cartesian_roundtrip_v1'
            or roundtrip.get('samples_modified') is not False
            or roundtrip.get('samples_filtered') is not False
            or roundtrip.get('additional_openmm_configurations') != 0
            or roundtrip.get('error_nm') != report.get('native_cartesian_roundtrip_error')
            or roundtrip.get('status') not in (
                'passed', 'strict_inverse_rejected', 'nonfinite_reconstruction', 'cartesian_mismatch')):
        raise RuntimeError('Raw Cartesian roundtrip diagnostic incomplete')
    if stage == 'mechanism_smoke' and roundtrip['status'] != 'passed':
        raise RuntimeError('Component smoke Cartesian roundtrip failed')
    count = plan['budgets']['smoke' if stage == 'mechanism_smoke' else 'geometry']['final_samples']
    error = roundtrip['error_nm']
    if (roundtrip.get('sample_count') != count or roundtrip.get('tolerance_nm') != 2e-5
            or (roundtrip['status'] == 'passed' and (
                not isinstance(error, (int, float)) or not math.isfinite(error) or not 0 <= error <= 2e-5))
            or (roundtrip['status'] == 'cartesian_mismatch' and (
                not isinstance(error, (int, float)) or not math.isfinite(error) or error <= 2e-5))
            or (roundtrip['status'] in ('strict_inverse_rejected', 'nonfinite_reconstruction') and error is not None)
            or (roundtrip['status'] == 'strict_inverse_rejected' and not roundtrip.get('inverse_error'))):
        raise RuntimeError('Cartesian roundtrip status/error/count mismatch')
    variant = plan['candidates'][candidate]
    contract = report['diagnostics'].get('component_contract', {})
    if (contract.get('fc_loss_mode') != 'log_variance'
            or contract.get('corrector_matching_mode') != 'full'
            or contract.get('terminal_force_clip_used_by_fc') is not False):
        raise RuntimeError('Common component-loss contract incomplete')
    if contract.get('sampling_metadata') != report['sampling_metadata']:
        raise RuntimeError('Component initial/final kernel metadata differs')
    cfg, _ = configuration(plan, Path(plan['remote_root']), stage, candidate, report['seed'])
    for name in ('base_sde', 'integrator'):
        if contract.get(name+'_config') != OmegaConf.to_container(cfg[name], resolve=True):
            raise RuntimeError('Resolved component configuration differs from plan: '+name)
    kernel = contract['kernel']
    if (kernel['solver'] != SOLVERS[variant.get('solver', 'legacy')]
            or kernel['grid'] != 'uniform_t'
            or kernel['reference'] != cfg.base_sde._target_.rsplit('.', 1)[1]
            or kernel['timesteps'] != report['sampling_metadata']['kernel']['timesteps']
            or kernel['additional_openmm_configurations'] != 0
            or kernel['additional_sampled_paths'] != 0):
        raise RuntimeError('Component kernel/diagnostic provenance mismatch')
    budget = plan['budgets']['smoke' if stage == 'mechanism_smoke' else 'geometry']
    if len(kernel['timesteps']) != budget['integration_steps']+1:
        raise RuntimeError('Component solver resolution differs from plan')
    variance = plan['component_diagnostics']['integrated_diffusion_variance']
    if not math.isclose(kernel['integrated_diffusion_variance'], variance, rel_tol=2e-5, abs_tol=2e-7):
        raise RuntimeError('Component continuous diffusion budget differs from plan')
    for key in ('forward_composed', 'backward_composed'):
        values = kernel[key]
        if (not all(math.isfinite(value) for value in values.values())
                or values['conditional_noise_variance'] <= 0):
            raise RuntimeError('Invalid component transition covariance')


def check_frozen_inputs(plan):
    for path, key in [('data_path', 'data_sha256'), ('calibration_path', 'calibration_sha256'),
                      ('diagnostic_report_path', 'diagnostic_report_sha256')]:
        if sha256(plan[path]) != plan[key]:
            raise RuntimeError('Frozen component input changed: '+path)
    for name, expected in plan['cached_baseline']['files'].items():
        if sha256(Path(plan['cached_baseline']['directory'])/name) != expected:
            raise RuntimeError('Cached component control changed: '+name)


def run_case(plan, campaign, stage, candidate, seed, source_sha):
    from tr_asbs.process.chart_source import load_calibration_bundle
    cfg, output = configuration(plan, campaign, stage, candidate, seed)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Preserve existing attempt; use a fresh output/snapshot for retry')
    bundle = load_calibration_bundle(cfg.potential.calibration_bundle_path)
    if bundle.get('energy_definition', {}).get('mode') != 'physical':
        raise RuntimeError('Components require the fixed physical-density bank')
    budget = plan['budgets']['smoke' if stage == 'mechanism_smoke' else 'geometry']
    observer = ComponentObserver(bundle, output, cfg.eval_samples_final, budget['phase_samples'],
        cfg.potential.chart_key, plan['stability_diagnostics']['bootstrap_count'])
    return train_case(dict(plan, max_openmm_per_geometry_run=plan['candidates'][candidate]['openmm_cap']),
        campaign, stage, cfg.potential.chart_key, seed, source_sha, config_and_output=(cfg, output),
        observer=observer, provenance=dict(candidate=candidate, plan_sha256=sha256(campaign/'plan.json'),
            upstream_source_sha256=plan['upstream_source_sha256'],
            diagnostic_report_sha256=plan['diagnostic_report_sha256'],
            cached_baseline=plan['cached_baseline'], study_family=plan['study_family']))


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
    check_frozen_inputs(plan)
    if args.stage == 'preflight':
        checks = []
        for candidate in plan['candidates']:
            output = run_case(plan, campaign, 'mechanism_smoke', candidate, 0, source_sha)
            report = json.loads((output/'result.json').read_text())
            document = json.loads((output/'receipt.json').read_text())
            verify_stability_report(report, plan, 'mechanism_smoke', candidate, document['files'])
            verify_component_report(report, plan, 'mechanism_smoke', candidate)
            if report['cost']['total_openmm_configurations'] != expected_cost(plan, 'mechanism_smoke', candidate)['total_openmm_configurations']:
                raise RuntimeError('Component smoke cost mismatch')
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
