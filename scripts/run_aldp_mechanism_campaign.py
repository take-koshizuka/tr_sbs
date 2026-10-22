#!/usr/bin/env python3
"""Isolated geometry diagnoses through the existing ALDP training runner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np
import torch

from diagnose_aldp_geometry import distance_summary, energy_summary, nonbonded_pairs
from run_aldp_chart_campaign import (configuration as base_configuration,
    distribution_metrics, receipt, sha256, train_case, verify_source, write_json)


@contextmanager
def diagnostic_rng(device):
    """Diagnostics must leave the training RNG and policy mode unchanged."""
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        yield


def configuration(plan, campaign, stage, candidate, seed):
    from omegaconf import open_dict
    variant = plan['candidates'][candidate]
    chart = variant.get('chart_key', 'current')
    upstream = Path(plan['upstream_campaign'])
    cfg, _ = base_configuration(plan, upstream,
        'smoke' if stage == 'mechanism_smoke' else 'geometry_pilot', chart, seed)
    output = campaign / stage / f'{candidate}_seed{seed}'
    with open_dict(cfg):
        cfg.name, cfg.root = output.name, str(output)
        cfg.fabric.loggers.save_dir = str(output)
        cfg.checkpoint_directory = str(output / 'checkpoints')
        cfg.sample_directory = str(output / 'samples')
        cfg.source.scale = variant['source_scale']
        cfg.terminal_cost.grad_clip_val = variant['terminal_gradient_clip']
        if 'energy_mode' in plan:
            cfg.potential.energy_mode = plan['energy_mode']
            cfg.potential.calibration_bundle_path = plan['calibration_path']
            cfg.source.calibration_bundle_path = plan['calibration_path']
        if stage != 'mechanism_smoke':
            cfg.max_outer_loops = variant['outer_loops']
    return cfg, output


@torch.no_grad()
def paired_physical_source_audit(module, y, cartesian, chart_key):
    """Check the same seeded physical source through current and shifted charts."""
    from tr_asbs.process.chart_source import ChartMappedGaussianPrior
    with diagnostic_rng(module.fabric.device):
        reference = ChartMappedGaussianPrior(module.cfg.potential.calibration_bundle_path,
            'current', dim=module.source.dim, scale=module.source.scale,
            device=module.fabric.device)
        torch.manual_seed(12455)
        base_y = reference.sample((len(y),))
        base_z, base_jacobian = reference.chart(base_y)
        base_x = module.potential.aldp_potential.z_to_x(base_z)
        _, jacobian = module.source.chart(y)
        physical_error = float((base_x - cartesian).abs().max())
        density_error = float(((reference.log_prob(base_y) - base_jacobian)
            - (module.source.log_prob(y) - jacobian)).abs().max())
        if not all(math.isfinite(value) for value in (physical_error, density_error)):
            raise RuntimeError('Nonfinite paired physical-source audit')
        if physical_error > 2e-5 or density_error > 2e-3:
            raise RuntimeError('Shifted chart changed the paired physical source')
        return dict(status='passed', reference_chart='current', chart=chart_key,
            sample_count=len(y), source_scale=module.source.scale,
            max_cartesian_error_nm=physical_error, max_physical_log_density_error=density_error,
            rng_isolated=True, additional_openmm_configurations=0)


class MechanismObserver:
    def __init__(self, bundle, output, count, extended, chart_key='current', *,
                 prefix_outer_loops=2, prefix_epoch=80):
        from openmmtools.testsystems import AlanineDipeptideImplicit
        self.bundle, self.output, self.count, self.extended = bundle, output, count, extended
        self.chart_key = chart_key
        self.prefix_outer_loops, self.prefix_epoch = prefix_outer_loops, prefix_epoch
        self.pairs = nonbonded_pairs(AlanineDipeptideImplicit(constraints=None).topology)
        xyz = bundle['banks']['a']['x'].numpy().reshape(-1, 22, 3)
        self.ref_distances = np.linalg.norm(
            xyz[:, self.pairs[:, 0]] - xyz[:, self.pairs[:, 1]], axis=-1).min(axis=1)
        self.probe = None
        self.report = dict(source={}, untrained_policy={}, online=[])

    @torch.no_grad()
    def state(self, module, y, cartesian, energy):
        from tr_asbs.utils.md_utils import compute_all_dihedrals, extract_phi_psi
        xyz = cartesian.detach().cpu().numpy()
        energy = energy.detach().cpu().reshape(-1).numpy()
        angles = extract_phi_psi(compute_all_dihedrals(cartesian, 22, 3)).numpy()
        gradient = module.potential.grad_E(y[:min(128, len(y))]).double().norm(dim=-1).cpu().numpy()
        clip = float(module.cfg.terminal_cost.grad_clip_val)
        banks = self.bundle['banks']
        metrics = {}
        for key in ('a', 'b'):
            bank = banks[key]
            indices = np.random.default_rng(22022).choice(len(bank['x']), len(y), replace=False)
            metrics[key] = distribution_metrics(angles, energy,
                bank['phi_psi'][indices].numpy(), bank['energies'][indices].numpy())
        return dict(physical_energy=energy_summary(energy, banks['a']['energies'].numpy()),
            geometry=distance_summary(xyz, self.pairs, self.ref_distances), metrics=metrics,
            native_gradient_norm=energy_summary(gradient),
            fraction_gradient_above_terminal_clip=float(np.mean(gradient > clip)),
            median_terminal_clip_multiplier=float(np.median(np.minimum(1., clip / (gradient + 1e-6)))))

    @torch.no_grad()
    def policy(self, module, label):
        torch.manual_seed(12345)
        x0 = module.source.sample((self.count,))
        result = module.integrator.run_with_logweights(module.integrator,
            module.source.log_prob, module.potential.log_prob, x0,
            center_every_step=False, return_trajectory=False, progress_bar=False)
        y = result['data_1']
        cartesian = module.potential.native_to_cartesian(y)
        energy = module.potential.eval_x(cartesian).detach()
        for name, tensor in [('cartesian', cartesian), ('physical_energy', energy)]:
            np.save(self.output / f'{label}_{name}.npy', tensor.cpu().numpy())
        from tr_asbs.utils.eval_logweight import compute_raw_path_ess
        state = self.state(module, y, cartesian, energy)
        state.update(raw_path_ess_fraction=float(compute_raw_path_ess(result['log_Ph_Pref'])),
            raw_log_path_ratio_variance=float(result['log_Ph_Pref'].double().var(unbiased=False)),
            achieved_forward_kl=float(result['log_Pref_Pu_det'].double().mean()))
        return state

    @torch.no_grad()
    def controls(self, module):
        return {str(t): energy_summary(module.controller(
            torch.full((len(self.probe),), t, device=self.probe.device), self.probe
            ).double().norm(dim=-1).cpu().numpy()) for t in (0., .5, 1.)}

    @torch.no_grad()
    def __call__(self, module, event, payload=None):
        with diagnostic_rng(module.fabric.device):
            if event == 'initial':
                torch.manual_seed(12455)
                y = module.source.sample((self.count,))
                self.probe = y[:min(64, self.count)].detach()
                cartesian = module.potential.native_to_cartesian(y)
                if self.chart_key != 'current':
                    self.report['physical_source_audit'] = paired_physical_source_audit(
                        module, y, cartesian, self.chart_key)
                energy = module.potential.eval_x(cartesian).detach()
                self.report['source'] = self.state(module, y, cartesian, energy)
                self.report['untrained_policy'] = self.policy(module, 'untrained')
                self.report['initial_control_norm'] = self.controls(module)
            elif event == 'online':
                item = dict(epoch=payload['epoch'], global_step=int(module.global_step),
                            gamma=float(module.gamma), control_norm=self.controls(module),
                            training_losses=getattr(module, 'last_training_losses', {}))
                buffer = module.adjoint_buffer.storage
                if buffer.get('log_Ph_Pref'):
                    iw = module._compute_iw({key: torch.stack(buffer[key])
                        for key in ('log_Ph_Pref', 'log_Pu_Pref')}, module.gamma).double()
                    item['buffer_iw_ess_fraction'] = float((iw.sum().square() / iw.square().sum() / len(iw)).cpu())
                self.report['online'].append(item)
                if self.extended and payload['epoch'] == self.prefix_epoch:
                    label = f'outer{self.prefix_outer_loops}'
                    state = self.policy(module, label)
                    state.update(epoch=payload['epoch'], global_step=int(module.global_step))
                    self.report[f'after_{label}'] = state
            elif event == 'final':
                self.report['trained_policy'] = self.state(module, payload['y'],
                    payload['cartesian'], payload['physical_energy'])
                self.report['diagnostic_rng_isolated'] = True
                self.report['source_seed'] = 12455
                self.report['paired_policy_report_seed'] = 12345
                self.report['cut_role'] = f'{self.chart_key}_fixed_for_diagnosis_not_admitted'
                return self.report


def run_case(plan, campaign, stage, candidate, seed, source_sha):
    from tr_asbs.process.chart_source import load_calibration_bundle
    cfg, output = configuration(plan, campaign, stage, candidate, seed)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Preserve existing attempt; use a new version/output')
    bundle = load_calibration_bundle(cfg.potential.calibration_bundle_path)
    if sha256(cfg.potential.calibration_bundle_path) != plan['calibration_sha256']:
        raise RuntimeError('Frozen calibration SHA mismatch')
    if plan.get('energy_mode') == 'physical':
        if sha256(plan['diagnostic_report_path']) != plan['diagnostic_report_sha256']:
            raise RuntimeError('Physical density audit SHA mismatch')
        if bundle.get('energy_definition', {}).get('mode') != 'physical':
            raise RuntimeError('Physical mechanism requires physical-density energy banks')
    variant = plan['candidates'][candidate]
    prefix = (variant.get('smoke_prefix_outer_loops', 0) if stage == 'mechanism_smoke'
              else variant.get('diagnostic_prefix_outer_loops', 2 if candidate == 'training5x' else 0))
    if prefix < 0 or prefix > cfg.max_outer_loops:
        raise RuntimeError('Diagnostic prefix lies outside the training budget')
    observer = MechanismObserver(bundle, output, cfg.eval_samples_final, prefix > 0,
        cfg.potential.chart_key, prefix_outer_loops=prefix or 2,
        prefix_epoch=prefix * (cfg.adjoint_matching_epochs + cfg.corrector_matching_epochs))
    budget = dict(plan, max_openmm_per_geometry_run=plan['candidates'][candidate]['openmm_cap'])
    return train_case(budget, campaign, stage, cfg.potential.chart_key, seed, source_sha,
        config_and_output=(cfg, output), observer=observer,
        provenance=dict(candidate=candidate, plan_sha256=sha256(campaign / 'plan.json'),
                        upstream_source_sha256=plan['upstream_source_sha256'],
                        diagnostic_report_sha256=plan['diagnostic_report_sha256'],
                        cached_baseline=plan['cached_baseline']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'mechanism_pilot'), required=True)
    parser.add_argument('--candidate', default='source_half')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / 'plan.json').read_text())
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Configured remote GPU unavailable')
    source_sha = verify_source()
    if args.stage == 'preflight':
        checks = []
        for candidate in plan['candidates']:
            output = run_case(plan, campaign, 'mechanism_smoke', candidate, 0, source_sha)
            report = json.loads((output / 'result.json').read_text())
            if report['rejected_training_steps'] or report['native_cartesian_roundtrip_error'] > 2e-5:
                raise RuntimeError('Mechanism smoke failed')
            if not report['diagnostics']['diagnostic_rng_isolated']:
                raise RuntimeError('Diagnostic RNG isolation missing')
            checks.append(dict(candidate=candidate, result_sha256=sha256(output / 'result.json'),
                               receipt_sha256=sha256(output / 'receipt.json')))
        write_json(campaign / 'preflight/preflight_result.json',
            dict(status='passed', checks=checks, source_sha256=source_sha))
        receipt(campaign / 'preflight', dict(stage='preflight'),
            dict(source_sha256=source_sha, plan_sha256=sha256(campaign / 'plan.json'),
                 calibration_sha256=plan['calibration_sha256']))
    else:
        run_case(plan, campaign, args.stage, args.candidate, args.seed, source_sha)
    print('postflight_verified=true', flush=True)


if __name__ == '__main__':
    main()
