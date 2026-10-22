#!/usr/bin/env python3
"""Version the physical-density bank and exercise the existing training pipeline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from run_aldp_chart_campaign import configuration, receipt, sha256, train_case, verify_source, write_json


def prepare(plan, campaign, source_sha):
    from tr_asbs.fab.boltzgen.openmm_interface import regularize_energy
    from tr_asbs.potential.alanine_dipeptide_energy import AlanineDipeptideEnergyIC
    from tr_asbs.process.chart_source import load_calibration_bundle

    old_path = Path(plan['legacy_calibration_path'])
    if sha256(old_path) != plan['legacy_calibration_sha256']:
        raise RuntimeError('Legacy calibration SHA mismatch')
    if sha256(plan['data_path']) != plan['data_sha256']:
        raise RuntimeError('Validation trajectory SHA mismatch')
    old = load_calibration_bundle(old_path)
    args = dict(dim=60, ref_samples_path=plan['data_path'], coordinate_chart='euclidean',
                energy_mode='physical', device='cpu')
    for role, selection in [('transform', plan['calibration']), ('reference', plan['bank_a'])]:
        args.update({f'{role}_{key}': value for key, value in selection.items()})
    potential = AlanineDipeptideEnergyIC(**args)
    if old['system'] != potential.aldp_potential.system_metadata:
        raise RuntimeError('Molecular system changed')
    potential.aldp_potential.p.coordinate_transform.load_state_dict(old['coordinate_transform'], strict=True)
    potential.chart.load_state_dict(old['base_chart'], strict=True)
    potential.chart.set_dihedral_offsets(old['chart_offsets']['current'])

    updated = copy.deepcopy(old)
    updated['energy_definition'] = potential.aldp_potential.energy_definition
    updated['energy_provenance'] = dict(source_sha256=source_sha,
        plan_sha256=sha256(campaign / 'plan.json'), legacy_calibration_sha256=sha256(old_path),
        role='same frozen coordinates; physical energies re-evaluated, no reselection')
    bank_checks = {}
    cut = float(old['system']['energy_cut'])
    cap = float(old['system']['energy_max'])
    for key, bank in old['banks'].items():
        raw = potential.eval_x(bank['x']).detach().cpu()
        if not torch.isfinite(raw).all() or not torch.all(raw < cut):
            raise RuntimeError('Frozen MD bank is nonfinite or crosses the legacy energy cut')
        difference = float((raw - bank['energies']).abs().max())
        if difference > plan['md_bank_energy_atol_kT']:
            raise RuntimeError('Re-evaluated MD bank differs from frozen energies')
        updated['banks'][key]['energies'] = raw
        bank_checks[key] = dict(samples=len(raw), maximum_energy=float(raw.max()),
                               maximum_legacy_difference_kT=difference)
    path = campaign / 'preflight/calibration.pt'
    path.parent.mkdir(parents=True)
    torch.save(updated, path)

    # True force versus native energy finite differences, with all Jacobians.
    potential.aldp_potential.p.double()
    potential.chart.double()
    x = old['banks']['a']['x'][:4].double()
    y = potential.cartesian_to_native(x).detach().requires_grad_(True)
    gradient = torch.autograd.grad(potential.eval(y).sum(), y)[0]
    bond_index = int(torch.nonzero(potential.chart.kinds == 0)[-1])
    angle_index = int(torch.nonzero(potential.chart.kinds == 1)[0])
    indices = [bond_index, angle_index, *old['phi_psi_native_indices']]
    step = plan['finite_difference_step']
    discrepancies = {}
    for index in indices:
        delta = torch.zeros_like(y)
        delta[:, index] = step
        finite = (potential.eval(y + delta) - potential.eval(y - delta)) / (2 * step)
        error = (finite - gradient[:, index]).abs() / (1 + finite.abs())
        discrepancies[str(index)] = float(error.max())
        if not torch.isfinite(error).all() or error.max() > plan['gradient_relative_atol']:
            raise RuntimeError('Physical force/native density finite-difference mismatch')
    # A real OpenMM collision checks that even energies above the old finite cap
    # survive in the density and derivative. Atoms0/20 are nonbonded hydrogens.
    extreme = x[:1].clone().reshape(1, 22, 3)
    extreme[:, 0] = extreme[:, 20] + extreme.new_tensor([1e-3, 0., 0.])
    extreme = extreme.reshape(1, 66).requires_grad_(True)
    raw = potential.eval_x(extreme)
    force = torch.autograd.grad(raw.sum(), extreme)[0]
    regularized = regularize_energy(raw, raw.new_tensor(cut), raw.new_tensor(cap))
    if (not torch.isfinite(raw).all() or not torch.isfinite(force).all()
            or not torch.all(raw > cap) or not torch.all(raw > regularized)):
        raise RuntimeError('Physical tail audit failed')
    audit = dict(status='passed', source_sha256=source_sha,
        plan_sha256=sha256(campaign / 'plan.json'), energy_definition=updated['energy_definition'],
        calibration_sha256=sha256(path), legacy_calibration_sha256=sha256(old_path),
        coordinate_transform_and_chart_reused=True, frame_selection_unchanged=True,
        bank_checks=bank_checks, finite_difference_relative_errors=discrepancies,
        collision_energy_kT=float(raw.detach()[0]),
        collision_legacy_regularized_energy_kT=float(regularized.detach()[0]),
        collision_maximum_force_kT_nm=float(force.abs().max()),
        setup_openmm_configurations=potential.aldp_potential.openmm_configurations_evaluated,
        scientific_status='execution_gate_only_not_reference_or_geometry_admission')
    write_json(campaign / 'preflight/density_audit.json', audit)
    return path, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / 'plan.json').read_text())
    if plan['energy_mode'] != 'physical':
        raise RuntimeError('This preflight only admits a physical-density execution path')
    if (campaign / 'preflight').exists():
        raise RuntimeError('Preserve previous preflight; use a new version/output')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Configured remote CUDA unavailable')
    source_sha = verify_source()
    path, audit = prepare(plan, campaign, source_sha)
    cfg, output = configuration(plan, campaign, 'smoke', 'current', 0)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.potential.energy_mode = 'physical'
        cfg.potential.calibration_bundle_path = str(path)
        cfg.source.calibration_bundle_path = str(path)
    result_path = train_case(plan, campaign, 'smoke', 'current', 0, source_sha,
        config_and_output=(cfg, output), provenance=dict(role='physical_density_execution_preflight',
            plan_sha256=sha256(campaign / 'plan.json'), legacy_target_status='diagnostic_only'))
    result = json.loads((result_path / 'result.json').read_text())
    expected_steps = 2 * plan['budgets']['smoke']['outer_loops'] * plan['budgets']['smoke']['epochs_per_phase'] * plan['budgets']['smoke']['steps_per_epoch']
    if (result['rejected_training_steps'] or result['global_step'] != expected_steps
            or result['native_cartesian_roundtrip_error'] > 2e-5
            or result['sampling_metadata']['data_provenance']['energy_definition']['mode'] != 'physical'):
        raise RuntimeError('Physical-density training/checkpoint smoke failed')
    write_json(campaign / 'preflight/preflight_result.json', dict(status='passed',
        source_sha256=source_sha, plan_sha256=sha256(campaign / 'plan.json'),
        calibration_sha256=sha256(path), density_audit_sha256=sha256(campaign / 'preflight/density_audit.json'),
        smoke_directory=str(result_path), result_sha256=sha256(result_path / 'result.json'),
        receipt_sha256=sha256(result_path / 'receipt.json'),
        cost=dict(setup_openmm_configurations=audit['setup_openmm_configurations'],
                  smoke_openmm_configurations=result['cost']['total_openmm_configurations']),
        scientific_status='execution_gate_only_not_reference_or_geometry_admission'))
    receipt(campaign / 'preflight', dict(stage='physical_preflight'),
        dict(source_sha256=source_sha, plan_sha256=sha256(campaign / 'plan.json'),
             legacy_calibration_sha256=plan['legacy_calibration_sha256']))
    print('postflight_verified=true', flush=True)


if __name__ == '__main__':
    main()
