#!/usr/bin/env python3
"""A-only finite-grid prefix screening after an audited normal reference.

This analysis proposes at most one shorter candidate on each axis. It neither
submits jobs nor establishes a minimum, convergence or geometry admission.
"""
import argparse
import json
import math
from pathlib import Path

from run_aldp_chart_campaign import sha256, write_json


def prefix_checks(candidate, reference, limits):
    checks = {}

    def upper(name, value, maximum):
        value, maximum = float(value), float(maximum)
        checks[name] = dict(value=value, bound=maximum,
            passed=math.isfinite(value) and math.isfinite(maximum) and value <= maximum)

    def lower(name, value, minimum):
        value, minimum = float(value), float(minimum)
        checks[name] = dict(value=value, bound=minimum,
            passed=math.isfinite(value) and math.isfinite(minimum) and value >= minimum)

    upper('heldout_loss', candidate['validation']['heldout_loss'],
        limits['heldout_loss_ratio']*reference['validation']['heldout_loss'])
    current, full = candidate['policy'], reference['policy']
    # B is disclosure only; every decision below reads the preregistered A bank.
    a, ar = current['metrics']['a'], full['metrics']['a']
    for field, limit in (
        ('torsion_js_mean', 'torsion_js_absolute_increase'),
        ('coarse_basin_tvd', 'coarse_basin_tvd_absolute_increase'),
        ('missing_reference_mass', 'missing_mass_absolute_increase')):
        upper(field, a[field], ar[field]+limits[limit])
    upper('physical_energy_w2', a['physical_energy_w2'],
        ar['physical_energy_w2']*limits['physical_energy_w2_ratio'])
    upper('close_contact_fraction', current['geometry']['fraction_below_reference_q01'],
        full['geometry']['fraction_below_reference_q01']+limits['close_contact_fraction_absolute_increase'])
    for index in (0, 1):
        upper('near_cut_'+str(index), current['near_cut_fraction_phi_psi'][index],
            full['near_cut_fraction_phi_psi'][index]+limits['near_cut_fraction_absolute_increase'])
    upper('raw_log_path_ratio_variance', current['raw_log_path_ratio_variance'],
        full['raw_log_path_ratio_variance']*limits['raw_log_path_ratio_variance_ratio'])
    lower('raw_path_ess_fraction', current['raw_path_ess_fraction'],
        full['raw_path_ess_fraction']*limits['raw_path_ess_fraction_ratio_min'])
    return checks


def screen(report, contract):
    if (report.get('global_step') != 300000 or report.get('rejected_training_steps')
        or report.get('minimum_inner_duration') != 'not_established'):
        raise ValueError('A completed normal-duration reference is required')
    if contract['selection_bank'] != 'a' or contract['test_bank'] != 'closed':
        raise ValueError('Keep A selection and test-closed boundaries')
    if report['diagnostics']['curve_contract'] != contract:
        raise ValueError('Use the original preregistered curve contract')
    boundaries = report['diagnostics']['inner_boundaries']
    expected = [(p,e) for p in ('u','h') for e in [0]+contract['milestones'][p]]
    if [(row['phase'],row['phase_epoch']) for row in boundaries] != expected:
        raise ValueError('Incomplete or reordered prefix curves')
    result = dict(schema='aldp_inner_prefix_screening_v1',status='screened',
        decision_bank='a',disclosure_bank='b',test='closed',phases={},
        minimum_inner_duration='not_established',geometry_admission='not_evaluated',
        next_outer_confirmation='not_evaluated',new_jobs_submitted=0)
    for phase in ('u','h'):
        rows = [row for row in boundaries if row['phase']==phase and row['phase_epoch']>0]
        reference = rows[-1]
        comparisons = []
        for row in rows:
            if (row['validation']['sample_count'] != contract['validation_samples']
                or row['policy']['physical_energy']['count'] != contract['quality_samples']):
                raise ValueError('Wrong validation/raw-quality sample count')
            checks = prefix_checks(row,reference,contract['non_deterioration_limits'])
            comparisons.append(dict(phase_epoch=row['phase_epoch'],snapshot=row['snapshot'],
                prefix_gates_passed=all(item['passed'] for item in checks.values()),checks=checks,
                raw_physical_energy_tail_quantiles=row['policy']['physical_energy']['quantiles'],
                disclosure_b=row['policy']['metrics']['b']))
        shorter = [r for r in comparisons if r['prefix_gates_passed']
                   and r['phase_epoch'] < contract[phase+'_epochs']]
        candidate = shorter[0] if shorter else None
        result['phases'][phase] = dict(reference_epoch=reference['phase_epoch'],comparisons=comparisons,
            proposed_shorter_epoch=candidate['phase_epoch'] if candidate else None,
            proposed_snapshot=candidate['snapshot'] if candidate else None,
            forward_quality_is_invariant_at_fixed_u=phase=='h',
            proposal_role='screening only; shorter u changes h data and both require next-outer confirmation')
    return result


def verified_inputs(run, plan_path, audit_path):
    plan = json.loads(plan_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'normal_curve_verified' or audit['receipt_files_verified'] < 1:
        raise ValueError('Completed accounting/source/input/full-receipt audit required')
    if (audit['source_sha256'] != plan['source_manifest_sha256']
        or audit['plan_sha256'] != sha256(plan_path)
        or audit['result_sha256'] != sha256(run/'result.json')):
        raise ValueError('Normal curve audit SHA mismatch')
    rec = json.loads((run/'receipt.json').read_text())
    for name,digest in rec['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(run/path) != digest:
            raise ValueError('Receipt path/SHA mismatch: '+name)
    report = json.loads((run/'result.json').read_text())
    if (report['source_sha256'] != audit['source_sha256']
        or report['provenance']['plan_sha256'] != audit['plan_sha256']):
        raise ValueError('Result provenance differs from audited normal reference')
    return report,plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-directory',required=True,type=Path)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report,plan=verified_inputs(args.run_directory,args.plan,args.audit)
    result=screen(report,plan['curve_contract'])
    result.update(input_result_sha256=sha256(args.run_directory/'result.json'),
        input_plan_sha256=sha256(args.plan),input_audit_sha256=sha256(args.audit))
    write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],minimum='not_established',
        proposed_shorter_epochs={p:result['phases'][p]['proposed_shorter_epoch'] for p in ('u','h')})))


if __name__=='__main__':
    main()
