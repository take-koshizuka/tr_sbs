#!/usr/bin/env python3
"""One scheduler snapshot: verify preflight, submit planned pilots, then report."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path

from monitor_aldp_chart_campaign import ACTIVE, execute, read_manifest, scheduler
from run_aldp_chart_campaign import ROOT, receipt, sha256, verify_source, write_json

FIELDS = ('job_id', 'stage', 'candidate', 'seed', 'run_directory', 'log_path',
          'repo', 'python', 'source_sha256', 'plan_sha256')


def verify_case(directory, plan, source_sha, stage, candidate, seed):
    directory = Path(directory)
    chart = plan['candidates'][candidate].get('chart_key', 'current')
    document = json.loads((directory / 'receipt.json').read_text())
    if document.get('schema') != 'aldp_chart_receipt_v1' or not document.get('postflight_verified'):
        raise RuntimeError('Invalid receipt')
    if document['identity'] != dict(stage=stage, chart=chart, seed=seed):
        raise RuntimeError('Receipt identity mismatch')
    inputs = document['inputs']
    if inputs['source_sha256'] != source_sha or inputs['calibration_sha256'] != plan['calibration_sha256']:
        raise RuntimeError('Receipt input SHA mismatch')
    for name, expected in document['files'].items():
        if sha256(directory / name) != expected:
            raise RuntimeError('Artifact SHA mismatch: ' + name)
    report = json.loads((directory / 'result.json').read_text())
    provenance = report['provenance']
    if provenance != inputs['provenance'] or provenance['candidate'] != candidate:
        raise RuntimeError('Candidate provenance mismatch')
    if provenance['plan_sha256'] != plan['_plan_sha256']:
        raise RuntimeError('Plan SHA mismatch')
    if report['status'] != 'artifact_complete' or report['rejected_training_steps']:
        raise RuntimeError('Training incomplete or numerically failed')
    if report['config_sha256'] != sha256(directory / 'config.yaml'):
        raise RuntimeError('Config SHA mismatch')
    if (report['stage'], report['chart'], report['seed']) != (stage, chart, seed):
        raise RuntimeError('Result identity mismatch')
    if chart != 'current':
        metadata = report['sampling_metadata']
        if metadata['target']['chart_key'] != chart or metadata['source']['chart_key'] != chart:
            raise RuntimeError('Sampling chart metadata mismatch')
        audit = report['diagnostics'].get('physical_source_audit', {})
        if (audit.get('status') != 'passed' or audit.get('chart') != chart
                or audit.get('reference_chart') != 'current'
                or audit.get('source_scale') != plan['candidates'][candidate]['source_scale']
                or audit.get('sample_count') != plan['budgets'][
                    'smoke' if stage == 'mechanism_smoke' else 'geometry']['final_samples']
                or audit.get('additional_openmm_configurations') != 0
                or not audit.get('rng_isolated')
                or not math.isfinite(audit.get('max_cartesian_error_nm', math.inf))
                or not math.isfinite(audit.get('max_physical_log_density_error', math.inf))
                or audit['max_cartesian_error_nm'] > 2e-5
                or audit['max_physical_log_density_error'] > 2e-3):
            raise RuntimeError('Paired physical-source audit incomplete')
    budget = plan['budgets']['smoke' if stage == 'mechanism_smoke' else 'geometry']
    loops = budget['outer_loops'] if stage == 'mechanism_smoke' else plan['candidates'][candidate]['outer_loops']
    first = plan['candidates'][candidate].get('smoke_initial_adjoint_matching_epochs'
        if stage == 'mechanism_smoke' else 'initial_adjoint_matching_epochs')
    first = budget['epochs_per_phase'] if first is None else first
    expected_steps = (loops * budget['epochs_per_phase'] * 2
                      + first - budget['epochs_per_phase']) * budget['steps_per_epoch']
    if report['global_step'] != expected_steps:
        raise RuntimeError('Optimizer budget incomplete')
    cost = report['cost']
    if cost['total_openmm_configurations'] != sum(cost[key] for key in (
        'training_and_online_openmm_configurations', 'final_evaluation_openmm_configurations',
        'diagnostic_openmm_configurations')):
        raise RuntimeError('Cost boundaries disagree')
    if cost['total_openmm_configurations'] > plan['candidates'][candidate]['openmm_cap']:
        raise RuntimeError('OpenMM cap exceeded')
    variant = plan['candidates'][candidate]
    prefix_loops = (variant.get('smoke_prefix_outer_loops', 0) if stage == 'mechanism_smoke'
                    else variant.get('diagnostic_prefix_outer_loops', 0))
    if prefix_loops:
        prefix = report['diagnostics'].get(f'after_outer{prefix_loops}', {})
        epoch = prefix_loops * budget['epochs_per_phase'] * 2
        if prefix.get('epoch') != epoch or prefix.get('global_step') != epoch * budget['steps_per_epoch']:
            raise RuntimeError('Prefix diagnostic incomplete or wrong training step')
        for suffix in ('cartesian', 'physical_energy'):
            if f'outer{prefix_loops}_{suffix}.npy' not in document['files']:
                raise RuntimeError('Prefix array absent from receipt')
        for bank in ('a', 'b'):
            for key in plan['geometry_gate']['minimum_thresholds']:
                if not math.isfinite(prefix['metrics'][bank][key]):
                    raise RuntimeError('Nonfinite prefix metric: ' + key)
    for bank in report['metrics'].values():
        for key in plan['geometry_gate']['minimum_thresholds']:
            if not math.isfinite(bank[key]):
                raise RuntimeError('Nonfinite metric: ' + key)
    if plan.get('stability_diagnostics'):
        from run_aldp_stability_campaign import verify_stability_report
        verify_stability_report(report, plan, stage, candidate, document['files'])
    if plan.get('component_diagnostics'):
        from run_aldp_component_campaign import verify_component_report
        verify_component_report(report, plan, stage, candidate)
    return report


def verify_job(row, campaign, plan):
    source_sha = sha256(ROOT / 'source_manifest.json')
    if row['source_sha256'] != source_sha or row['plan_sha256'] != plan['_plan_sha256']:
        raise RuntimeError('Ledger source/plan SHA mismatch')
    if 'postflight_verified=true' not in Path(row['log_path']).read_text(errors='replace'):
        raise RuntimeError('Missing postflight log marker')
    if row['stage'] != 'preflight':
        return verify_case(row['run_directory'], plan, source_sha, row['stage'],
                           row['candidate'], int(row['seed']))
    directory = Path(row['run_directory'])
    document = json.loads((directory / 'receipt.json').read_text())
    if not document.get('postflight_verified') or document['inputs']['source_sha256'] != source_sha:
        raise RuntimeError('Preflight receipt incomplete')
    if document['inputs']['plan_sha256'] != plan['_plan_sha256']:
        raise RuntimeError('Preflight plan SHA mismatch')
    for name, expected in document['files'].items():
        if sha256(directory / name) != expected:
            raise RuntimeError('Preflight artifact SHA mismatch')
    report = json.loads((directory / 'preflight_result.json').read_text())
    if report['status'] != 'passed' or {r['candidate'] for r in report['checks']} != set(plan['candidates']):
        raise RuntimeError('Preflight does not cover all candidates')
    for check in report['checks']:
        path = campaign / 'mechanism_smoke' / f"{check['candidate']}_seed0"
        for name, key in [('result.json', 'result_sha256'), ('receipt.json', 'receipt_sha256')]:
            if sha256(path / name) != check[key]:
                raise RuntimeError('Preflight smoke SHA mismatch')
        verify_case(path, plan, source_sha, 'mechanism_smoke', check['candidate'], 0)
    return report


def submit(campaign, plan, python, rows, stage, candidate='', dry_run=False):
    if any((r['stage'], r['candidate'], int(r['seed'])) == (stage, candidate, 0) for r in rows):
        return None
    output = (campaign / 'preflight' if stage == 'preflight'
              else campaign / stage / f'{candidate}_seed0')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Unledgered output exists; reconcile before submitting')
    label = 'prep' if stage == 'preflight' else candidate
    log = campaign / 'logs' / f'aldp-mech-{label}-%j.out'
    command = ['sbatch', '--parsable', f'--job-name=aldp-mech-{label}', '--partition=gpu',
        '--gres=gpu:1', '--cpus-per-task=4', '--mem=16G', '--time=02:00:00',
        '--exclude=htc-gpu024', f'--output={log}', str(ROOT / plan.get(
            'executor', 'scripts/slurm/exec_aldp_mechanism_campaign.sh')),
        str(ROOT), str(python), str(campaign), stage, candidate or 'source_half', '0']
    if dry_run:
        return dict(command=command, stage=stage, candidate=candidate)
    intent = campaign / 'submission_intents' / f'{stage}_{candidate or "all"}_seed0.json'
    if intent.exists():
        raise RuntimeError('Submission outcome needs reconciliation; never submit the same intent twice')
    write_json(intent, dict(command=command, output=str(output), status='dispatching'))
    job = execute(command).stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unparseable Slurm job ID: ' + job)
    row = dict(job_id=job, stage=stage, candidate=candidate, seed=0,
        run_directory=str(output), log_path=str(log).replace('%j', job), repo=str(ROOT),
        python=str(python), source_sha256=sha256(ROOT / 'source_manifest.json'),
        plan_sha256=plan['_plan_sha256'])
    write_json(intent, dict(command=command, row=row, status='submitted'))
    path = campaign / 'submitted_jobs.tsv'
    new = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t')
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--advance', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / 'plan.json').read_text())
    plan['_plan_sha256'] = sha256(campaign / 'plan.json')
    verify_source()
    if sha256(plan['calibration_path']) != plan['calibration_sha256']:
        raise RuntimeError('Frozen calibration changed')
    cached = plan.get('cached_baseline') or {}
    for name, expected in cached.get('files', {}).items():
        if sha256(Path(cached['directory']) / name) != expected:
            raise RuntimeError('Cached baseline changed: ' + name)
    if sha256(plan['diagnostic_report_path']) != plan['diagnostic_report_sha256']:
        raise RuntimeError('Upstream diagnosis changed')
    with (campaign / '.monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_manifest(campaign / 'submitted_jobs.tsv')
        verified, failures, counts = {}, [], {}
        for row in rows:
            state, code = scheduler(row['job_id'])
            counts[state] = counts.get(state, 0) + 1
            if state == 'COMPLETED' and code == '0:0':
                try:
                    verified[row['job_id']] = verify_job(row, campaign, plan)
                except Exception as exc:
                    failures.append(dict(job_id=row['job_id'], state=state, reason=str(exc)))
            elif state not in ACTIVE:
                failures.append(dict(job_id=row['job_id'], state=state, exit_code=code,
                    log_path=row['log_path'], log_tail=Path(row['log_path']).read_text(errors='replace')[-4000:]
                    if Path(row['log_path']).exists() else 'log unavailable'))
        new = []
        if args.advance and not failures:
            if not rows:
                row = submit(campaign, plan, args.python, rows, 'preflight', dry_run=args.dry_run)
                if row:
                    new.append(row)
            elif any(row['stage'] == 'preflight' and row['job_id'] in verified for row in rows):
                for candidate in plan['candidates']:
                    row = submit(campaign, plan, args.python, rows, 'mechanism_pilot',
                                 candidate, args.dry_run)
                    if row:
                        new.append(row)
        reports = [report for report in verified.values() if report.get('stage') == 'mechanism_pilot']
        complete = len(reports) == len(plan['candidates']) and not failures
        if complete:
            summary = dict(schema='aldp_mechanism_summary_v1', rows=reports,
                cached_baseline=plan['cached_baseline'], selected_chart=None,
                scientific_status='geometry_not_admitted_mechanism_diagnosis_only',
                next_action=plan.get('later_stages',
                    'review source coverage, clipping and learning curve before repeated cut comparison'))
            write_json(campaign / 'aggregate/mechanism_summary.json', summary)
            receipt(campaign / 'aggregate', dict(stage='mechanism_summary'),
                    dict(source_sha256=sha256(ROOT / 'source_manifest.json'),
                         plan_sha256=plan['_plan_sha256']))
        previous = json.loads((campaign / 'monitor_state.json').read_text()) if (campaign / 'monitor_state.json').exists() else {}
        waiting = previous.get('consecutive_pending_or_running', 0) + 1 if any(k in ACTIVE for k in counts) else 0
        state = dict(counts=counts, verified_job_ids=list(verified), failures=failures,
            new_submissions=new, complete=complete, consecutive_pending_or_running=waiting,
            recommended_monitor_minutes=30 if waiting >= 2 else 10)
        if not args.dry_run:
            write_json(campaign / 'monitor_state.json', state)
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
