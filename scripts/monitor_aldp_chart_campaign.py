#!/usr/bin/env python3
"""One-shot Slurm check/advance; call periodically, never wait on running jobs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

from run_aldp_chart_campaign import CHARTS, ROOT, sha256, write_json

ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING'}
FIELDS = ('job_id', 'stage', 'chart', 'seed', 'run_directory', 'log_path',
          'repo', 'python', 'source_sha256', 'plan_sha256')


def execute(arguments, check=True):
    return subprocess.run(arguments, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=check)


def scheduler(job):
    queue = execute(['squeue', '-h', '-j', job, '-o', '%T'], False)
    if queue.returncode == 0 and queue.stdout.strip():
        return queue.stdout.strip().splitlines()[0], ''
    account = execute(['sacct', '-n', '-P', '-j', job,
                       '--format=JobIDRaw,State,ExitCode'], False)
    for line in account.stdout.splitlines():
        parts = line.split('|')
        if len(parts) >= 3 and parts[0] == job:
            return parts[1].split()[0].rstrip('+'), parts[2]
    return 'UNKNOWN', ''


def read_manifest(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def verify_receipt(directory, row):
    receipt = json.loads((directory / 'receipt.json').read_text())
    if receipt.get('schema') != 'aldp_chart_receipt_v1' or not receipt.get('postflight_verified'):
        raise RuntimeError('Invalid postflight receipt')
    if receipt['inputs']['source_sha256'] != row['source_sha256']:
        raise RuntimeError('Receipt source SHA mismatch')
    for relative, expected in receipt['files'].items():
        if sha256(directory / relative) != expected:
            raise RuntimeError('Artifact SHA mismatch: ' + relative)
    if row['stage'] == 'preflight':
        if receipt['inputs']['plan_sha256'] != row['plan_sha256']:
            raise RuntimeError('Preflight plan SHA mismatch')
        report = json.loads((directory / 'preflight_result.json').read_text())
        if report['status'] != 'passed' or len(report['agreement']) != 3:
            raise RuntimeError('Preflight validation incomplete')
    else:
        report = json.loads((directory / 'result.json').read_text())
        if report['stage'] != row['stage'] or report['chart'] != row['chart'] or report['seed'] != int(row['seed']):
            raise RuntimeError('Result identity mismatch')
        if report['status'] != 'artifact_complete' or report['rejected_training_steps']:
            raise RuntimeError('Training did not complete cleanly')
        if report['calibration_sha256'] != sha256(directory.parents[1] / 'preflight/calibration.pt'):
            raise RuntimeError('Result calibration SHA mismatch')
        for bank in report['metrics'].values():
            for key in ('phi_psi_tvd', 'torsion_js_mean', 'physical_energy_w2',
                        'coarse_basin_tvd', 'missing_reference_mass'):
                if not math.isfinite(bank[key]):
                    raise RuntimeError('Nonfinite evaluation metric: ' + key)
    log = Path(row['log_path']).read_text(errors='replace')
    if 'postflight_verified=true' not in log:
        raise RuntimeError('Missing worker postflight marker')
    return report


def submit(stage, candidates, campaign, plan, python, rows, dry_run=False):
    source_sha = sha256(ROOT / 'source_manifest.json')
    known = {(row['stage'], row['chart'], int(row['seed'])) for row in rows}
    new = []
    for chart, seed in candidates:
        if (stage, chart, seed) in known:
            continue
        run_directory = (campaign / 'preflight' if stage == 'preflight'
                         else campaign / stage / f'{chart}_seed{seed}')
        if (run_directory / 'result.json').exists() or (run_directory / 'receipt.json').exists():
            raise RuntimeError('Unledgered result exists; reconcile before submitting')
        label = {'preflight': 'prep', 'geometry_pilot': 'geom', 'geometry_confirmation': 'conf'}[stage]
        name = f'aldp-{label}-{chart[:3]}-s{seed}'
        log_template = campaign / 'logs' / f'{name}-%j.out'
        command = ['sbatch', '--parsable', f'--job-name={name}', '--partition=gpu', '--gres=gpu:1',
            '--cpus-per-task=4', '--mem=16G', '--time=02:00:00',
            '--exclude=htc-gpu024', f'--output={log_template}',
            str(ROOT / 'exec_aldp_chart_campaign.sh'), str(ROOT), str(python),
            str(campaign), stage, chart, str(seed)]
        if dry_run:
            new.append(dict(command=command, stage=stage, chart=chart, seed=seed))
            continue
        job = execute(command).stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unparseable sbatch job ID: ' + job)
        row = dict(job_id=job, stage=stage, chart=chart, seed=seed,
            run_directory=str(run_directory), log_path=str(log_template).replace('%j', job),
            repo=str(ROOT), python=str(python), source_sha256=source_sha,
            plan_sha256=sha256(campaign / 'plan.json'))
        ledger = campaign / 'submitted_jobs.tsv'
        new_file = not ledger.exists()
        with ledger.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t')
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        new.append(row)
    return new


def summarize_geometry(reports, campaign, plan):
    floor = json.loads((campaign / 'preflight/calibration_report.json').read_text())['reference_floor_q95']
    thresholds = {key: max(plan['geometry_gate']['minimum_thresholds'][key],
                           plan['geometry_gate']['floor_multiplier'] * floor[key])
                  for key in plan['geometry_gate']['minimum_thresholds']}
    cells = []
    for row in reports:
        if not row.get('stage', 'preflight').startswith('geometry'):
            continue
        quality = row['metrics']['a']
        passing = all(quality[key] <= limit for key, limit in thresholds.items())
        cells.append(dict(chart=row['chart'], seed=row['seed'], metrics=quality,
            pipeline_pass=True, pilot_quality_gate_pass=passing,
            cost=row['cost'], near_cut_fraction=row['near_cut_fraction_phi_psi']))
    admissible = []
    for chart in ('current', 'low_density'):
        matching = [row for row in cells if row['chart'] == chart]
        if len(matching) == 3 and all(row['pilot_quality_gate_pass'] for row in matching):
            admissible.append(chart)
    selected = min(admissible, key=lambda chart: sum(row['metrics']['phi_psi_tvd']
                    for row in cells if row['chart'] == chart)) if admissible else None
    summary = dict(schema='aldp_geometry_summary_v1', rows=cells, thresholds=thresholds,
        reference_floor_q95=floor, selected_chart=selected,
        status='pilot_geometry_admitted' if selected else 'geometry_not_admitted',
        scientific_status='sampling_convergence_not_evaluated',
        high_density_role='diagnostic_only_never_selected',
        next_action='freeze chart/source then solver-only ablation' if selected
                    else 'do not tune references; diagnose quality and cut sensitivity first')
    write_json(campaign / 'geometry_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--advance', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / 'plan.json').read_text())
    (campaign / 'logs').mkdir(exist_ok=True)
    with (campaign / 'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_manifest(campaign / 'submitted_jobs.tsv')
        for row in rows:
            if row['plan_sha256'] != sha256(campaign / 'plan.json'):
                raise RuntimeError('Campaign plan changed after submission')
        failures, reports, counts = [], [], {}
        for row in rows:
            state, exit_code = scheduler(row['job_id'])
            counts[state] = counts.get(state, 0) + 1
            if state == 'COMPLETED' and exit_code == '0:0':
                try:
                    reports.append(verify_receipt(Path(row['run_directory']), row))
                except Exception as error:
                    failures.append(dict(job_id=row['job_id'], reason=repr(error), log_path=row['log_path']))
            elif state not in ACTIVE and state != 'UNKNOWN':
                path = Path(row['log_path'])
                failures.append(dict(job_id=row['job_id'], reason=f'{state}/{exit_code}',
                    log_path=row['log_path'], log_tail='\n'.join(path.read_text(errors='replace').splitlines()[-35:])
                    if path.exists() else 'log missing'))
        def verified(stage):
            return sum(report.get('stage', 'preflight') == stage for report in reports)
        new, next_stage, summary = [], None, None
        if not rows:
            next_stage, candidates = 'preflight', [('current', 0)]
        elif verified('preflight') == 1 and not failures:
            if verified('geometry_pilot') < 3:
                next_stage, candidates = 'geometry_pilot', [(chart, 0) for chart in CHARTS]
            elif verified('geometry_confirmation') < 6:
                next_stage, candidates = 'geometry_confirmation', [(chart, seed) for chart in CHARTS for seed in (1, 2)]
            else:
                summary = summarize_geometry(reports, campaign, plan)
        if next_stage and args.advance:
            new = submit(next_stage, candidates, campaign, plan, args.python, rows, args.dry_run)
        previous_path = campaign / 'monitor_state.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        waiting = bool(rows) and not failures and not new and any(state in ACTIVE for state in counts)
        consecutive = previous.get('consecutive_pending_or_running', 0) + 1 if waiting else 0
        result = dict(counts=counts, submitted_jobs=len(rows) + len(new), verified_artifacts=len(reports),
            failures=failures, new_submissions=new, next_stage=next_stage,
            geometry_summary=summary, consecutive_pending_or_running=consecutive,
            recommended_monitor_minutes=30 if consecutive >= 2 else 10,
            complete=summary is not None, remote_root=str(campaign))
        if not args.dry_run:
            write_json(previous_path, result)
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
