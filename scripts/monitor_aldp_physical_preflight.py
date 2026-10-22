#!/usr/bin/env python3
"""One-shot physical preflight accounting/log/hash audit, without dispatching work."""

import argparse
import fcntl
import json
from pathlib import Path

from monitor_aldp_chart_campaign import ACTIVE, read_manifest, scheduler, verify_receipt
from run_aldp_chart_campaign import ROOT, sha256, verify_source, write_json


def verify(row, campaign, plan):
    directory = Path(row['run_directory'])
    document = json.loads((directory / 'receipt.json').read_text())
    if (document.get('schema') != 'aldp_chart_receipt_v1'
            or not document.get('postflight_verified')
            or document['identity'] != dict(stage='physical_preflight')
            or document['inputs']['source_sha256'] != row['source_sha256']
            or document['inputs']['plan_sha256'] != row['plan_sha256']
            or document['inputs']['legacy_calibration_sha256'] != plan['legacy_calibration_sha256']):
        raise RuntimeError('Physical preflight receipt provenance mismatch')
    for name, expected in document['files'].items():
        if sha256(directory / name) != expected:
            raise RuntimeError('Physical preflight file SHA mismatch: ' + name)
    if 'postflight_verified=true' not in Path(row['log_path']).read_text(errors='replace'):
        raise RuntimeError('Missing worker postflight marker')
    report = json.loads((directory / 'preflight_result.json').read_text())
    audit_path = directory / 'density_audit.json'
    audit = json.loads(audit_path.read_text())
    if (report['status'] != 'passed' or audit['status'] != 'passed'
            or report['source_sha256'] != row['source_sha256']
            or report['plan_sha256'] != row['plan_sha256']
            or audit['plan_sha256'] != row['plan_sha256']
            or audit['source_sha256'] != row['source_sha256']
            or report['calibration_sha256'] != sha256(directory / 'calibration.pt')
            or report['density_audit_sha256'] != sha256(audit_path)
            or audit['calibration_sha256'] != report['calibration_sha256']
            or audit['energy_definition']['mode'] != 'physical'):
        raise RuntimeError('Physical preflight validation mismatch')
    smoke = Path(report['smoke_directory'])
    if smoke != campaign / 'smoke/current_seed0':
        raise RuntimeError('Unexpected smoke output path')
    for name, key in [('result.json', 'result_sha256'), ('receipt.json', 'receipt_sha256')]:
        if sha256(smoke / name) != report[key]:
            raise RuntimeError('Physical smoke SHA mismatch')
    smoke_row = dict(row, stage='smoke', chart='current', seed='0')
    result = verify_receipt(smoke, smoke_row)
    if (result['sampling_metadata']['data_provenance']['energy_definition'] != audit['energy_definition']
            or result['config_sha256'] != sha256(smoke / 'config.yaml')
            or result['global_step'] != 2 or result['native_cartesian_roundtrip_error'] > 2e-5
            or result['provenance']['plan_sha256'] != row['plan_sha256']):
        raise RuntimeError('Physical smoke scientific configuration mismatch')
    return dict(report=report, audit=audit, smoke=result,
                verified_files=len(document['files']) + len(json.loads((smoke / 'receipt.json').read_text())['files']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    with (campaign / '.monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_sha = verify_source()
        plan_path = campaign / 'plan.json'
        plan = json.loads(plan_path.read_text())
        rows = read_manifest(campaign / 'submitted_jobs.tsv')
        if len(rows) != 1:
            raise RuntimeError('Expected exactly one physical preflight; reconcile ledger')
        row = rows[0]
        if (row['source_sha256'] != source_sha or row['plan_sha256'] != sha256(plan_path)
                or row['repo'] != str(ROOT) or row['stage'] != 'physical_preflight'):
            raise RuntimeError('Physical ledger source/plan mismatch')
        state, exit_code = scheduler(row['job_id'])
        item = dict(row, state=state, exit_code=exit_code, verified=False)
        if state == 'COMPLETED' and exit_code == '0:0':
            item['verification'] = verify(row, campaign, plan)
            item['verified'] = True
        elif state not in ACTIVE:
            log = Path(row['log_path'])
            item['failure_log_tail'] = '\n'.join(log.read_text(errors='replace').splitlines()[-35:]) if log.exists() else 'Log not available'
        previous_path = campaign / 'monitor_state.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        streak = previous.get('consecutive_active_checks', 0) + 1 if state in ACTIVE else 0
        summary = dict(status='physical_preflight_snapshot', job=item,
            consecutive_active_checks=streak, recommended_interval_minutes=30 if streak >= 2 else 10,
            scientific_status='execution_gate_only_not_reference_or_geometry_admission',
            next_action='review matched legacy diagnosis and design a physical-density pilot only after gate passes')
        write_json(campaign / 'monitor_state.json', summary)
        print(json.dumps(dict(job_id=row['job_id'], state=state, exit_code=exit_code,
            verified=item['verified'], consecutive_active_checks=streak,
            recommended_interval_minutes=summary['recommended_interval_minutes'])))


if __name__ == '__main__':
    main()
