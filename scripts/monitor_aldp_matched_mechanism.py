#!/usr/bin/env python3
"""Reuse a fully verified unchanged preflight for the matched-budget extension."""

from __future__ import annotations

import json
from pathlib import Path

import monitor_aldp_mechanism_campaign as monitor
from run_aldp_chart_campaign import sha256, write_json
from run_aldp_mechanism_campaign import configuration

_read_manifest = monitor.read_manifest
_verify_job = monitor.verify_job


def smoke_equivalence(original_plan, new_plan, original_campaign, new_campaign):
    """Only output locations differ in reused tiny execution configurations."""
    from omegaconf import OmegaConf
    if not set(new_plan['candidates']).issubset(original_plan['candidates']):
        raise ValueError('New candidate requires a new preflight')
    signatures = {}
    for candidate in new_plan['candidates']:
        normalized = []
        for plan, campaign in [(original_plan, original_campaign), (new_plan, new_campaign)]:
            cfg, _ = configuration(plan, Path(campaign), 'mechanism_smoke', candidate, 0)
            data = OmegaConf.to_container(cfg, resolve=True)
            for key in ('name', 'root', 'checkpoint_directory', 'sample_directory'):
                data.pop(key)
            data['fabric']['loggers']['save_dir'] = 'OUTPUT'
            normalized.append(data)
        if normalized[0] != normalized[1]:
            raise ValueError('Smoke scientific/runtime settings differ: ' + candidate)
        signatures[candidate] = normalized[0]
    return signatures


def read_manifest(path):
    rows = _read_manifest(path)
    for row in rows:
        if row['stage'] == 'preflight_reused':
            # The on-disk ledger retains the original job's real plan SHA and
            # explicitly labels this as reused evidence, never a new submission.
            row['stage'], row['_preflight_reused'] = 'preflight', True
    return rows


def verify_job(row, campaign, plan):
    if sha256(Path(__file__)) != plan['matched_monitor_sha256']:
        raise RuntimeError('Matched monitor SHA mismatch')
    if not row.get('_preflight_reused'):
        return _verify_job(row, campaign, plan)
    evidence = plan['reuse_preflight']
    original = Path(evidence['original_campaign'])
    original_plan = json.loads((original / 'plan.json').read_text())
    original_plan['_plan_sha256'] = sha256(original / 'plan.json')
    if original_plan['_plan_sha256'] != evidence['original_plan_sha256']:
        raise RuntimeError('Reused preflight original plan changed')
    if sha256(original / 'submitted_jobs.tsv') != evidence['original_ledger_sha256']:
        raise RuntimeError('Reused preflight original ledger changed')
    original_row = next(r for r in _read_manifest(original / 'submitted_jobs.tsv')
                        if r['job_id'] == evidence['job_id'] and r['stage'] == 'preflight')
    for key in monitor.FIELDS:
        if key != 'stage' and str(row[key]) != str(original_row[key]):
            raise RuntimeError('Reused ledger changed original job provenance: ' + key)
    if row['source_sha256'] != sha256(monitor.ROOT / 'source_manifest.json'):
        raise RuntimeError('Changed source requires a fresh preflight')
    if sha256(original / 'preflight/preflight_result.json') != evidence['preflight_result_sha256']:
        raise RuntimeError('Reused preflight result changed')
    if plan['calibration_sha256'] != original_plan['calibration_sha256']:
        raise RuntimeError('Changed calibration requires a fresh preflight')
    report = _verify_job(original_row, original, original_plan)
    signatures = smoke_equivalence(original_plan, plan, original, campaign)
    write_json(campaign / 'reused_preflight_verification.json',
        dict(schema='aldp_reused_preflight_verification_v1', status='passed',
             original_job_id=row['job_id'], original_plan_sha256=original_plan['_plan_sha256'],
             current_plan_sha256=plan['_plan_sha256'], source_sha256=row['source_sha256'],
             original_receipt_sha256=sha256(original / 'preflight/receipt.json'),
             exact_smoke_signature_matches=list(signatures), original_report=report,
             rule='same source SHA and all smoke settings except output locations; all original receipts verified'))
    return report


if __name__ == '__main__':
    monitor.read_manifest, monitor.verify_job = read_manifest, verify_job
    monitor.main()
