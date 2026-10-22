#!/usr/bin/env python3
"""One registered h-duration follow-up with an immutable cached normal control.

Reuses the existing Slurm worker, exact-prefix adapter, and comparison limits.
No u/h pair is proposed after the failed u1500 next-outer confirmation.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path


def choose_h_candidate(screening, attempts):
    if screening['decision_bank'] != 'a' or screening['test'] != 'closed':
        raise ValueError('A-only screening and closed test required')
    rows = screening['phases']['h']['comparisons']
    epochs = [row['phase_epoch'] for row in rows]
    if epochs != [125, 250, 500, 750, 1000]:
        raise ValueError('Use the original registered h grid')
    tried = [row['h_epochs'] for row in attempts]
    if len(tried) != len(set(tried)) or any(epoch not in epochs[:-1] for epoch in tried):
        raise ValueError('Duplicate or unregistered prior h attempts')
    if any(row['conditional_gates_passed'] for row in attempts):
        raise ValueError('A shorter duration already passed; no further follow-up')
    return next((row['phase_epoch'] for row in rows if row['prefix_gates_passed']
                 and row['phase_epoch'] < 1000 and row['phase_epoch'] not in tried), None)


def compare_cached_reference(fresh, normal, plan):
    from run_aldp_inner_continuation import compare_one_axis
    if fresh['source_sha256'] != plan['source_manifest_sha256']:
        raise ValueError('Fresh result source differs from the registered follow-up')
    path = plan['comparison_h_validation_path']
    for report in (fresh, normal):
        for key in ('first_outer_validation', 'next_outer_validation'):
            value = report['diagnostics'][key]
            if (value['batch_path'] != path or value['sample_count'] != 256
                    or value['bank'] != 'identical normal-outer1 frozen endpoints/bridge draws'):
                raise ValueError('Identical fixed normal BC bank required')
    comparison_plan = deepcopy(plan)
    comparison_plan['cases']['normal'] = deepcopy(plan['normal_cache']['case'])
    result = compare_one_axis({'normal': normal, 'h_short': fresh}, comparison_plan)
    result.update(cached_normal=True, cached_normal_job_id=plan['normal_cache']['row']['job_id'],
                  retained_u_epochs=2000, compared_h_epochs=plan['cases']['h_short']['h_epochs'],
                  no_paired_confirmation=True,
                  prior_study_paid_openmm_configurations=plan['prior_study_paid_openmm_configurations'],
                  new_study_paid_openmm_configurations=plan['prior_study_paid_openmm_configurations']
                  + fresh['cost']['total_openmm_configurations'])
    return result


def prepare(parent_campaign, campaign, source_sha):
    from analyze_aldp_inner_curve import screen
    from monitor_aldp_inner_continuation import checked_ledger, verify_completed
    from monitor_aldp_jax_inner_curve import verify_prerequisite
    from run_aldp_chart_campaign import ROOT, sha256, write_json
    from run_aldp_inner_continuation import configuration, expected_cost, verify_plan
    parent = json.loads((parent_campaign / 'plan.json').read_text())
    audit_path = parent_campaign / 'monitor_state.json'
    audit = json.loads(audit_path.read_text())
    comparison_path = parent_campaign / 'comparison.json'
    decision = json.loads(comparison_path.read_text())
    if (audit['status'] != 'continuations_verified'
            or audit['source_sha256'] != parent['source_manifest_sha256']
            or audit['plan_sha256'] != sha256(parent_campaign / 'plan.json')
            or audit['comparison_sha256'] != sha256(comparison_path)):
        raise ValueError('Fully verified parent comparison required')
    verify_prerequisite(parent)
    rows = checked_ledger(parent_campaign, parent, parent['source_manifest_sha256'])
    old_manifest_path = Path(rows[0]['repo']) / 'source_manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    new_manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    if (sha256(old_manifest_path) != parent['source_manifest_sha256']
            or any(new_manifest['files'].get(name) != digest
                   for name, digest in old_manifest['files'].items())):
        raise ValueError('Follow-up must preserve all previously verified source files')
    reports = {}
    for row in rows:
        obs = next(item for item in audit['observations'] if item['job_id'] == row['job_id'])
        if obs['status'] != 'receipt_verified':
            raise ValueError('Audited completed parent case required')
        for suffix in ('', '.batch', '.extern'):
            account = obs['accounting'].get(row['job_id'] + suffix, {})
            if account.get('state') != 'COMPLETED' or account.get('exit_code') != '0:0':
                raise ValueError('All parent accounting must complete0:0')
        reports[row['candidate']], _ = verify_completed(row, parent_campaign, parent)
        if sha256(Path(row['run_directory']) / 'result.json') != audit['results_sha256'][row['candidate']]:
            raise ValueError('Parent result differs from its completion audit')
    if parent.get('followup_type') == 'h_duration':
        normal_cache = deepcopy(parent['normal_cache'])
        attempts = deepcopy(parent['h_duration_attempts'])
        prior_paid = decision['new_study_paid_openmm_configurations']
    else:
        if decision['candidates']['u_short']['conditional_gates_passed']:
            raise ValueError('This fallback is restricted to the failed u-short confirmation')
        normal_row = next(row for row in rows if row['candidate'] == 'normal')
        normal_cache = dict(row=normal_row, case=parent['cases']['normal'],
                            campaign=str(parent_campaign), plan_path=str(parent_campaign / 'plan.json'))
        attempts = []
        prior_paid = decision['paid_study_openmm_configurations']
    attempts.append(dict(h_epochs=parent['cases']['h_short']['h_epochs'],
                         conditional_gates_passed=decision['candidates']['h_short']['conditional_gates_passed'],
                         comparison_sha256=sha256(comparison_path)))
    normal_report = json.loads(Path(parent['normal_report_path']).read_text())
    screening = screen(normal_report, parent['curve_contract'])
    epoch = choose_h_candidate(screening, attempts)
    if epoch is None:
        raise ValueError('No untested shorter h prefixes passed; retain the normal duration')
    point = next(row for row in normal_report['diagnostics']['inner_boundaries']
                 if row['phase'] == 'h' and row['phase_epoch'] == epoch)
    initial_run = Path(parent['normal_report_path']).parent
    existing_path = campaign / 'plan.json'
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing.get('schema') != 'aldp_h_duration_pending_v1':
            raise RuntimeError('Prepared plan exists; never overwrite its registration')
    if (campaign / 'submitted_jobs.tsv').exists() or (campaign / 'submission_intents').exists():
        raise RuntimeError('Existing dispatch state; never prepare over submitted jobs')
    audit_copy = campaign / 'prior_completion_audit.json'
    if audit_copy.exists():
        raise RuntimeError('Existing audit copy; reconcile preparation before retry')
    audit_copy.parent.mkdir(parents=True, exist_ok=True)
    audit_copy.write_bytes(audit_path.read_bytes())
    frozen = dict(parent['frozen_inputs'])
    frozen[str(audit_copy)] = sha256(audit_copy)
    for path in (parent_campaign / 'plan.json', comparison_path,
                 parent_campaign / 'submitted_jobs.tsv', old_manifest_path):
        frozen[str(path)] = sha256(path)
    for row in rows:
        out = Path(row['run_directory'])
        rec = json.loads((out / 'receipt.json').read_text())
        frozen.update({str(out / name): digest for name, digest in rec['files'].items()})
        for path in (out / 'receipt.json', Path(row['log_path'])):
            frozen[str(path)] = sha256(path)
    case = dict(u_epochs=2000, h_epochs=epoch, outer_loops=2, seed=0, saved_phase='h',
                saved_phase_epoch=epoch, checkpoint_path=str(initial_run / point['snapshot']),
                cached_global_step=100 * (2000 + epoch), expected_provenance=normal_report['provenance'],
                parent_boundary=point, reuse_first_outer_policy=True,
                first_h_validation_path=str(initial_run / 'heldout_h.pt'), save_first_h_prefix=None)
    plan = deepcopy(parent)
    plan.update(cases={'h_short': case}, followup_type='h_duration', normal_cache=normal_cache,
                h_duration_attempts=attempts, retained_u_epochs=2000, source_manifest_sha256=source_sha,
                frozen_inputs=frozen, remote_root=str(campaign.parent), remote_code=str(ROOT),
                prior_study_paid_openmm_configurations=prior_paid, quality='not_evaluated', minimum='not_established')
    verify_plan(plan)
    write_json(existing_path, plan)
    cfg, _ = configuration(plan, campaign, 'h_short')
    from omegaconf import OmegaConf
    planned = campaign / 'planned_configs/h_short.yaml'
    planned.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, planned)
    cost = expected_cost(cfg, case)
    if cost['total_openmm_configurations'] > plan['openmm_cap']:
        raise ValueError('Registered marginal molecular cost exceeds its cap')
    write_json(campaign / 'preparation_verification.json', dict(status='prepared_no_jobs',
               source_sha256=source_sha, plan_sha256=sha256(existing_path), h_epochs=epoch,
               config_sha256=sha256(planned), expected_cost=cost, cached_normal_retraining=False,
               prior_study_paid_openmm_configurations=prior_paid, test='closed'))
    return plan


def main():
    from monitor_aldp_chart_campaign import ACTIVE
    from monitor_aldp_inner_continuation import checked_ledger, submit_cases, verify_completed
    from monitor_aldp_jax_inner_campaign import accounting
    from monitor_aldp_jax_inner_curve import verify_prerequisite
    from run_aldp_chart_campaign import sha256, verify_source, write_json
    from run_aldp_inner_continuation import verify_plan
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--python', required=True)
    parser.add_argument('--prepare-from', type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.prepare_from and args.submit:
        parser.error('Preparation and submission are separate operations')
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = verify_source()
        if args.prepare_from:
            plan = prepare(args.prepare_from.resolve(), campaign, source)
            print(json.dumps(dict(status='prepared_no_jobs', h_epochs=plan['cases']['h_short']['h_epochs'])))
            return
        plan = json.loads((campaign / 'plan.json').read_text())
        verify_plan(plan)
        if plan.get('followup_type') != 'h_duration' or set(plan['cases']) != {'h_short'} or source != plan['source_manifest_sha256']:
            raise RuntimeError('One registered fresh h case with an immutable cached control required')
        verify_prerequisite(plan)
        if args.submit:
            rows = submit_cases(campaign, plan, args.python, source)
            result = dict(status='submitted', rows=rows, observed_scheduler_state=False,
                          active_checks=0, next_check_interval_minutes=10)
        else:
            rows = checked_ledger(campaign, plan, source)
            if len(rows) != 1:
                raise RuntimeError('One fresh h duration only; no duplicated cached control')
            row = rows[0]
            state, accounts = accounting(row['job_id'])
            obs = dict(job_id=row['job_id'], candidate='h_short', scheduler_state=state, accounting=accounts)
            log = Path(row['log_path'])
            if log.exists():
                obs.update(log_sha256=sha256(log), log_tail='\n'.join(log.read_text(errors='replace').splitlines()[-16:]))
            previous = campaign / 'monitor_state.json'
            old = json.loads(previous.read_text()) if previous.exists() else {}
            checks = old.get('active_checks', 0) + 1 if state in ACTIVE else 0
            result = dict(status='active' if state in ACTIVE else 'failed_or_accounting_unknown',
                          observations=[obs], active_checks=checks, next_check_interval_minutes=30 if checks >= 2 else 10)
            obs['status'] = result['status']
            if state == 'COMPLETED':
                for suffix in ('', '.batch', '.extern'):
                    account = accounts.get(row['job_id'] + suffix, {})
                    if account.get('state') != 'COMPLETED' or account.get('exit_code') != '0:0':
                        raise RuntimeError('All fresh accounting must complete0:0')
                fresh, count = verify_completed(row, campaign, plan)
                cache = plan['normal_cache']
                cached_plan = json.loads(Path(cache['plan_path']).read_text())
                normal, _ = verify_completed(cache['row'], Path(cache['campaign']), cached_plan)
                comparison = compare_cached_reference(fresh, normal, plan)
                write_json(campaign / 'comparison.json', comparison)
                digest = sha256(Path(row['run_directory']) / 'result.json')
                obs.update(status='receipt_verified', receipt_files_verified=count, result_sha256=digest)
                passed = comparison['candidates']['h_short']['conditional_gates_passed']
                result.update(status='continuations_verified', results_sha256={'h_short': digest},
                              comparison_sha256=sha256(campaign / 'comparison.json'),
                              cached_normal_result_sha256=sha256(Path(cache['row']['run_directory']) / 'result.json'),
                              next_action='h_duration_conditional_pass' if passed else 'h_duration_failed_next_registered_prefix')
        result.update(checked_at=datetime.now(timezone.utc).isoformat(), source_sha256=source,
                      plan_sha256=sha256(campaign / 'plan.json'), quality='not_evaluated',
                      minimum_inner_duration='not_established')
        write_json(campaign / 'monitor_state.json', result)
        print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
