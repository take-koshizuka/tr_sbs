#!/usr/bin/env python3
"""Bounded normal-curve submission/Slurm snapshot; no job waits or tuning."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from monitor_aldp_chart_campaign import ACTIVE, execute, read_manifest
from monitor_aldp_jax_inner_campaign import FIELDS, accounting
from run_aldp_chart_campaign import ROOT, sha256, verify_source, write_json
from run_aldp_jax_inner_preflight import check_files


def verify_prerequisite(plan):
    check_files(plan['frozen_inputs'])
    proof = json.loads(Path(plan['backend_verification_path']).read_text())
    if proof.get('status') != 'backend_verified' or proof['job_id'] != plan['backend_job_id']:
        raise RuntimeError('Verified JAX backend proof required')
    if sha256(ROOT/'src/tr_asbs/experiment/jax_inner.py') != plan['jax_backend_sha256']:
        raise RuntimeError('JAX math/backend changed after GPU verification')


def submit(campaign, plan, python, source_sha):
    rows = read_manifest(campaign/'submitted_jobs.tsv')
    if rows:
        if len(rows) != 1:
            raise RuntimeError('One normal-duration reference only')
        return rows[0]
    output = campaign/'normal_inner_curve/seed0'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Unledgered output exists; reconcile before submission')
    intent = campaign/'submission_intents/normal_inner_curve_seed0.json'
    if intent.exists():
        raise RuntimeError('Preserve existing dispatch intent; never submit twice')
    log = campaign/'logs/aldp-jax-curve-%j.out'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = ['sbatch','--parsable','--job-name=aldp-jax-curve','--partition=gpu',
        '--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=06:00:00',
        '--exclude=htc-gpu024',f'--output={log}',str(ROOT/'scripts/slurm/exec_aldp_jax_inner_curve.sh'),
        str(ROOT),str(python),str(campaign)]
    write_json(intent, dict(status='dispatching', command=command, output=str(output)))
    response = execute(command).stdout.strip().split(';')[0]
    if not response.isdigit():
        raise RuntimeError('Unparseable job ID; reconcile intent instead of retrying')
    row = dict(job_id=response,stage='normal_inner_curve',candidate='normal',seed=0,
        run_directory=str(output),log_path=str(log).replace('%j',response),repo=str(ROOT),
        python=str(python),source_sha256=source_sha,plan_sha256=sha256(campaign/'plan.json'))
    write_json(intent, dict(status='submitted',row=row,command=command,
        submitted_at=datetime.now(timezone.utc).isoformat()))
    with (campaign/'submitted_jobs.tsv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,delimiter='\t')
        writer.writeheader();writer.writerow(row);stream.flush();os.fsync(stream.fileno())
    return row


def verify_completed(row, campaign):
    output = Path(row['run_directory'])
    rec = json.loads((output/'receipt.json').read_text())
    if rec.get('schema') != 'aldp_chart_receipt_v1' or not rec.get('postflight_verified'):
        raise RuntimeError('Invalid normal-curve receipt')
    plan = json.loads((campaign/'plan.json').read_text())
    if rec['identity'] != dict(stage='normal_inner_curve',chart='low_density',seed=0):
        raise RuntimeError('Normal-curve receipt identity mismatch')
    provenance=rec['inputs']['provenance']
    if (rec['inputs']['source_sha256'] != row['source_sha256']
        or provenance['source_sha256'] != row['source_sha256']
        or provenance['plan_sha256'] != row['plan_sha256']):
        raise RuntimeError('Normal-curve provenance mismatch')
    for name,digest in rec['files'].items():
        if sha256(output/name) != digest:
            raise RuntimeError('Normal-curve receipt SHA mismatch: '+name)
    result=json.loads((output/'result.json').read_text())
    if (result['source_sha256'] != row['source_sha256'] or result['global_step'] != 300000
        or result['scientific_status'] != 'not_evaluated'
        or result['minimum_inner_duration'] != 'not_established'
        or result['config_sha256'] != sha256(output/'config.yaml')
        or result['provenance'] != provenance):
        raise RuntimeError('Incomplete normal reference or invalid selection claim')
    if result['diagnostics']['backend_adam_update_counts'] != {'u':200000,'h':100000}:
        raise RuntimeError('Wrong Adam phase update counts')
    expected=[(p,e) for p in ('u','h') for e in [0]+plan['curve_contract']['milestones'][p]]
    boundaries=result['diagnostics']['inner_boundaries']
    if [(r['phase'],r['phase_epoch']) for r in boundaries] != expected:
        raise RuntimeError('Missing/reordered inner boundaries')
    for r in boundaries:
        epoch=r['phase_epoch']+(2000 if r['phase']=='h' else 0)
        if r['epoch'] != epoch or r['global_step'] != 100*epoch or r['snapshot'] not in rec['files']:
            raise RuntimeError('Invalid exact prefix checkpoint identity/count')
    if result['cost']['total_openmm_configurations'] != 285376:
        raise RuntimeError('Normal reference cost boundary changed')
    if 'postflight_verified=true' not in Path(row['log_path']).read_text(errors='replace'):
        raise RuntimeError('Normal worker completion marker absent')
    return result,len(rec['files'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--python',required=True)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    plan=json.loads((campaign/'plan.json').read_text())
    with (campaign/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source_sha=verify_source()
        if source_sha != plan['source_manifest_sha256']:
            raise RuntimeError('Normal reference source snapshot changed')
        verify_prerequisite(plan)
        if args.submit:
            row=submit(campaign,plan,args.python,source_sha)
            result=dict(status='submitted',row=row,observed_scheduler_state=False,
                active_checks=0,next_check_interval_minutes=10,quality='not_evaluated',
                minimum_inner_duration='not_established')
        else:
            rows=read_manifest(campaign/'submitted_jobs.tsv')
            if len(rows) != 1:
                raise RuntimeError('One normal curve ledger required')
            row=rows[0]
            if row['source_sha256'] != source_sha or row['plan_sha256'] != sha256(campaign/'plan.json'):
                raise RuntimeError('Normal curve ledger SHA mismatch')
            state,accounts=accounting(row['job_id'])
            old=campaign/'monitor_state.json'
            previous=json.loads(old.read_text()) if old.exists() else {}
            active=previous.get('active_checks',0)+1 if state in ACTIVE else 0
            result=dict(job_id=row['job_id'],scheduler_state=state,accounting=accounts,
                source_sha256=source_sha,plan_sha256=row['plan_sha256'],active_checks=active,
                next_check_interval_minutes=30 if active>=2 else 10,
                quality='not_evaluated',minimum_inner_duration='not_established')
            logfile=Path(row['log_path'])
            if logfile.exists():
                result.update(log_sha256=sha256(logfile),log_tail='\n'.join(logfile.read_text(errors='replace').splitlines()[-16:]))
            progress=Path(row['run_directory'])/'learning_curve_progress.json'
            if progress.exists():
                info=json.loads(progress.read_text())
                result['completed_boundaries']=[{k:r[k] for k in ('phase','phase_epoch','epoch','global_step')}
                                               for r in info['boundaries']]
                result['current_openmm_configurations']=info['current_openmm_configurations']
            if state=='COMPLETED':
                keys=[row['job_id']+suffix for suffix in ('','.batch','.extern')]
                if any(accounts.get(k,{}).get('state')!='COMPLETED'
                       or accounts[k]['exit_code']!='0:0' for k in keys):
                    raise RuntimeError('All root/batch/extern must complete with exit0')
                report,files=verify_completed(row,campaign)
                result.update(status='normal_curve_verified',receipt_files_verified=files,
                    cost=report['cost'],metrics=report['metrics'],
                    result_sha256=sha256(Path(row['run_directory'])/'result.json'),
                    next_action='A-only prefix screening; implement limited shorter candidates and paired next-outer confirmation')
            elif state in ACTIVE:
                result['status']='active'
            else:
                result.update(status='failed_or_accounting_unknown',
                    next_action='inspect failure/log; only simple implementation bugs may retry in new snapshots')
        result['checked_at']=datetime.now(timezone.utc).isoformat()
        write_json(campaign/'monitor_state.json',result)
        print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
