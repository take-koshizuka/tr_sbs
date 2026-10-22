#!/usr/bin/env python3
"""Submit one bounded numerical diagnostic or take one Slurm snapshot."""
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


def submit(campaign,plan,python,source_sha):
    rows=read_manifest(campaign/'submitted_jobs.tsv')
    if rows:
        if len(rows)!=1:
            raise RuntimeError('One numerical diagnostic only')
        return rows[0]
    output=campaign/'numerical_diagnostic/seed0'
    intent=campaign/'submission_intents/numerical_diagnostic_seed0.json'
    if intent.exists() or (output.exists() and any(output.iterdir())):
        raise RuntimeError('Existing intent/output requires reconciliation; never submit twice')
    log=campaign/'logs/aldp-jax-diagnostic-%j.out';log.parent.mkdir(parents=True,exist_ok=True)
    command=['sbatch','--parsable','--job-name=aldp-jax-diagnostic','--partition=gpu',
        '--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=00:20:00',
        '--exclude=htc-gpu024',f'--output={log}',str(ROOT/'exec_aldp_jax_failure_diagnostic.sh'),
        str(ROOT),str(python),str(campaign)]
    write_json(intent,dict(status='dispatching',command=command,output=str(output)))
    job=execute(command).stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unparseable dispatch; reconcile intent')
    row=dict(job_id=job,stage='jax_numerical_diagnostic',candidate='failed_u0_replay',seed=0,
        run_directory=str(output),log_path=str(log).replace('%j',job),repo=str(ROOT),
        python=str(python),source_sha256=source_sha,plan_sha256=sha256(campaign/'plan.json'))
    write_json(intent,dict(status='submitted',command=command,row=row,
        submitted_at=datetime.now(timezone.utc).isoformat()))
    with (campaign/'submitted_jobs.tsv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,delimiter='\t')
        writer.writeheader();writer.writerow(row);stream.flush();os.fsync(stream.fileno())
    return row


def verify_completed(row,campaign,plan):
    output=Path(row['run_directory']);rec=json.loads((output/'receipt.json').read_text())
    if (rec.get('schema')!='aldp_chart_receipt_v1' or not rec.get('postflight_verified')
        or rec['identity']!=dict(stage='jax_numerical_diagnostic',chart='low_density',seed=0)
        or rec['inputs']!=dict(source_sha256=row['source_sha256'],plan_sha256=row['plan_sha256'])):
        raise RuntimeError('Invalid diagnostic receipt identity/provenance')
    for name,digest in rec['files'].items():
        if sha256(output/name)!=digest:
            raise RuntimeError('Diagnostic receipt SHA mismatch: '+name)
    report=json.loads((output/'result.json').read_text())
    if (report['source_sha256']!=row['source_sha256'] or report['plan_sha256']!=row['plan_sha256']
        or report['config_sha256']!=sha256(output/'config.yaml')
        or report['scientific_status']!='not_evaluated' or report['normal_reference_completed']
        or report['global_step']!=report['neural_updates']
        or report['global_step']>plan['replay_epoch_limit']*100
        or report['openmm_configurations']>plan['openmm_cap']
        or report['failed_job_id']!=plan['failed_job_id']):
        raise RuntimeError('Invalid bounded numerical diagnostic result')
    if report['status']=='numerical_rejection_captured':
        if 'rejected_batch.pt' not in rec['files'] or not report['diagnosis']:
            raise RuntimeError('Rejected batch/evidence missing')
    elif report['status']!='no_rejection_within_registered_replay_limit':
        raise RuntimeError('Unknown diagnostic outcome')
    if 'postflight_verified=true' not in Path(row['log_path']).read_text(errors='replace'):
        raise RuntimeError('Diagnostic worker completion marker absent')
    return report,len(rec['files'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--python',required=True)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();campaign=args.campaign.resolve()
    plan=json.loads((campaign/'plan.json').read_text())
    with (campaign/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source_sha=verify_source();check_files(plan['frozen_inputs'])
        if (source_sha!=plan['source_manifest_sha256'] or sha256(
                ROOT/'src/tr_asbs/experiment/jax_inner.py')!=plan['jax_backend_sha256']):
            raise RuntimeError('Diagnostic changes production JAX math/source')
        if args.submit:
            row=submit(campaign,plan,args.python,source_sha)
            result=dict(status='submitted',row=row,active_checks=0,
                next_check_interval_minutes=10,observed_scheduler_state=False)
        else:
            rows=read_manifest(campaign/'submitted_jobs.tsv')
            if len(rows)!=1:
                raise RuntimeError('One diagnostic ledger row required')
            row=rows[0]
            if row['source_sha256']!=source_sha or row['plan_sha256']!=sha256(campaign/'plan.json'):
                raise RuntimeError('Diagnostic ledger SHA mismatch')
            state,accounts=accounting(row['job_id'])
            old=campaign/'monitor_state.json';previous=json.loads(old.read_text()) if old.exists() else {}
            active=previous.get('active_checks',0)+1 if state in ACTIVE else 0
            result=dict(job_id=row['job_id'],scheduler_state=state,accounting=accounts,
                source_sha256=source_sha,plan_sha256=row['plan_sha256'],active_checks=active,
                next_check_interval_minutes=30 if active>=2 else 10)
            log=Path(row['log_path'])
            if log.exists():
                result.update(log_sha256=sha256(log),log_tail='\n'.join(log.read_text(errors='replace').splitlines()[-16:]))
            if state=='COMPLETED':
                if any(accounts.get(row['job_id']+suffix,{}).get('state')!='COMPLETED'
                    or accounts[row['job_id']+suffix]['exit_code']!='0:0' for suffix in ('','.batch','.extern')):
                    raise RuntimeError('All diagnostic accounting rows must complete exit0')
                report,count=verify_completed(row,campaign,plan)
                result.update(status='numerical_diagnostic_verified',receipt_files_verified=count,
                    result_sha256=sha256(Path(row['run_directory'])/'result.json'),report=report,
                    next_action='recover all evidence; distinguish shared numeric range from JAX-only failure before retry')
            elif state in ACTIVE:
                result['status']='active'
            else:
                result.update(status='failed_or_accounting_unknown',next_action='inspect logs; preserve failed output')
        result.update(checked_at=datetime.now(timezone.utc).isoformat(),quality='not_evaluated',
            minimum_inner_duration='not_established')
        write_json(campaign/'monitor_state.json',result)
        print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
