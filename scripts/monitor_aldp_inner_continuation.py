#!/usr/bin/env python3
"""Bounded ledger/Slurm monitor for registered exact-prefix continuations."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path

from monitor_aldp_chart_campaign import ACTIVE,execute,read_manifest
from monitor_aldp_jax_inner_campaign import FIELDS,accounting
from monitor_aldp_jax_inner_curve import verify_prerequisite
from run_aldp_chart_campaign import ROOT,sha256,verify_source,write_json
from run_aldp_inner_continuation import (
    compare_one_axis,compare_paired,configuration,expected_cost,verify_plan,
)


def checked_ledger(campaign,plan,source_sha):
    rows=read_manifest(campaign/'submitted_jobs.tsv')
    names=[r['candidate'] for r in rows];jobs=[r['job_id'] for r in rows]
    if len(set(names))!=len(names) or len(set(jobs))!=len(jobs) or not set(names)<=set(plan['cases']):
        raise RuntimeError('Duplicate/foreign continuation ledger entry')
    for row in rows:
        if (row['source_sha256']!=source_sha or row['plan_sha256']!=sha256(campaign/'plan.json')
            or row['stage']!='continuation' or row['seed']!='0'
            or row['run_directory']!=str(campaign/'continuation'/row['candidate'])):
            raise RuntimeError('Continuation ledger identity/SHA mismatch')
    return rows


def submit_cases(campaign,plan,python,source_sha):
    rows=checked_ledger(campaign,plan,source_sha)
    existing={row['candidate'] for row in rows}
    for name in plan['cases']:
        if name in existing:continue
        output=campaign/'continuation'/name
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Unledgered continuation output; reconcile before dispatch')
        intent=campaign/'submission_intents'/f'{name}_seed0.json'
        if intent.exists():raise RuntimeError('Existing dispatch intent; never submit twice')
        log=campaign/'logs'/f'aldp-inner-{name}-%j.out';log.parent.mkdir(parents=True,exist_ok=True)
        command=['sbatch','--parsable',f'--job-name=aldp-inner-{name}','--partition=gpu',
            '--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=06:00:00',
            '--exclude=htc-gpu024',f'--output={log}',str(ROOT/'scripts/slurm/exec_aldp_inner_continuation.sh'),
            str(ROOT),str(python),str(campaign),name]
        write_json(intent,dict(status='dispatching',candidate=name,command=command,output=str(output)))
        job=execute(command).stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Unknown dispatch outcome; reconcile intent before retry')
        row=dict(job_id=job,stage='continuation',candidate=name,seed=0,run_directory=str(output),
            log_path=str(log).replace('%j',job),repo=str(ROOT),python=str(python),
            source_sha256=source_sha,plan_sha256=sha256(campaign/'plan.json'))
        write_json(intent,dict(status='submitted',row=row,command=command,
            submitted_at=datetime.now(timezone.utc).isoformat()))
        ledger=campaign/'submitted_jobs.tsv';exists=ledger.exists()
        with ledger.open('a' if exists else 'x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,delimiter='\t')
            if not exists:writer.writeheader()
            writer.writerow(row);stream.flush();os.fsync(stream.fileno())
        rows.append(row)
    return rows


def verify_completed(row,campaign,plan):
    out=Path(row['run_directory']);rec=json.loads((out/'receipt.json').read_text())
    name=row['candidate'];case=plan['cases'][name]
    if (rec.get('schema')!='aldp_chart_receipt_v1' or not rec.get('postflight_verified')
        or rec['identity']!=dict(stage='continuation',candidate=name,seed=0)):
        raise RuntimeError('Invalid continuation receipt identity')
    provenance=rec['inputs']['provenance']
    if (rec['inputs']['source_sha256']!=row['source_sha256'] or provenance['source_sha256']!=row['source_sha256']
        or provenance['plan_sha256']!=row['plan_sha256']
        or provenance['parent_snapshot_sha256']!=sha256(case['checkpoint_path'])):
        raise RuntimeError('Continuation parent/source/plan provenance mismatch')
    for filename,digest in rec['files'].items():
        path=Path(filename)
        if path.is_absolute() or '..' in path.parts or sha256(out/path)!=digest:
            raise RuntimeError('Continuation receipt path/SHA mismatch')
    report=json.loads((out/'result.json').read_text());cfg,_=configuration(plan,campaign,name)
    cost=expected_cost(cfg,case);diagnostic=report['diagnostics']
    if (report['candidate']!=name or report['provenance']!=provenance
        or report['source_sha256']!=row['source_sha256'] or report['config_sha256']!=sha256(out/'config.yaml')
        or report['global_step']!=cost['final_optimizer_steps'] or report['rejected_training_steps']
        or report['minimum_inner_duration']!='not_established' or report['scientific_status']!='not_evaluated'
        or report['expected_marginal_cost']!=cost):
        raise RuntimeError('Incomplete continuation or premature scientific claim')
    if (diagnostic['backend_adam_update_counts']!={'u':cost['u_adam_steps'],'h':cost['h_adam_steps']}
        or not diagnostic['resume_marker']['restored'] or not diagnostic['resume_marker']['initial_online_suppressed']):
        raise RuntimeError('Wrong optimizer/RNG continuation boundary')
    for key,value in cost.items():
        if key.endswith('configurations') and report['cost'][key]!=value:
            raise RuntimeError('Marginal cost differs from registered counter boundary')
    for key,validation in (('first_outer_complete','first_outer_validation'),('trained_policy','next_outer_validation')):
        if diagnostic[key]['physical_energy']['count']!=1024 or diagnostic[validation]['sample_count']!=256:
            raise RuntimeError('Incomplete fixed-bank quality/BC sample counts')
    if 'postflight_verified=true' not in Path(row['log_path']).read_text(errors='replace'):
        raise RuntimeError('Missing continuation worker marker')
    return report,len(rec['files'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--python',required=True)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();campaign=args.campaign.resolve()
    plan=json.loads((campaign/'plan.json').read_text());verify_plan(plan)
    with (campaign/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source_sha=verify_source()
        if source_sha!=plan['source_manifest_sha256']:raise RuntimeError('Continuation source changed')
        verify_prerequisite(plan)
        if args.submit:
            rows=submit_cases(campaign,plan,args.python,source_sha)
            result=dict(status='submitted',rows=rows,observed_scheduler_state=False,active_checks=0,
                next_check_interval_minutes=10)
        else:
            rows=checked_ledger(campaign,plan,source_sha)
            if {r['candidate'] for r in rows}!=set(plan['cases']):
                raise RuntimeError('Incomplete dispatch ledger; reconcile before monitoring')
            reports={};observations=[];failed=False;active=False
            for row in rows:
                state,accounts=accounting(row['job_id'])
                item=dict(job_id=row['job_id'],candidate=row['candidate'],scheduler_state=state,accounting=accounts)
                logfile=Path(row['log_path'])
                if logfile.exists():item.update(log_sha256=sha256(logfile),
                    log_tail='\n'.join(logfile.read_text(errors='replace').splitlines()[-16:]))
                if state=='COMPLETED':
                    if any(accounts.get(row['job_id']+suffix,{}).get('state')!='COMPLETED'
                        or accounts[row['job_id']+suffix]['exit_code']!='0:0' for suffix in ('','.batch','.extern')):
                        raise RuntimeError('All continuation root/batch/extern must complete0:0')
                    report,count=verify_completed(row,campaign,plan)
                    reports[row['candidate']]=report
                    item.update(status='receipt_verified',receipt_files_verified=count,
                        result_sha256=sha256(Path(row['run_directory'])/'result.json'))
                elif state in ACTIVE:active=True;item['status']='active'
                else:failed=True;item['status']='failed_or_accounting_unknown'
                observations.append(item)
            old=campaign/'monitor_state.json';previous=json.loads(old.read_text()) if old.exists() else {}
            checks=previous.get('active_checks',0)+1 if active and not failed else 0
            result=dict(status='failed_or_accounting_unknown' if failed else 'active',observations=observations,
                active_checks=checks,next_check_interval_minutes=30 if checks>=2 else 10)
            if len(reports)==len(rows):
                comparison=(compare_one_axis(reports,plan) if plan['study_stage']=='one_axis'
                    else compare_paired(reports['paired'],plan))
                write_json(campaign/'comparison.json',comparison)
                result.update(status='continuations_verified',comparison_sha256=sha256(campaign/'comparison.json'),
                    results_sha256={name:sha256(campaign/'continuation'/name/'result.json') for name in reports},
                    next_action=comparison['pair_status'])
        result.update(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha,
            plan_sha256=sha256(campaign/'plan.json'),quality='not_evaluated',minimum_inner_duration='not_established')
        write_json(campaign/'monitor_state.json',result)
        print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
