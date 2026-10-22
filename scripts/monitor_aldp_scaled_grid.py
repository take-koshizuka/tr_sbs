"""Full-only LV continuation; reuse cached preflights and keep four failures."""
import argparse
import fcntl
import json
from pathlib import Path

import monitor_aldp_noise_solver_grid as shared
from aldp_scaled_release import combined_active_jobs, combined_health, validate_plan, verify_static_release
from monitor_aldp_chart_campaign import execute
from run_aldp_chart_campaign import ROOT, sha256, write_json


def submit(campaign, plan, python, candidate, rows):
    if candidate not in validate_plan(plan):
        raise RuntimeError('Historical failure or incompatible candidate cannot be dispatched')
    if len(combined_active_jobs()) >= plan['combined_ALDP_gpu_limit']:
        raise RuntimeError('Combined CE/LV GPU slots full')
    if combined_health(campaign, plan)['failures']:
        raise RuntimeError('Unclassified cross-loss failure blocks all dispatch')
    key = 'full__'+candidate
    intent = campaign/'intents'/(key+'.json')
    if intent.exists() or (campaign/'full'/candidate).exists():
        raise RuntimeError('Existing intent/output: duplicate prohibited')
    log = campaign/'logs'/(key+'_%j.log')
    log.parent.mkdir(exist_ok=True)
    command = ['sbatch','--parsable','--partition=gpu','--gres=gpu:1','--cpus-per-task=4',
        '--mem=32G','--exclude=htc-gpu024','--time=3-00:00:00',
        '--job-name=aldp-lv-scaled-'+candidate,'--output='+str(log),
        str(ROOT/'scripts/slurm/exec_aldp_noise_solver_grid.sh'),str(ROOT),str(python),str(campaign),'full',candidate]
    write_json(intent, dict(status='dispatching',command=command))
    job = execute(command).stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unknown submission outcome: reconcile intent, never duplicate')
    row = dict(job_id=job,stage='full',candidate=candidate,run_directory=str(campaign/'full'/candidate),
        log_path=str(log).replace('%j',job),source_sha256=plan['source_manifest_sha256'],
        plan_sha256=sha256(campaign/'plan.json'),python=str(python))
    rows.append(row)
    write_json(intent,dict(status='submitted',row=row,command=command))
    write_json(campaign/'ledger.json',rows)
    return row


def pass_once(campaign, python, advance=False):
    plan = json.loads((campaign/'plan.json').read_text())
    release = verify_static_release(plan, verify_preflights=False)
    result = shared.pass_once(campaign, python, advance=False)
    health = combined_health(campaign, plan)
    combined = combined_active_jobs()
    ledger = json.loads((campaign/'ledger.json').read_text()) if (campaign/'ledger.json').exists() else []
    dispatched = {r['candidate'] for r in ledger}
    new = []
    if advance and not result['failures'] and not health['failures']:
        for candidate in plan['remaining_candidates']:
            if candidate in dispatched:
                continue
            if len(combined)+len(new) >= plan['combined_ALDP_gpu_limit']:
                break
            if len(new) >= plan['max_new_submissions_per_pass']:
                break
            new.append(submit(campaign, plan, python, candidate, ledger))
    # A new submission is not a terminal cell, even before its first queue row.
    streak = 0 if new or not result['active'] else result['consecutive_active']
    result.update(new_submissions=new,complete=result['complete'] and not new,
        consecutive_active=streak,recommended_monitor_minutes=30 if streak>=2 else 10,
        scaled_release=release,cross_loss_health=health,combined_ALDP_active=combined)
    write_json(campaign/'monitor_state.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--advance',action='store_true')
    args = parser.parse_args()
    with (args.campaign/'.monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(json.dumps(pass_once(args.campaign,args.python,args.advance),allow_nan=False))
