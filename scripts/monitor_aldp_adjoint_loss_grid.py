"""Bounded smoke-only release for the new loss axis; full needs a new registration."""
import argparse
import fcntl
import json
from pathlib import Path

import monitor_aldp_noise_solver_grid as shared
from monitor_aldp_chart_campaign import execute
from run_aldp_chart_campaign import ROOT,sha256,write_json
from run_aldp_adjoint_loss_grid import loss_matrix,verify
from analyze_aldp_loss_pairs import compare_original_surface
from aldp_adjoint_release import combined_active_jobs, diagnostic_release_state


def submit(campaign, plan, python, stage, candidate, rows):
    if stage not in plan['allowed_stages']:
        raise RuntimeError('Stage not released by immutable plan')
    if plan.get('combined_ALDP_gpu_limit'):
        if len(combined_active_jobs()) >= plan['combined_ALDP_gpu_limit']:
            raise RuntimeError('Combined CE/LV GPU slots full')
    if stage == 'full' and not diagnostic_release_state(plan)['ready']:
        raise RuntimeError('Unverified diagnostic retry: full dispatch prohibited')
    key = stage+'__'+candidate
    intent = campaign/'intents'/f'{key}.json'
    if intent.exists() or (campaign/stage/candidate).exists():
        raise RuntimeError('Existing intent/output; duplicate dispatch prohibited')
    command = ['sbatch','--parsable','--partition=gpu','--gres=gpu:1','--cpus-per-task=4',
        '--mem=32G','--exclude=htc-gpu024','--time='+('00:45:00' if stage=='smoke' else '3-00:00:00'),'--job-name=aldp-adjoint-'+candidate,
        '--output='+str(campaign/'logs'/f'{key}_%j.log'),str(ROOT/'exec_aldp_adjoint_loss_grid.sh'),
        str(ROOT),str(python),str(campaign),stage,candidate]
    (campaign/'logs').mkdir(exist_ok=True)
    write_json(intent,dict(status='dispatching',command=command))
    job = execute(command).stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unknown submission outcome; reconcile intent before any retry')
    row = dict(job_id=job,stage=stage,candidate=candidate,
        run_directory=str(campaign/stage/candidate),
        log_path=str(campaign/'logs'/f'{key}_{job}.log'),
        source_sha256=plan['source_manifest_sha256'],plan_sha256=sha256(campaign/'plan.json'),python=str(python))
    write_json(intent,dict(status='submitted',row=row,command=command))
    rows.append(row);write_json(campaign/'ledger.json',rows)
    return row


def pass_once(campaign, python, advance=False):
    plan = json.loads((campaign/'plan.json').read_text())
    if plan['fc_loss_mode'] != 'cross_entropy' or plan['matrix'] != loss_matrix():
        raise RuntimeError('Original-loss compatibility contract changed')
    if plan['allowed_stages'] not in (['smoke'], ['full']):
        raise RuntimeError('Release exactly one stage per immutable plan')
    if plan['allowed_stages'] == ['full'] and not plan.get('cached_smokes'):
        raise RuntimeError('Full release requires all pinned original-loss preflights')
    old_verify, old_compare = shared.verify, shared.compare
    shared.verify, shared.compare = verify, compare_original_surface
    try:
        result = shared.pass_once(campaign, python, advance=False)
    finally:
        shared.verify, shared.compare = old_verify, old_compare
    rows = json.loads((campaign/'ledger.json').read_text()) if (campaign/'ledger.json').exists() else []
    stage = plan['allowed_stages'][0]
    dispatched = {row['candidate'] for row in rows if row['stage']==stage}
    required = {cell['candidate'] for cell in plan['matrix'] if not cell['incompatible_reason']}
    passed = {row['candidate'] for row in result['verified'] if row['stage']=='smoke'}
    if stage == 'full' and passed != required:
        raise RuntimeError('Original-loss preflight set incomplete; full release prohibited')
    release_state = diagnostic_release_state(plan) if stage == 'full' else dict(ready=True,status='smoke_only')
    combined = combined_active_jobs() if plan.get('combined_ALDP_gpu_limit') else []
    new = []
    if advance and not result['failures'] and release_state['ready']:
        for cell in plan['matrix']:
            if cell['incompatible_reason'] or cell['candidate'] in dispatched:
                continue
            if len(result['active'])+len(new) >= plan['max_concurrent_gpu_jobs']:
                break
            if plan.get('combined_ALDP_gpu_limit') and len(combined)+len(new) >= plan['combined_ALDP_gpu_limit']:
                break
            new.append(submit(campaign,plan,python,stage,cell['candidate'],rows))
    all_smokes = required == passed and not any(row['stage']=='smoke' for row in result['failures'])
    streak = 0 if new or not result['active'] else result['consecutive_active']
    result.update(new_submissions=new,consecutive_active=streak,recommended_monitor_minutes=30 if streak>=2 else 10,
        all_original_smokes_verified=all_smokes,
        full_release=('registered_full_campaign' if release_state['ready'] else release_state['status']) if stage=='full' else ('needs_new_registered_campaign' if all_smokes else 'blocked_pending_smokes'),
        diagnostic_release_state=release_state,combined_ALDP_active=combined,
        LV_full_release='blocked_pending_numerical_validation',complete=False if stage=='smoke' else result['complete'])
    write_json(campaign/'monitor_state.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--advance',action='store_true')
    args=parser.parse_args()
    with (args.campaign/'.monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(json.dumps(pass_once(args.campaign,args.python,args.advance),indent=2))
