#!/usr/bin/env python3
"""One bounded scheduler snapshot; registered transition probe only."""
import argparse,json,csv,fcntl
from pathlib import Path
from monitor_aldp_jax_inner_campaign import accounting,FIELDS
from monitor_aldp_chart_campaign import execute,read_manifest,ACTIVE
from run_aldp_chart_campaign import ROOT,sha256,verify_source,write_json
from run_aldp_outer_extension import validate_registration

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--submit',action='store_true');a=p.parse_args();c=a.campaign
 with (c/'monitor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);plan=json.loads((c/'plan.json').read_text());validate_registration(plan['registration']);source=verify_source()
  if source!=plan['source_manifest_sha256']:raise RuntimeError('Source mismatch')
  for f,h in plan['frozen_inputs'].items():
   if sha256(f)!=h:raise RuntimeError('Input SHA mismatch '+f)
  ledger=c/'submitted_jobs.tsv';rows=read_manifest(ledger)
  if a.submit:
   if rows:raise RuntimeError('Probe already submitted')
   intent=c/'submission_intent.json';out=c/'outer_extension/normal'
   if intent.exists() or out.exists():raise RuntimeError('Existing dispatch/output; reconcile')
   log=c/'logs/aldp-outer-full-%j.out';log.parent.mkdir(parents=True,exist_ok=True)
   cmd=['sbatch','--parsable','--partition=gpu','--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=06:00:00','--exclude=htc-gpu024',f'--output={log}',str(ROOT/'scripts/slurm/exec_aldp_outer_extension_full.sh'),str(ROOT),a.python,str(c)]
   write_json(intent,dict(status='dispatching',command=cmd));job=execute(cmd).stdout.strip().split(';')[0]
   if not job.isdigit():raise RuntimeError('Unparseable job; reconcile')
   row=dict(job_id=job,stage='outer_extension',candidate='normal',seed=0,run_directory=str(out),log_path=str(log).replace('%j',job),repo=str(ROOT),python=a.python,source_sha256=source,plan_sha256=sha256(c/'plan.json'))
   with ledger.open('x') as f:w=csv.DictWriter(f,fieldnames=FIELDS,delimiter='\t');w.writeheader();w.writerow(row)
   write_json(intent,dict(status='submitted',row=row));result=dict(status='submitted',row=row,active_checks=0,next_check_interval_minutes=10,quality='not_evaluated',full_extension_submitted=True)
  else:
   if len(rows)!=1:raise RuntimeError('Exactly one probe required')
   row=rows[0];state,accounts=accounting(row['job_id']);prev=json.loads((c/'monitor_state.json').read_text()) if (c/'monitor_state.json').exists() else {};active=prev.get('active_checks',0)+1 if state in ACTIVE else 0
   result=dict(job_id=row['job_id'],scheduler_state=state,accounting=accounts,source_sha256=source,plan_sha256=sha256(c/'plan.json'),active_checks=active,next_check_interval_minutes=30 if active>=2 else 10,full_extension_submitted=True)
   log=Path(row['log_path']);result['log_sha256']=sha256(log) if log.exists() else None
   if state=='COMPLETED':
    assert all(accounts.get(k,{}).get('state')=='COMPLETED' and accounts[k]['exit_code']=='0:0' for k in [row['job_id'],row['job_id']+'.batch',row['job_id']+'.extern'])
    out=Path(row['run_directory']);rec=json.loads((out/'receipt.json').read_text());r=json.loads((out/'result.json').read_text())
    assert rec['postflight_verified'] and rec['identity']==dict(stage='outer_extension',candidate='normal',seed=0)
    assert r['status']=='artifact_complete' and r['source_sha256']==source and r['provenance']['plan_sha256']==row['plan_sha256']==result['plan_sha256']
    assert r['config_sha256']==sha256(out/'config.yaml') and r['provenance']['parent_snapshot_sha256']==plan['registration']['parent_snapshot_sha256']
    diag=r['diagnostics']
    assert r['global_step']==900000 and r['new_optimizer_steps']==300000 and r['cached_global_step']==600000
    assert not r['rejected_training_steps'] and diag['backend_adam_update_counts']=={'u':600000,'h':300000}
    assert diag['resume_marker']['restored'] and diag['resume_marker']['initial_online_suppressed']
    assert r['cost']['total_openmm_configurations']==260224
    expected=[(3,'u',0,600000),(3,'u',2000,800000),(3,'h',0,800000),(3,'h',1000,900000)]
    assert [(x['outer_loop'],x['phase'],x['phase_epoch'],x['global_step']) for x in diag['inner_boundaries']]==expected
    for f,h in rec['files'].items():assert sha256(out/f)==h
    assert 'postflight_verified=true' in log.read_text()
    result.update(status='outer_extension_verified',receipt_files_verified=len(rec['files']),result_sha256=sha256(out/'result.json'),report=r,comparison=json.loads((out/'comparison.json').read_text()))
   elif state in ACTIVE:result['status']='active'
   else:result.update(status='attention_required',log_tail=log.read_text(errors='replace')[-6000:] if log.exists() else None)
  write_json(c/'monitor_state.json',result);print(json.dumps(result))
if __name__=='__main__':main()
