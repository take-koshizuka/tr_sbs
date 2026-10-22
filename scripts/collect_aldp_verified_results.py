#!/usr/bin/env python3
"""Recover audited ALDP results with resumable transfer and streaming SHA checks.

This is host-side bookkeeping, not a worker or scheduler monitor. It never
submits jobs or queries Slurm. Active/unverified campaigns are not downloaded.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


REMOTE_PROBE=r'''
import base64,hashlib,json,pathlib,sys
args=json.loads(sys.argv[1]);campaign=pathlib.Path(args['campaign'])
audit_path=campaign/'monitor_state.json'
audit=json.loads(audit_path.read_text())
if audit.get('status') not in ('normal_curve_verified','continuations_verified','numerical_diagnostic_verified'):
 print(json.dumps(dict(status='not_ready',audit_status=audit.get('status'),
  job_id=audit.get('job_id'),checked_at=audit.get('checked_at'))));sys.exit(0)
code=pathlib.Path(args['code'])
sys.path[:0]=[str(code/'scripts'),str(code/'src')]
from run_aldp_chart_campaign import sha256,verify_source
from monitor_aldp_chart_campaign import read_manifest
plan_path=campaign/'plan.json';plan=json.loads(plan_path.read_text())
source=verify_source()
if source!=plan['source_manifest_sha256'] or source!=audit['source_sha256'] or sha256(plan_path)!=audit['plan_sha256']:
 raise RuntimeError('Campaign/audit/source SHA mismatch')
if args.get('expected_source') is not None and source!=args['expected_source']:
 raise RuntimeError('Requested immutable source differs')
normal=audit['status']=='normal_curve_verified'
numerical=audit['status']=='numerical_diagnostic_verified'
if numerical:
 from monitor_aldp_jax_failure_diagnostic import verify_completed as numerical_completed
 from run_aldp_jax_inner_preflight import check_files
 check_files(plan['frozen_inputs'])
 if sha256(code/'src/tr_asbs/experiment/jax_inner.py')!=plan['jax_backend_sha256']:
  raise RuntimeError('Numerical diagnostic production backend changed')
else:
 from monitor_aldp_jax_inner_curve import verify_prerequisite,verify_completed as normal_completed
 from monitor_aldp_inner_continuation import checked_ledger,verify_completed as continuation_completed
 verify_prerequisite(plan)
rows=read_manifest(campaign/'submitted_jobs.tsv') if (normal or numerical) else checked_ledger(campaign,plan,source)
if (normal or numerical) and (len(rows)!=1 or rows[0]['job_id']!=audit['job_id']):
 raise RuntimeError('One audited reference/diagnostic required')
if not normal and not numerical and {row['candidate'] for row in rows}!=set(plan['cases']):
 raise RuntimeError('All continuation cases must be audited')
metadata={};runs=[]
def add_metadata(name,path):
 data=pathlib.Path(path).read_bytes()
 metadata[name]=dict(sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())
add_metadata('plan.json',plan_path);add_metadata('completion_audit.json',audit_path)
add_metadata('submitted_jobs.tsv',campaign/'submitted_jobs.tsv')
add_metadata('source_manifest.json',code/'source_manifest.json')
for row in rows:
 name=row['candidate'];out=pathlib.Path(row['run_directory'])
 labels=('failed_u0_replay',) if numerical else ('normal','u_short','h_short','paired')
 if name not in labels or out.resolve().is_relative_to(campaign.resolve()) is False:
  raise RuntimeError('Foreign recovery label/output')
 if row['source_sha256']!=source or row['plan_sha256']!=audit['plan_sha256'] or row['repo']!=str(code):
  raise RuntimeError('Ledger source/plan/code mismatch')
 accounts=audit['accounting'] if (normal or numerical) else next(item['accounting'] for item in audit['observations'] if item['job_id']==row['job_id'])
 for suffix in ('','.batch','.extern'):
  state=accounts.get(row['job_id']+suffix,{})
  if state.get('state')!='COMPLETED' or state.get('exit_code')!='0:0':
   raise RuntimeError('Completed0:0 root/batch/extern audit required')
 if numerical:report,count=numerical_completed(row,campaign,plan)
 else:report,count=normal_completed(row,campaign) if normal else continuation_completed(row,campaign,plan)
 expected_result=audit['result_sha256'] if (normal or numerical) else audit['results_sha256'][name]
 if sha256(out/'result.json')!=expected_result:raise RuntimeError('Audited result changed')
 rec=json.loads((out/'receipt.json').read_text());files={}
 for filename,digest in rec['files'].items():
  path=pathlib.Path(filename)
  if path.is_absolute() or '..' in path.parts or not (out/path).resolve().is_relative_to(out.resolve()):
   raise RuntimeError('Unsafe receipt path')
  files[filename]=dict(sha256=digest,bytes=(out/path).stat().st_size)
 files['receipt.json']=dict(sha256=sha256(out/'receipt.json'),bytes=(out/'receipt.json').stat().st_size)
 add_metadata('logs/'+row['job_id']+'.out',row['log_path'])
 runs.append(dict(candidate=name,job_id=row['job_id'],remote_directory=str(out),files=files,
  verified_receipt_files=count,result_sha256=expected_result))
print(json.dumps(dict(status='remote_receipts_verified',source_sha256=source,
 plan_sha256=audit['plan_sha256'],remote_campaign=str(campaign),runs=runs,metadata=metadata)))
'''


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def relative_path(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Recovery path must stay inside its result directory')
    return path


def fingerprint(record):
    identity=dict(source=record['source_sha256'],plan=record['plan_sha256'],
        campaign=record['remote_campaign'],runs=[{key:run[key] for key in (
            'candidate','job_id','remote_directory','result_sha256')} for run in record['runs']])
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def prepare_destination(output,record):
    marker=output/'recovery_manifest.json'
    identity=fingerprint(record)
    if marker.exists():
        if json.loads(marker.read_text())['identity_sha256']!=identity:
            raise RuntimeError('Existing destination belongs to a different result cohort')
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError('Nonempty unregistered destination; preserve existing files')
    output.mkdir(parents=True,exist_ok=True)
    required=sum(info['bytes'] for run in record['runs'] for info in run['files'].values())
    remaining=0
    largest=0
    for run in record['runs']:
        for name,info in run['files'].items():
            path=output/'runs'/run['candidate']/relative_path(name)
            remaining+=max(0,info['bytes']-(path.stat().st_size if path.exists() else 0))
            largest=max(largest,info['bytes'])
    # rsync may keep the previous file while writing its replacement.
    if shutil.disk_usage(output).free<remaining+max(largest,64*1024*1024):
        raise RuntimeError('Insufficient destination space for declared receipt files')
    for name,entry in record['metadata'].items():
        data=base64.b64decode(entry['data'])
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise RuntimeError('Recovered metadata SHA mismatch')
        path=output/'metadata'/relative_path(name);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    manifest={key:value for key,value in record.items() if key!='metadata'}
    manifest.update(status='download_pending',identity_sha256=identity,
        metadata_sha256={name:item['sha256'] for name,item in record['metadata'].items()},
        declared_receipt_bytes=required)
    marker.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_download(output,record):
    count=0
    for run in record['runs']:
        base=output/'runs'/run['candidate']
        for name,entry in run['files'].items():
            path=base/relative_path(name)
            if not path.is_file() or path.stat().st_size!=entry['bytes'] or file_sha(path)!=entry['sha256']:
                raise RuntimeError('Incomplete or changed downloaded receipt file: '+name)
            count+=1
    marker=output/'recovery_manifest.json'
    manifest=json.loads(marker.read_text())
    if manifest['identity_sha256']!=fingerprint(record):raise RuntimeError('Recovery cohort changed')
    manifest.update(status='all_receipt_files_recovered_sha_verified',recovered_files=count,
        quality='not_evaluated',minimum_inner_duration='not_established')
    marker.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remote-campaign',required=True)
    parser.add_argument('--remote-code',required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--host',default='z3.zib.de')
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9.-]+',args.host):raise ValueError('Invalid SSH host')
    account='tkoshizuka@'+args.host
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2']
    payload=dict(campaign=args.remote_campaign,code=args.remote_code,expected_source=args.expected_source_sha256)
    command=shlex.join(['env','JAX_PLATFORMS=cpu','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        args.python,'-c',REMOTE_PROBE,json.dumps(payload)])
    response=subprocess.run(ssh+[account,command],capture_output=True,text=True,check=True,timeout=100)
    record=json.loads(response.stdout)
    if record['status']=='not_ready':print(json.dumps(record));return
    output=args.output.resolve();manifest=prepare_destination(output,record)
    if args.metadata_only:
        print(json.dumps({key:manifest[key] for key in ('status','declared_receipt_bytes','identity_sha256')}));return
    transport=shlex.join(ssh)
    for run in record['runs']:
        destination=output/'runs'/run['candidate'];destination.mkdir(parents=True,exist_ok=True)
        source=account+':'+shlex.quote(run['remote_directory'].rstrip('/')+'/')
        subprocess.run(['rsync','-a','--partial','--timeout=60','-e',transport,'--',source,str(destination)+'/'],check=True)
    manifest=verify_download(output,record)
    print(json.dumps({key:manifest[key] for key in ('status','recovered_files','declared_receipt_bytes')}))


if __name__=='__main__':main()
