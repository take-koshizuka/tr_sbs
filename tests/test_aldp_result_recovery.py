"""Receipt recovery identity, partial transfer and integrity; no simulations."""
import base64
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_aldp_verified_results import (
    REMOTE_PROBE,prepare_destination,relative_path,verify_download,
)


def fixture():
    files={'result.json':b'{"minimum":"not_established"}',
           'inner_snapshots/prefix.pt':b'tiny byte fixture, not a molecular checkpoint',
           'receipt.json':b'{"postflight_verified":true}'}
    metadata=b'{"status":"normal_curve_verified"}'
    record=dict(status='remote_receipts_verified',source_sha256='source',plan_sha256='plan',
        remote_campaign='/frozen/campaign',runs=[dict(candidate='normal',job_id='5199220',
            remote_directory='/frozen/campaign/normal',result_sha256=hashlib.sha256(files['result.json']).hexdigest(),
            files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()})],
        metadata={'completion_audit.json':{'data':base64.b64encode(metadata).decode(),
            'sha256':hashlib.sha256(metadata).hexdigest()}})
    return record,files


def test_partial_download_remains_unverified_and_same_cohort_can_resume(tmp_path):
    compile(REMOTE_PROBE,'remote_recovery_probe','exec')
    record,files=fixture();output=tmp_path/'recovery'
    prepare_destination(output,record)
    base=output/'runs/normal'
    first=base/'result.json';first.parent.mkdir(parents=True);first.write_bytes(files['result.json'])
    with pytest.raises(RuntimeError,match='Incomplete or changed'):
        verify_download(output,record)
    assert json.loads((output/'recovery_manifest.json').read_text())['status']=='download_pending'
    prepare_destination(output,record)
    for name,data in files.items():
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    proof=verify_download(output,record)
    assert proof['status']=='all_receipt_files_recovered_sha_verified' and proof['recovered_files']==3
    assert proof['quality']=='not_evaluated'


def test_corrupted_checkpoint_rejects_verification_and_foreign_cohort(tmp_path):
    record,files=fixture();output=tmp_path/'recovery';prepare_destination(output,record)
    for name,data in files.items():
        path=output/'runs/normal'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    path=output/'runs/normal/inner_snapshots/prefix.pt';path.write_bytes(b'x'*len(files['inner_snapshots/prefix.pt']))
    with pytest.raises(RuntimeError,match='Incomplete or changed'):
        verify_download(output,record)
    changed=deepcopy(record);changed['runs'][0]['job_id']='foreign-job'
    with pytest.raises(RuntimeError,match='different result cohort'):
        prepare_destination(output,changed)


def test_recovery_preserves_unregistered_files_and_rejects_traversal(tmp_path):
    record,files=fixture();output=tmp_path/'existing';output.mkdir()
    existing=output/'user_file.txt';existing.write_bytes(b'preserve')
    with pytest.raises(RuntimeError,match='Nonempty unregistered'):
        prepare_destination(output,record)
    assert existing.read_bytes()==b'preserve'
    for name in ('../outside.pt','/absolute.pt',''):
        with pytest.raises(ValueError,match='inside its result directory'):relative_path(name)


def test_native_rsync_partial_transfer_then_full_receipt_verification(tmp_path):
    rsync=shutil.which('rsync')
    if rsync is None:pytest.skip('Native rsync unavailable')
    record,files=fixture();source=tmp_path/'source';output=tmp_path/'recovery'
    for name,data in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    prepare_destination(output,record)
    destination=output/'runs/normal';destination.mkdir(parents=True)
    partial=destination/'inner_snapshots/prefix.pt';partial.parent.mkdir()
    partial.write_bytes(files['inner_snapshots/prefix.pt'][:5])
    subprocess.run([rsync,'-a','--partial','--timeout=60','--',str(source)+'/',str(destination)+'/'],
        check=True,capture_output=True)
    proof=verify_download(output,record)
    assert proof['recovered_files']==3 and partial.read_bytes()==files['inner_snapshots/prefix.pt']


@pytest.mark.parametrize('exit_code', ['0:0','1:0'])
def test_diagnostic_probe_uses_actual_receipt_verifier_and_zero_exit_audit(
        tmp_path,monkeypatch,capsys,exit_code):
    """Synthetic accounting/bytes only; these are not molecular results."""
    pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
    import run_aldp_chart_campaign as worker
    from monitor_aldp_jax_inner_campaign import FIELDS
    code=tmp_path/'code';campaign=tmp_path/'results'
    output=campaign/'numerical_diagnostic/seed0';output.mkdir(parents=True)
    backend=code/'src/tr_asbs/experiment/jax_inner.py';backend.parent.mkdir(parents=True)
    backend.write_bytes(b'protocol fixture only, not an executable backend')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=code/'source_manifest.json'
    manifest.write_text(json.dumps(dict(files={'src/tr_asbs/experiment/jax_inner.py':digest(backend)})))
    source=digest(manifest)
    plan=dict(source_manifest_sha256=source,frozen_inputs={},jax_backend_sha256=digest(backend),
        replay_epoch_limit=100,openmm_cap=12800,failed_job_id='synthetic_failed')
    (campaign/'plan.json').write_text(json.dumps(plan));plan_sha=digest(campaign/'plan.json')
    (output/'config.yaml').write_text('protocol_fixture: true\n')
    report=dict(status='numerical_rejection_captured',source_sha256=source,plan_sha256=plan_sha,
        config_sha256=digest(output/'config.yaml'),scientific_status='not_evaluated',
        normal_reference_completed=False,global_step=3,neural_updates=3,openmm_configurations=4,
        failed_job_id='synthetic_failed',diagnosis={'synthetic_protocol_fixture':True})
    (output/'result.json').write_text(json.dumps(report))
    (output/'rejected_batch.pt').write_bytes(b'protocol bytes; not a model or checkpoint')
    worker.receipt(output,dict(stage='jax_numerical_diagnostic',chart='low_density',seed=0),
                   dict(source_sha256=source,plan_sha256=plan_sha))
    log=campaign/'synthetic.out';log.write_text('protocol fixture\npostflight_verified=true\n')
    row=dict(job_id='synthetic',stage='jax_numerical_diagnostic',candidate='failed_u0_replay',seed=0,
        run_directory=str(output),log_path=str(log),repo=str(code),python='unused',
        source_sha256=source,plan_sha256=plan_sha)
    with (campaign/'submitted_jobs.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,delimiter='\t');writer.writeheader();writer.writerow(row)
    audit=dict(status='numerical_diagnostic_verified',job_id='synthetic',source_sha256=source,
        plan_sha256=plan_sha,result_sha256=digest(output/'result.json'),
        accounting={'synthetic'+suffix:dict(state='COMPLETED',exit_code=exit_code)
                    for suffix in ('','.batch','.extern')})
    (campaign/'monitor_state.json').write_text(json.dumps(audit))
    monkeypatch.setattr(worker,'ROOT',code)
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(sys,'argv',['recovery_probe',json.dumps(
        dict(campaign=str(campaign),code=str(code),expected_source=source))])
    if exit_code!='0:0':
        with pytest.raises(RuntimeError,match='Completed0:0'):
            exec(compile(REMOTE_PROBE,'remote_probe','exec'),{})
        return
    exec(compile(REMOTE_PROBE,'remote_probe','exec'),{})
    record=json.loads(capsys.readouterr().out)
    assert record['runs'][0]['candidate']=='failed_u0_replay'
    assert record['runs'][0]['verified_receipt_files']==3
    assert 'rejected_batch.pt' in record['runs'][0]['files']
    recovery=tmp_path/'recovery';prepare_destination(recovery,record)
    shutil.copytree(output,recovery/'runs/failed_u0_replay')
    proof=verify_download(recovery,record)
    assert proof['recovered_files']==4 and proof['quality']=='not_evaluated'
