import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SPEC=importlib.util.spec_from_file_location('light',Path(__file__).parents[1]/'scripts/monitor_aldp_loss_light.py')
light=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(light)


def setup(tmp_path):
    campaign=tmp_path/'lv/results';campaign.mkdir(parents=True)
    (campaign.parent/'code/scripts').mkdir(parents=True)
    (campaign.parent/'code/source_manifest.json').write_text('{}')
    (campaign/'plan.json').write_text('{}');(campaign/'intents').mkdir()
    run=campaign/'full/case';run.mkdir(parents=True)
    log=campaign/'job.log';log.write_text('training running')
    row=dict(job_id='123',source_sha256=light.digest(campaign.parent/'code/source_manifest.json'),
        plan_sha256=light.digest(campaign/'plan.json'),run_directory=str(run),log_path=str(log),stage='full',candidate='case')
    (campaign/'ledger.json').write_text(json.dumps([row]))
    (campaign/'intents/case.json').write_text(json.dumps(dict(status='submitted',row=row)))
    reg=dict(pools=dict(LV=dict(campaign=str(campaign),plan_sha256=row['plan_sha256'],source_sha256=row['source_sha256'])),
        previously_verified_completed={},initial_active_streak=0)
    return reg,row,run


def account(state='RUNNING'):
    return '\n'.join(f'{job}|{state}|0:0' for job in ['123','123.batch','123.extern'])


def test_unchanged_running_poll_does_not_read_large_artifacts_or_launch_full(tmp_path,monkeypatch):
    reg,row,run=setup(tmp_path)
    campaign=Path(reg['pools']['LV']['campaign'])
    second=dict(row,job_id='124',candidate='second')
    (campaign/'ledger.json').write_text(json.dumps([row,second]))
    (campaign/'intents/second.json').write_text(json.dumps(dict(status='submitted',row=second)))
    (run/'large.pt').write_bytes(b'untouched')
    original=light.digest
    def hash_small(path):
        assert Path(path).name!='large.pt'
        return original(path)
    monkeypatch.setattr(light,'digest',hash_small)
    monkeypatch.setattr(light,'command',lambda args:account()+'\n'+account().replace('123','124')
        if args[0]=='sacct' else '123|aldp-case|RUNNING\n124|aldp-second|RUNNING')
    monkeypatch.setattr(light.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Full executed')))
    out=tmp_path/'monitor/passes/one';out.mkdir(parents=True)
    result=light.pass_once(reg,out,'python',advance=True)
    assert result['mode']=='light_status' and not result['needs_full_validation']
    assert result['new_submissions']==[] and result['recommended_monitor_minutes']==10
    out2=tmp_path/'monitor/passes/two';out2.mkdir()
    assert light.pass_once(reg,out2,'python',advance=True)['recommended_monitor_minutes']==30


def test_completion_requires_full_validation_not_queue_disappearance(tmp_path):
    reg,row,run=setup(tmp_path)
    assert light.inspect(reg,account('COMPLETED'),'')['needs_full_validation']
    assert light.inspect(reg,'','')['failures'][0]['reason']=='missing_accounting'


def test_failure_blocks_and_preserves_failure_evidence(tmp_path):
    reg,row,run=setup(tmp_path);(run/'failure.json').write_text('{}')
    result=light.inspect(reg,account(),'123|aldp-case|RUNNING')
    assert result['failures'][0]['reason']=='failure_artifact' and result['needs_full_validation']


def test_changed_verified_receipt_is_not_silently_accepted(tmp_path):
    reg,row,run=setup(tmp_path);(run/'receipt.json').write_text('{}')
    reg['previously_verified_completed']['123']={'receipt.json':light.digest(run/'receipt.json')}
    assert not light.inspect(reg,account('COMPLETED'),'123|aldp-case|RUNNING')['failures']
    (run/'receipt.json').write_text('{"changed":1}')
    assert light.inspect(reg,account('COMPLETED'),'')['failures'][0]['reason']=='verified_terminal_metadata_changed'


def test_untracked_queue_or_uncertain_intent_blocks(tmp_path):
    reg,row,run=setup(tmp_path)
    assert light.inspect(reg,account(),'999|aldp-other|RUNNING')['failures']
    p=Path(reg['pools']['LV']['campaign'])/'intents/case.json';p.write_text('{"status":"dispatching"}')
    assert light.inspect(reg,account(),'123|aldp-case|RUNNING')['failures']


def test_changed_plan_rejected(tmp_path):
    reg,row,run=setup(tmp_path)
    (Path(reg['pools']['LV']['campaign'])/'plan.json').write_text('{"changed":1}')
    import pytest
    with pytest.raises(RuntimeError,match='plan changed'):
        light.inspect(reg,account(),'123|aldp-case|RUNNING')


def test_full_boundary_keeps_fail_closed_and_original_arguments(tmp_path,monkeypatch):
    reg,row,run=setup(tmp_path);(run/'failure.json').write_text('{}')
    parent=Path(reg['pools']['LV']['campaign']).parent/'code/scripts/monitor_aldp_loss_campaigns.py'
    parent.write_text('parent');reg['parent_monitor_sha256']=light.digest(parent)
    monkeypatch.setattr(light,'command',lambda args:account() if args[0]=='sacct' else '123|aldp-case|RUNNING')
    calls=[]
    full=dict(reports={'LV':dict(failures=[],verified=[])},active=[],new_submissions=[],dispatch_blocked=False,complete=True)
    def execute(args,**kw):
        calls.append((args,kw));return SimpleNamespace(returncode=0,stdout=json.dumps(full),stderr='')
    monkeypatch.setattr(light.subprocess,'run',execute)
    out=tmp_path/'monitor/passes/one';out.mkdir(parents=True)
    result=light.pass_once(reg,out,'python',advance=True)
    assert '--advance' not in calls[0][0] and calls[0][1]['timeout']==480
    assert result['dispatch_blocked'] and not result['complete'] and result['failures']
