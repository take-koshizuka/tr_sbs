
import pytest

pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
import json
from types import SimpleNamespace


import aldp_scaled_release as release
from run_aldp_chart_campaign import sha256
from run_aldp_noise_solver_grid import configuration, matrix, expected_cost


def plan():
    failed = [dict(job_id=j,candidate=c,retry_allowed=False,scientific_status='not_evaluated')
              for j,c in [('5263587','brownian_constant__euler'),('5263588','brownian_constant__legacy'),
                          ('5278815','brownian_constant__diffusion_euler'),('5278816','brownian_linear__euler')]]
    compatible = {r['candidate'] for r in matrix(True) if not r['incompatible_reason']}
    remaining = sorted(compatible-{r['candidate'] for r in failed})
    return dict(allowed_stages=['full'],controller_backward_mode='loss_scaled_fp64_clip',
        fc_loss_mode='log_variance',matrix=matrix(True),outer_loops=5,combined_ALDP_gpu_limit=2,
        max_new_submissions_per_pass=1,remaining_candidates=remaining,
        classified_terminal_failures=dict(rows=failed),base_config_relative='scripts/aldp_noise_solver_baseline.yaml',
        integrated_diffusion_variance=.249999)


def test_registered_remaining_grid_keeps_all_failed_cells_and_rejects_retry():
    p=plan()
    assert len(release.validate_plan(p)) == 14
    with pytest.raises(RuntimeError,match='Only unsubmitted'):
        release.require_scaled_release(p,'full','brownian_constant__euler')
    p['classified_terminal_failures']['rows'][0]['retry_allowed'] = True
    with pytest.raises(RuntimeError,match='retry/admission'):
        release.validate_plan(p)


def test_scaled_configuration_changes_only_numerical_mode_and_preserves_cost(tmp_path):
    p=plan();candidate='ou__linear_exponential'
    scaled,_=configuration(p,tmp_path,'full',candidate)
    original=dict(p);original.pop('controller_backward_mode')
    native,_=configuration(original,tmp_path,'full',candidate)
    from omegaconf import OmegaConf
    a=OmegaConf.to_container(scaled,resolve=True);b=OmegaConf.to_container(native,resolve=True)
    assert a.pop('fc_backward_mode') == 'loss_scaled_fp64_clip'
    assert a==b
    assert expected_cost(scaled,128)==expected_cost(native,128)
    assert expected_cost(scaled,128)['total_openmm_configurations']==1294528
    with pytest.raises(ValueError,match='full-only'):
        configuration(p,tmp_path,'smoke',candidate)


def health_fixture(tmp_path,state):
    (tmp_path/'plan.json').write_text('{}')
    row=dict(job_id='123',stage='full',candidate='cell',source_sha256='source',
             plan_sha256=sha256(tmp_path/'plan.json'),run_directory=str(tmp_path/'full/cell'))
    (tmp_path/'ledger.json').write_text(json.dumps([row]))
    return row, f'123|{state}|0:0\n123.batch|{state}|0:0\n123.extern|COMPLETED|0:0\n'


def test_a_failed_peer_blocks_even_when_extern_is_still_active(tmp_path,monkeypatch):
    row,_=health_fixture(tmp_path,'FAILED')
    monkeypatch.setattr(release,'execute',lambda args:SimpleNamespace(stdout='123|FAILED|1:0\n123.batch|FAILED|1:0\n123.extern|RUNNING|0:0\n'))
    health=release.campaign_health(tmp_path,'source',row['plan_sha256'])
    assert len(health['failures'])==1 and not health['active']


def test_a_failure_artifact_blocks_while_scheduler_still_says_running(tmp_path,monkeypatch):
    row,raw=health_fixture(tmp_path,'RUNNING')
    output=tmp_path/'full/cell';output.mkdir(parents=True)
    (output/'failure.json').write_text('{}')
    monkeypatch.setattr(release,'execute',lambda args:SimpleNamespace(stdout=raw))
    assert release.campaign_health(tmp_path,'source',row['plan_sha256'])['failures']


def test_new_job_with_unavailable_accounting_is_reserved_not_completed(tmp_path,monkeypatch):
    row,_=health_fixture(tmp_path,'RUNNING')
    monkeypatch.setattr(release,'execute',lambda args:SimpleNamespace(stdout=''))
    health=release.campaign_health(tmp_path,'source',row['plan_sha256'])
    assert len(health['active'])==1 and not health['failures']


def test_combined_capacity_blocks_before_intent(tmp_path,monkeypatch):
    import monitor_aldp_scaled_grid as m
    monkeypatch.setattr(m,'combined_active_jobs',lambda:[['ce1'],['ce2']])
    with pytest.raises(RuntimeError,match='slots full'):
        m.submit(tmp_path,plan(),'python','ou__euler',[])
    assert not (tmp_path/'intents').exists()


def test_peer_failure_blocks_before_intent(tmp_path,monkeypatch):
    import monitor_aldp_scaled_grid as m
    monkeypatch.setattr(m,'combined_active_jobs',lambda:[])
    monkeypatch.setattr(m,'combined_health',lambda *a:dict(failures=[dict(job_id='ce1')]))
    with pytest.raises(RuntimeError,match='cross-loss failure'):
        m.submit(tmp_path,plan(),'python','ou__euler',[])
    assert not (tmp_path/'intents').exists()


def test_registered_diagnostic_proof_cannot_be_changed(tmp_path):
    p=plan();output=tmp_path/'parent/diagnostic';output.mkdir(parents=True)
    (output/'gate.json').write_text('{}');(output/'receipt.json').write_text('{}')
    p['diagnostic_release_guard']=dict(campaign=str(output.parent),gate_sha256='changed',receipt_sha256='changed')
    with pytest.raises(RuntimeError,match='proof changed'):
        release.verify_static_release(p)
