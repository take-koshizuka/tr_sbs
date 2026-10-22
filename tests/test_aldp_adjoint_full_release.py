import hashlib
import json
from types import SimpleNamespace

import pytest

import aldp_adjoint_release as release
from run_aldp_adjoint_loss_grid import loss_matrix


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


def fixture(tmp_path):
    campaign = tmp_path/'diagnostic_parent/results'
    source = tmp_path/'diagnostic_parent/code/source_manifest.json'
    write(source, dict(files={}))
    tolerances = dict(loss_relative=1e-4, gradient_relative_l2=1e-3,
                      parameter_max_absolute=1e-5, update_relative_l2=.02)
    dp = dict(source_manifest_sha256=sha(source), frozen_inputs={}, tolerances=tolerances,
              initial_Adam_counters=dict(normal=[199618], witness=[100701]))
    write(campaign/'plan.json', dp)
    write(campaign/'submission.json', dict(job_id='123'))
    rows = []
    for case in ['normal', 'witness']:
        for number in range(1, 21):
            expected = [dp['initial_Adam_counters'][case][0]+number]
            rows.append(dict(case=case, update=number, gates=dict(original=True),
                             Adam_counters=expected, reference_Adam_counters=expected,
                             expected_Adam_counters=expected))
    gate = dict(status='passed', tolerances=tolerances, source_sha256=sha(source),
                plan_sha256=sha(campaign/'plan.json'), OpenMM=0, optimizer_updates=40,
                reference_optimizer_updates=40, matmul_precision='highest',
                optimizer_updates_discarded=True, production_changed=False,
                full_training_submitted=False, rows=rows)
    write(campaign/'diagnostic/gate.json', gate)
    write(campaign/'diagnostic/receipt.json', dict(files={'gate.json': sha(campaign/'diagnostic/gate.json')},
          inputs=dict(source_sha256=sha(source), plan_sha256=sha(campaign/'plan.json'))))
    return dict(allowed_stages=['full'], diagnostic_release_guard=dict(campaign=str(campaign),
                job_id='123', plan_sha256=sha(campaign/'plan.json'), source_sha256=sha(source), tolerances=tolerances))


def test_full_guard_requires_complete_successful_retry_and_all_rows(tmp_path, monkeypatch):
    p = fixture(tmp_path)
    monkeypatch.setattr(release, 'execute', lambda args: SimpleNamespace(stdout='123|COMPLETED|0:0\n123.batch|COMPLETED|0:0\n123.extern|COMPLETED|0:0\n'))
    assert release.diagnostic_release_state(p)['ready']
    campaign = tmp_path/'diagnostic_parent/results'
    g = json.loads((campaign/'diagnostic/gate.json').read_text())
    g['rows'][0]['gates']['original'] = False
    write(campaign/'diagnostic/gate.json', g)
    receipt = json.loads((campaign/'diagnostic/receipt.json').read_text())
    receipt['files']['gate.json'] = sha(campaign/'diagnostic/gate.json')
    write(campaign/'diagnostic/receipt.json', receipt)
    with pytest.raises(AssertionError):
        release.diagnostic_release_state(p)


@pytest.mark.parametrize('state', ['RUNNING', 'FAILED'])
def test_active_or_failed_retry_never_releases_more_work(tmp_path, monkeypatch, state):
    p = fixture(tmp_path)
    monkeypatch.setattr(release, 'execute', lambda args: SimpleNamespace(stdout=f'123|{state}|1:0\n'))
    assert not release.diagnostic_release_state(p)['ready']
    with pytest.raises(RuntimeError, match='blocked'):
        release.require_full_release(p)


def test_full_monitor_keeps_smoke_proof_while_full_is_active(tmp_path, monkeypatch):
    import monitor_aldp_adjoint_loss_grid as m
    smoke = [dict(stage='smoke', candidate=c['candidate']) for c in loss_matrix() if not c['incompatible_reason']]
    p = dict(allowed_stages=['full'], fc_loss_mode='cross_entropy', matrix=loss_matrix(),
             cached_smokes={'pinned': True}, max_concurrent_gpu_jobs=2, combined_ALDP_gpu_limit=2)
    write(tmp_path/'plan.json', p)
    monkeypatch.setattr(m.shared, 'pass_once', lambda *a, **kw: dict(verified=smoke, failures=[],
                        active=[dict(stage='full', candidate='brownian_constant__euler')], complete=False, consecutive_active=1))
    monkeypatch.setattr(m, 'diagnostic_release_state', lambda p: dict(ready=True, status='verified_saved_batch_gate'))
    monkeypatch.setattr(m, 'combined_active_jobs', lambda: [['1', 'aldp-ce', 'RUNNING']])
    result = m.pass_once(tmp_path, 'python')
    assert result['all_original_smokes_verified']
    assert result['full_release'] == 'registered_full_campaign'


def test_combined_lv_slot_allows_only_one_new_ce_job(tmp_path, monkeypatch):
    import monitor_aldp_adjoint_loss_grid as m
    smoke = [dict(stage='smoke', candidate=c['candidate']) for c in loss_matrix() if not c['incompatible_reason']]
    p = dict(allowed_stages=['full'], fc_loss_mode='cross_entropy', matrix=loss_matrix(),
             cached_smokes={'pinned': True}, max_concurrent_gpu_jobs=2, combined_ALDP_gpu_limit=2)
    write(tmp_path/'plan.json', p)
    monkeypatch.setattr(m.shared, 'pass_once', lambda *a, **kw: dict(verified=smoke, failures=[], active=[], complete=False, consecutive_active=0))
    monkeypatch.setattr(m, 'diagnostic_release_state', lambda p: dict(ready=True, status='verified_saved_batch_gate'))
    monkeypatch.setattr(m, 'combined_active_jobs', lambda: [['1', 'aldp-lv', 'RUNNING']])
    monkeypatch.setattr(m, 'submit', lambda campaign, plan, python, stage, candidate, rows: dict(stage=stage, candidate=candidate))
    result = m.pass_once(tmp_path, 'python', advance=True)
    assert len(result['new_submissions']) == 1


def test_full_dispatch_guard_runs_before_intent_or_sbatch(tmp_path, monkeypatch):
    import monitor_aldp_adjoint_loss_grid as m
    monkeypatch.setattr(m, 'combined_active_jobs', lambda: [['1'], ['2']])
    with pytest.raises(RuntimeError, match='slots full'):
        m.submit(tmp_path, dict(allowed_stages=['full'], combined_ALDP_gpu_limit=2), 'python', 'full', 'case', [])
    assert not (tmp_path/'intents').exists()


def test_direct_worker_cannot_bypass_smoke_only_plan():
    with pytest.raises(RuntimeError, match='stage is not released'):
        release.require_full_release(dict(allowed_stages=['smoke']), 'full')
