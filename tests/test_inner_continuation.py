"""Shared-loop duration changes and coupled BC provenance on tiny Gaussians."""

import pytest

pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
from copy import deepcopy
from pathlib import Path
import json
import random
import sys
from types import SimpleNamespace

import jax
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_aldp_inner_curve import screen
from run_aldp_inner_continuation import (
    ContinuationObserver, compare_one_axis, make_one_axis_cases, policy_checks, verify_plan,
)
import monitor_aldp_inner_continuation as monitor
import run_aldp_inner_continuation as runner
from run_aldp_chart_campaign import sha256
from test_inner_curve_screening import fixture
from test_jax_inner_curve import gaussian_shared_fit
from tr_asbs.experiment.inner_observation import (
    attach_exact_inner_resume, continuation_training_contract, isolated_inner_diagnostic,
    observe_inner_boundaries, save_inner_snapshot,
)


def seeded_fit(u,h):
    module=gaussian_shared_fit(u_epochs=u,h_epochs=h)
    torch.manual_seed(9876);np.random.seed(23);random.seed(45)
    return module


def stop_at(module,path,phase,epoch,provenance):
    def callback(row):
        if row['outer_loop']==1 and row['phase']==phase and row['phase_epoch']==epoch:
            save_inner_snapshot(module,path,provenance)
            raise RuntimeError('registered test prefix reached')
    observe_inner_boundaries(module,callback,{'u':[epoch],'h':[epoch]})
    with pytest.raises(RuntimeError,match='registered test prefix'):
        module.fit()


@pytest.mark.parametrize('case',['u_short','h_short','paired'])
def test_registered_shortened_resume_matches_fresh_same_duration_through_next_outer(tmp_path,case):
    u,h=(1,2) if case=='u_short' else (3,1) if case=='h_short' else (1,1)
    full=seeded_fit(u,h);full.fit();final_rng=torch.get_rng_state().clone()
    provenance={'test':'normal first-outer exact prefix'}
    path=tmp_path/'prefix.pt'
    parent=seeded_fit(3,2)
    phase,epoch=('h',1) if case=='h_short' else ('u',1)
    stop_at(parent,path,phase,epoch,provenance)
    if case=='paired':
        # h must come from the shortened-u branch. A normal-u h checkpoint
        # would retain the wrong endpoint buffer and current/Adam counters.
        coupled=gaussian_shared_fit(u_epochs=1,h_epochs=2)
        attach_exact_inner_resume(coupled,path,expected_provenance=provenance,
            expected_training_contract=continuation_training_contract(parent.cfg))
        path=tmp_path/'coupled_h_prefix.pt'
        provenance={'test':'shortened-u h prefix'}
        stop_at(coupled,path,'h',1,provenance)
    resumed=gaussian_shared_fit(u_epochs=u,h_epochs=h)
    marker=attach_exact_inner_resume(resumed,path,expected_provenance=provenance,
        expected_training_contract=continuation_training_contract(full.cfg))
    resumed.fit()
    assert marker['restored'] and marker['initial_online_suppressed']
    assert resumed.current_epoch==2*(u+h) and resumed.global_step==4*(u+h)
    assert torch.equal(torch.get_rng_state(),final_rng)
    for phase in ('u','h'):
        for left,right in zip(jax.tree.leaves(full.test_backend.params[phase]),
                              jax.tree.leaves(resumed.test_backend.params[phase])):
            np.testing.assert_array_equal(left,right)
        for left,right in zip(jax.tree.leaves(full.test_backend.states[phase]),
                              jax.tree.leaves(resumed.test_backend.states[phase])):
            np.testing.assert_array_equal(left,right)
    for name in ('adjoint_buffer','corrector_buffer'):
        for key in getattr(full,name).storage:
            torch.testing.assert_close(torch.stack(getattr(full,name).storage[key]),
                torch.stack(getattr(resumed,name).storage[key]),rtol=0,atol=0)


def test_case_registration_never_combines_independent_h_snapshot(tmp_path):
    report,contract=fixture();report['provenance']={'source':'normal immutable curve'}
    cases=make_one_axis_cases(report,screen(report,contract),tmp_path)
    assert set(cases)=={'normal','u_short','h_short'}
    assert cases['u_short']['checkpoint_path']==str(tmp_path/'u250.pt')
    assert cases['u_short']['save_first_h_prefix']==125
    assert cases['h_short']['cached_global_step']==212500
    plan=dict(schema='aldp_inner_continuation_plan_v1',study_stage='one_axis',
              cases=cases,curve_contract=contract)
    verify_plan(plan)
    bad=deepcopy(plan);bad['study_stage']='paired'
    bad['cases']={'paired':dict(cases['h_short'],u_epochs=250)}
    with pytest.raises(ValueError,match='different u/h history'):
        verify_plan(bad)


def test_common_h_validation_detects_degradation_when_forward_metrics_are_equal():
    report,contract=fixture();policy=report['diagnostics']['inner_boundaries'][-1]['policy']
    baseline={'policy':policy,'matched_h_validation':{'heldout_loss':1.}}
    bad={'policy':deepcopy(policy),'matched_h_validation':{'heldout_loss':1.2}}
    bad['policy']['metrics']['b']['physical_energy_w2']=1e25
    checks=policy_checks(bad,baseline,contract['non_deterioration_limits'])
    assert not checks['heldout_loss']['passed']
    assert all(row['passed'] for name,row in checks.items() if name!='heldout_loss')


def test_common_bc_bank_is_separate_from_changed_u_endpoints_and_preserves_rng(tmp_path):
    normal=gaussian_shared_fit();short=gaussian_shared_fit()
    with torch.no_grad():
        short.controller.out_layer.bias.add_(.15)
    def observer(output):
        output.mkdir()
        obj=ContinuationObserver.__new__(ContinuationObserver)
        obj.output,obj.contract,obj.validation=output,{'validation_samples':5},{}
        return obj
    original,changed=observer(tmp_path/'normal'),observer(tmp_path/'short')
    rng=torch.get_rng_state().clone()
    with isolated_inner_diagnostic(normal):original._validation_start(normal,'h')
    with isolated_inner_diagnostic(short):changed._validation_start(short,'h')
    assert torch.equal(torch.get_rng_state(),rng)
    assert (tmp_path/'short/heldout_h_outer001.pt').exists()
    assert not torch.equal(original.validation['h']['data_1'],changed.validation['h']['data_1'])
    own=changed.validation['h']
    changed.comparison_h_validation=original.validation['h']
    changed.comparison_h_validation_path=str(tmp_path/'normal/heldout_h_outer001.pt')
    matched=changed._matched_h_loss(short)
    assert changed.validation['h'] is own
    changed.validation['h']=original.validation['h']
    expected=changed._validation_loss(short,'h')
    assert matched['heldout_loss']==expected['heldout_loss']
    assert matched['bank']=='identical normal-outer1 frozen endpoints/bridge draws'


def test_next_outer_failure_prevents_pair_even_if_first_outer_passes():
    report,contract=fixture();report['provenance']={'source':'normal'}
    cases=make_one_axis_cases(report,screen(report,contract),Path('/immutable/normal'))
    policy=deepcopy(report['diagnostics']['inner_boundaries'][-1]['policy'])
    def result():
        return dict(diagnostics={'first_outer_complete':deepcopy(policy),'trained_policy':deepcopy(policy),
            'first_outer_validation':{'heldout_loss':1.},'next_outer_validation':{'heldout_loss':1.}},
            cost={'total_openmm_configurations':1},new_optimizer_steps=1,cached_global_step=1,
            cached_reference_paid_openmm=285376)
    reports={name:result() for name in cases}
    reports['u_short']['diagnostics']['trained_policy']['raw_path_ess_fraction']=.1
    decision=compare_one_axis(reports,dict(cases=cases,curve_contract=contract))
    assert decision['passing_shorter_cases']==['h_short']
    assert decision['pair_status']=='not_requested_one_axis_failed_or_not_proposed'
    assert decision['minimum_inner_duration']=='not_established'


def test_dispatch_ledger_preserves_unknown_outcome_and_never_repeats_jobs(tmp_path,monkeypatch):
    plan={'cases':{'normal':{},'u_short':{}},'source_manifest_sha256':'source'}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    commands=[]
    def submit(command):
        commands.append(command)
        return SimpleNamespace(stdout=str(1000+len(commands))+';cluster\n')
    monkeypatch.setattr(monitor,'execute',submit)
    rows=monitor.submit_cases(tmp_path,plan,'remote/python','source')
    assert [r['job_id'] for r in rows]==['1001','1002']
    monitor.submit_cases(tmp_path,plan,'remote/python','source')
    assert len(commands)==2
    # An ambiguous intent is a reconciliation requirement, never permission
    # to rerun sbatch after a connection drop.
    other=tmp_path/'unknown';other.mkdir();(other/'plan.json').write_text(json.dumps(plan))
    (other/'submission_intents').mkdir()
    (other/'submission_intents/normal_seed0.json').write_text('{"status":"dispatching"}')
    with pytest.raises(RuntimeError,match='dispatch intent'):
        monitor.submit_cases(other,plan,'remote/python','source')
    assert len(commands)==2


def test_preparation_freezes_completed_audit_copy_not_mutable_monitor(tmp_path,monkeypatch):
    from omegaconf import OmegaConf
    report,contract=fixture();report['provenance']={'source':'normal'}
    report['cost']={'total_openmm_configurations':285376}
    normal=tmp_path/'normal';run=normal/'normal_inner_curve/seed0';run.mkdir(parents=True)
    cfg=gaussian_shared_fit().cfg
    OmegaConf.save(cfg,run/'config.yaml')
    parent=dict(frozen_inputs={},curve_contract=contract,backend_verification_path='/frozen/gpu/audit',
                backend_job_id='5194255',jax_backend_sha256='backend')
    (normal/'plan.json').write_text(json.dumps(parent))
    (run/'receipt.json').write_text(json.dumps({'files':{'config.yaml':sha256(run/'config.yaml')}}))
    audit=normal/'monitor_state.json';audit.write_text('{"status":"normal_curve_verified"}')
    monkeypatch.setattr(runner,'verified_inputs',lambda *args:(report,parent))
    output=tmp_path/'one_axis'
    plan=runner.prepare_one_axis(normal,audit,output,'source')
    digest=plan['frozen_inputs'][str(output/'normal_completion_audit.json')]
    audit.write_text('{"status":"normal_curve_verified","checked_at":"later"}')
    assert str(audit) not in plan['frozen_inputs']
    assert sha256(output/'normal_completion_audit.json')==digest
    with pytest.raises(RuntimeError,match='plan exists'):
        runner.prepare_one_axis(normal,audit,output,'source')
