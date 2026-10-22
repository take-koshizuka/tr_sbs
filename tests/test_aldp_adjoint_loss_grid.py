
import pytest

pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_aldp_adjoint_loss_grid import configuration,expected_cost,loss_matrix
from analyze_aldp_loss_pairs import compare_loss_pairs


def plan():
    return dict(fc_loss_mode='cross_entropy',matrix=loss_matrix(),
        base_config_relative='scripts/aldp_noise_solver_baseline.yaml',integrated_diffusion_variance=.249999,outer_loops=5)


def test_source_and_loss_compatibility_keeps_all_cells_and_reasons():
    original,lv=loss_matrix(),loss_matrix('log_variance')
    assert len(original)==len(lv)==35
    assert sum(not c['incompatible_reason'] for c in original)==15
    assert sum(not c['incompatible_reason'] for c in lv)==18
    assert all(c['incompatible_reason'] for c in original if c['preset'] in ('ou','vp'))
    assert all(c['fc_loss_mode']=='cross_entropy' for c in original)


@pytest.mark.parametrize('candidate',[c['candidate'] for c in loss_matrix() if not c['incompatible_reason']])
def test_loss_only_change_and_independent_cost_boundary(tmp_path,candidate):
    p=plan();cfg,_=configuration(p,tmp_path,'full',candidate)
    assert cfg.fc_loss_mode=='cross_entropy' and cfg.bc_loss_weight=='legacy'
    assert cfg.gamma_schedule=='trust_region' and cfg.per_time_anchor_target
    assert (cfg.adjoint_matching_epochs,cfg.corrector_matching_epochs,cfg.steps_per_epoch,cfg.max_outer_loops)==(2000,1000,100,5)
    assert cfg.source.scale==1 and cfg.potential.energy_mode=='physical'
    assert cfg.terminal_cost.grad_clip_val==1000 and cfg.integrator.steps==200
    ce=expected_cost(cfg,128)
    assert ce['optimizer_steps']==1500000
    assert ce['total_openmm_configurations']==3854528
    cfg.fc_loss_mode='log_variance'
    lv=expected_cost(cfg,128)
    assert lv['total_openmm_configurations']==1294528
    assert ce['total_openmm_configurations']-lv['total_openmm_configurations']==5*2000*4*64
    smoke,_=configuration(p,tmp_path,'smoke',candidate)
    assert expected_cost(smoke,8)['total_openmm_configurations']==584


def test_native_adjoint_function_is_weighted_three_time_matching_at_gamma_one():
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    x0=torch.tensor([[0.,1.],[1.,2.]])
    x1=x0+2
    terminal=torch.tensor([[2.,-1.],[1.,3.]])
    iw=torch.tensor([.5,1.5])
    module=SimpleNamespace(cfg=SimpleNamespace(train_batch_size=2),zeros_tensor=torch.tensor(0.),ones_tensor=torch.tensor(1.),
        fabric=SimpleNamespace(to_device=lambda x:x),gamma=1.,bridge_time_epsilon=.001,per_time_anchor_target=False,
        fc_loss_weight='legacy',base_sde=SimpleNamespace(sample_posterior=lambda **kw:(kw['data_0']+kw['data_1'])/2),
        controller=lambda t,x:2*x,_compute_iw=lambda batch,gamma:iw)
    loss=GammaASBSModule.adjoint_matching_step(module,dict(data_0=x0,data_1=x1,terminal_cost=terminal,control_1=torch.ones_like(x0)))['loss']
    expected=sum((((2*x+terminal).square().mean(-1))*iw).mean() for x in (x0,(x0+x1)/2,x1))
    torch.testing.assert_close(loss,expected)


def report(mode):
    keys=('phi_psi_tvd','torsion_js_mean','physical_energy_w2','coarse_basin_tvd','missing_reference_mass')
    return dict(status='artifact_complete',rejected_training_steps=0,global_step=1500000,seed=0,calibration_sha256='bank',
        sampling_metadata=dict(fc_loss_mode=mode,kernel={'preset':'geometric'},source='fixed',bc_loss_weight='legacy'),
        metrics={bank:dict.fromkeys(keys,.3) for bank in ('a','b')},cost={'wall_seconds':5},diagnostics=dict(phase_end=[],dual=[]))


def test_pair_uses_physical_A_not_different_objective_or_B():
    a,b=report('cross_entropy'),report('log_variance')
    a['metrics']['a']['torsion_js_mean']=.2
    a['metrics']['b']['torsion_js_mean']=999
    pair=compare_loss_pairs({'case':a},{'case':b})
    assert pair['rows'][0]['physical_A_original_minus_LV']['torsion_js_mean']==pytest.approx(-.1)
    assert not pair['objective_values_used_for_selection'] and pair['test_bank']=='closed'
    assert compare_loss_pairs({'case':a},{})['rows'][0]['status']=='pending_verified_full_pair'


@pytest.mark.parametrize('field',['rejected_training_steps','global_step','calibration_sha256'])
def test_failed_or_mismatched_pair_is_rejected(field):
    a,b=report('cross_entropy'),report('log_variance')
    a[field]='different' if field=='calibration_sha256' else 1
    with pytest.raises(ValueError):compare_loss_pairs({'case':a},{'case':b})


def test_stage_release_and_no_full_dispatch_from_smoke_plan(tmp_path,monkeypatch):
    import monitor_aldp_adjoint_loss_grid as monitor
    p=plan();p.update(allowed_stages=['smoke'],max_concurrent_gpu_jobs=2)
    (tmp_path/'plan.json').write_text(json.dumps(p))
    rows=[dict(candidate=c['candidate'],stage='smoke') for c in loss_matrix() if not c['incompatible_reason']]
    (tmp_path/'ledger.json').write_text(json.dumps(rows))
    monkeypatch.setattr(monitor.shared,'pass_once',lambda *a,**kw:dict(failures=[],active=[],verified=rows,consecutive_active=0,complete=False))
    monkeypatch.setattr(monitor,'submit',lambda *args:pytest.fail('All smoke rows cached: neither duplicate nor full dispatch'))
    result=monitor.pass_once(tmp_path,'python',advance=True)
    assert result['all_original_smokes_verified'] and not result['complete']
    assert result['full_release']=='needs_new_registered_campaign'


def test_full_requires_pinned_loss_specific_smoke_set(tmp_path):
    import monitor_aldp_adjoint_loss_grid as monitor
    p=plan();p.update(allowed_stages=['full'],max_concurrent_gpu_jobs=2)
    (tmp_path/'plan.json').write_text(json.dumps(p))
    with pytest.raises(RuntimeError,match='all pinned'):monitor.pass_once(tmp_path,'python',advance=True)


def test_duplicate_intent_is_rejected_before_scheduler_call(tmp_path):
    import monitor_aldp_adjoint_loss_grid as monitor
    (tmp_path/'intents').mkdir()
    (tmp_path/'intents/smoke__case.json').write_text('{}')
    with pytest.raises(RuntimeError,match='duplicate'):
        monitor.submit(tmp_path,dict(allowed_stages=['smoke']),'python','smoke','case',[])
