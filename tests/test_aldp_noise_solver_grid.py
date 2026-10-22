"""Finite kernel compatibility, fixed scientific axes and cost guards; no MD."""
from pathlib import Path
import sys
import pytest
import torch
from hydra.utils import instantiate

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_aldp_noise_solver_grid import matrix, configuration, expected_cost, incompatibility
from run_aldp_component_campaign import kernel_contract


def plan():
    return dict(matrix=matrix(),base_config_relative='scripts/aldp_noise_solver_baseline.yaml',
                integrated_diffusion_variance=.499998999999**2,outer_loops=5)


def test_grid_contains_all_presets_and_midpoint():
    rows=matrix()
    assert len(rows)==35
    assert len({r['candidate'] for r in rows})==35
    assert sum(r['incompatible_reason'] is None for r in rows)==21
    assert incompatibility('ou','legacy')
    assert incompatibility('vp','diffusion_euler')
    with pytest.raises(ValueError): incompatibility('new','euler')


@pytest.mark.parametrize('cell',[r for r in matrix() if r['incompatible_reason'] is None],ids=lambda r:r['candidate'])
def test_registered_kernels_and_asymmetric_normal_training(tmp_path,cell):
    p=plan()
    if cell['preset']=='vp':
        with pytest.raises(ValueError,match='stationary Gaussian'): configuration(p,tmp_path,'full',cell['candidate'])
        return
    cfg,_=configuration(p,tmp_path,'full',cell['candidate'])
    assert (cfg.adjoint_matching_epochs,cfg.corrector_matching_epochs,cfg.steps_per_epoch,cfg.max_outer_loops)==(2000,1000,100,5)
    assert cfg.source.scale==1 and cfg.potential.energy_mode=='physical'
    assert cfg.bc_loss_weight=='legacy' and cfg.fc_loss_mode=='log_variance'
    assert cfg.terminal_cost.grad_clip_val==1000 and cfg.seed==0
    assert cfg.integrator.steps==200 and cfg.integrator.grid=='uniform_t'
    base=instantiate(cfg.base_sde)
    integrator=instantiate(cfg.integrator,sde=base)
    kernel=kernel_contract(integrator)
    assert kernel['integrated_diffusion_variance']==pytest.approx(p['integrated_diffusion_variance'],rel=2e-6)
    assert kernel['forward_composed']['conditional_noise_variance']>0
    assert kernel['backward_composed']['conditional_noise_variance']>0
    if cell['solver'].endswith('_midpoint'): assert integrator.control_evaluation=='midpoint'
    cost=expected_cost(cfg,128)
    assert cost['optimizer_steps']==1500000
    assert cost['total_openmm_configurations']==1294528
    smoke,_=configuration(p,tmp_path,'smoke',cell['candidate'])
    assert smoke.integrator.steps==200
    assert list(smoke.num_samples_for_dual_loss)==[32,2]
    assert smoke.num_samples_for_dual_loss[0]//(smoke.inference_batch_size//smoke.num_samples_for_dual_loss[1])==1
    assert expected_cost(smoke,8)['total_openmm_configurations']==328


def test_incompatible_cells_cannot_be_configured(tmp_path):
    for cell in matrix():
        if cell['incompatible_reason']:
            with pytest.raises(ValueError): configuration(plan(),tmp_path,'full',cell['candidate'])


def test_observer_constructs_before_shared_runner_saves_config(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from run_aldp_noise_solver_grid import GridObserver
    from run_aldp_stability_campaign import StabilityObserver
    import run_aldp_noise_solver_grid as runner
    monkeypatch.setattr(StabilityObserver,'__init__',lambda self,*args: setattr(self,'report',{}))
    campaign=tmp_path/'results';output=campaign/'smoke'/'case';output.mkdir(parents=True)
    (campaign/'plan.json').write_text('{}')
    calibration=tmp_path/'calibration';calibration.write_bytes(b'fixed')
    cfg=SimpleNamespace(eval_samples_final=8,potential=SimpleNamespace(chart_key='low_density',calibration_bundle_path=str(calibration)))
    observer=GridObserver({},output,cfg,{},'source','smoke','case')
    assert not (output/'config.yaml').exists()
    assert 'config_sha256' not in observer.provenance
    (output/'config.yaml').write_text('resolved: true\n')
    monkeypatch.setattr(runner,'kernel_contract',lambda integrator: {})
    monkeypatch.setattr(StabilityObserver,'__call__',lambda *args: None)
    observer(SimpleNamespace(integrator=None),'initial')
    assert observer.provenance['config_sha256']==runner.sha256(output/'config.yaml')


def test_zero_or_truncated_dual_collection_is_rejected(tmp_path):
    cfg,_=configuration(plan(),tmp_path,'smoke','ou__euler')
    for groups,paths in ((2,2),(33,2),(32,3),(0,8)):
        cfg.num_samples_for_dual_loss=[groups,paths]
        with pytest.raises(ValueError,match='Dual bank'): expected_cost(cfg,8)


def test_shifted_chart_grid_excludes_vp_without_changing_original_reasons(tmp_path):
    generic=matrix(); shifted=matrix(True)
    assert sum(c['incompatible_reason'] is None for c in shifted)==18
    for before,after in zip(generic,shifted):
        if before['incompatible_reason']: assert before==after
    p=plan();p['matrix']=shifted
    for cell in shifted:
        if cell['preset']=='vp':
            with pytest.raises(ValueError): configuration(p,tmp_path,'full',cell['candidate'])


def test_cached_set_guard_rejects_missing_evidence(tmp_path):
    from monitor_aldp_noise_solver_grid import cached_smokes
    assert cached_smokes({})==[]
    p=dict(cached_smokes=dict(campaign=str(tmp_path),evidence_files={'plan.json':'wrong'}))
    (tmp_path/'plan.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Cached evidence changed'): cached_smokes(p)
