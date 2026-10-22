"""Component isolation, finite-kernel disclosures and accounting; no ALDP training."""

from copy import deepcopy
from pathlib import Path
import json
import math
import sys

from hydra.utils import instantiate
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_aldp_component_campaign import (ComponentObserver, configuration, constant_reference_equivalence,
    kernel_contract, verify_component_report)
from run_aldp_stability_campaign import expected_cost
from tr_asbs.process.integrator import (EulerMaruyamaIntegrator, ExponentialIntegrator,
    DiffusionTimeEulerIntegrator, LinearExponentialIntegrator)
from tr_asbs.process.reference import BrownianReference, LinearGaussianReference
from tr_asbs.process.noise_schedule import AnalyticNoiseSchedule
from tr_asbs.process.sde import VESDE


def plan(family, variants):
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/'reports/aldp_chart_20261007_5a093a95/physical_stability_v1/plan.json').read_text())
    p['study_family']=family
    p['component_diagnostics']=dict(integrated_diffusion_variance=.249999,noise_knots=200,ou_rate=1.)
    p['candidates']={name:dict(chart_key='low_density',source_scale=1.,terminal_gradient_clip=1000.,
        outer_loops=10,openmm_cap=120000,fc_loss_mode='log_variance',**fields)
        for name,fields in variants.items()}
    return p


def science(cfg):
    cfg=OmegaConf.to_container(cfg,resolve=True)
    for key in ('root','name','checkpoint_directory','sample_directory'):
        cfg.pop(key)
    cfg['fabric']['loggers'].pop('save_dir')
    return cfg


def roundtrip_report(count):
    return dict(schema='aldp_cartesian_roundtrip_v1',status='passed',error_nm=1e-7,
        sample_count=count,tolerance_nm=2e-5,additional_openmm_configurations=0,
        samples_modified=False,samples_filtered=False)


def test_solver_screen_only_changes_integrator_class(tmp_path):
    p=plan('solver',dict(legacy={'solver':'legacy'},euler={'solver':'euler'},tau={'solver':'diffusion_euler'}))
    baseline,_=configuration(p,tmp_path,'mechanism_pilot','legacy',0)
    baseline=science(baseline)
    for name in ('euler','tau'):
        cfg,_=configuration(p,tmp_path,'mechanism_pilot',name,0)
        cfg=science(cfg)
        assert [k for k in cfg if cfg[k]!=baseline[k]]==['integrator']
        assert [k for k in cfg['integrator'] if cfg['integrator'][k]!=baseline['integrator'][k]]==['_target_']


def test_noise_screen_fixes_solver_grid_loss_and_noise_knots(tmp_path):
    p=plan('noise',{k:dict(solver='diffusion_euler',noise_shape=k) for k in ('geometric','constant','cosine')})
    configs={}
    for name in p['candidates']:
        cfg,_=configuration(p,tmp_path,'mechanism_pilot',name,0)
        configs[name]=science(cfg)
        smoke,_=configuration(p,tmp_path,'mechanism_smoke',name,0)
        assert smoke.integrator.timesteps.steps==4 and cfg.integrator.timesteps.steps==200
        if name=='cosine':
            assert cfg.base_sde.noise_schedule.process.num_steps==200
            assert smoke.base_sde.noise_schedule.process.num_steps==200
        base=instantiate(cfg.base_sde)
        assert float(base.total_variance)==pytest.approx(.249999,rel=2e-6)
    for name in ('constant','cosine'):
        assert [k for k in configs[name] if configs[name][k]!=configs['geometric'][k]]==['base_sde']


def test_reference_screen_changes_rate_only_and_excludes_vp(tmp_path):
    p=plan('reference',{k:dict(solver='linear_exponential',noise_shape='constant',reference_kind=k)
                        for k in ('brownian','ou')})
    configs=[science(configuration(p,tmp_path,'mechanism_pilot',k,0)[0]) for k in p['candidates']]
    assert [k for k in configs[0] if configs[0][k]!=configs[1][k]]==['base_sde']
    assert [k for k in configs[0]['base_sde'] if configs[0]['base_sde'][k]!=configs[1]['base_sde'][k]]==['rate']
    p['candidates']['ou']['reference_kind']='vp'
    with pytest.raises(ValueError,match='VP excluded'):
        configuration(p,tmp_path,'mechanism_pilot','ou',0)


def test_confounded_component_settings_are_rejected(tmp_path):
    p=plan('noise',dict(a=dict(solver='euler'),b=dict(solver='diffusion_euler',noise_shape='constant')))
    with pytest.raises(ValueError,match='fixes one solver'):
        configuration(p,tmp_path,'mechanism_pilot','a',0)
    p['candidates']['b']['solver']='euler'
    p['candidates']['b']['bc_loss_weight']='reference_variance'
    with pytest.raises(ValueError,match='nuisance settings'):
        configuration(p,tmp_path,'mechanism_pilot','a',0)


def test_cost_contract_counts_logvar_buffer_without_terminal_force():
    p=plan('loss_bridge',dict(logvar={}))
    cost=expected_cost(p,'mechanism_pilot','logvar')
    assert cost['training_and_online_openmm_configurations']==29504
    assert cost['total_openmm_configurations']==42688
    assert expected_cost(p,'mechanism_smoke','logvar')['total_openmm_configurations']==80
    p['candidates']['logvar']['initial_adjoint_matching_epochs']=500
    assert expected_cost(p,'mechanism_pilot','logvar')['total_openmm_configurations']==105664
    p['candidates']['logvar']['fc_loss_mode']='path_kl'
    with pytest.raises(ValueError,match='No registered OpenMM'):
        expected_cost(p,'mechanism_pilot','logvar')


@pytest.mark.parametrize('cls',[EulerMaruyamaIntegrator,ExponentialIntegrator,DiffusionTimeEulerIntegrator])
def test_constant_brownian_cached_kernel_matches_zero_rate_linear(cls):
    variance=.249999
    brownian=cls(BrownianReference(AnalyticNoiseSchedule(kind='constant',total_variance=variance)),steps=4)
    linear=LinearExponentialIntegrator(LinearGaussianReference(kind='ou',rate=0.,diffusion_scale=variance**.5),steps=4)
    a,b=kernel_contract(brownian),kernel_contract(linear)
    for key in ('interval_forward_phi_k_q','interval_reverse_phi_k_q'):
        torch.testing.assert_close(torch.tensor(a[key]),torch.tensor(b[key]),rtol=2e-6,atol=2e-7)
    audit=constant_reference_equivalence(brownian,linear)
    assert audit['status']=='passed' and audit['additional_openmm_configurations']==audit['additional_sampled_paths']==0


def test_kernel_contract_distinguishes_integrated_and_discrete_euler_variance():
    state=torch.get_rng_state().clone()
    euler=kernel_contract(EulerMaruyamaIntegrator(VESDE(.001,.5),steps=200))
    exact=kernel_contract(ExponentialIntegrator(VESDE(.001,.5),steps=200))
    assert torch.equal(state,torch.get_rng_state())
    assert euler['integrated_diffusion_variance']==pytest.approx(exact['integrated_diffusion_variance'])
    assert euler['forward_composed']['conditional_noise_variance']>1.02*euler['integrated_diffusion_variance']
    assert exact['forward_composed']['conditional_noise_variance']==pytest.approx(exact['integrated_diffusion_variance'],rel=2e-5)
    assert euler['additional_sampled_paths']==euler['additional_openmm_configurations']==0


def test_ou_contract_reports_covariance_and_source_propagation_limit():
    variance=.249999
    d=kernel_contract(LinearExponentialIntegrator(LinearGaussianReference(kind='ou',rate=1.,diffusion_scale=variance**.5),steps=4))
    assert d['forward_composed']['phi']==pytest.approx(math.exp(-1),rel=2e-6)
    assert d['forward_composed']['conditional_noise_variance']==pytest.approx(variance*(1-math.exp(-2))/2,rel=2e-6)
    assert d['backward_composed']['conditional_noise_variance']==pytest.approx(variance*(math.exp(2)-1)/2,rel=2e-6)
    assert d['conditional_noise_only'] and 'nonGaussian' in d['source_note']


def test_zero_rate_reference_preserves_brownian_bc_bridge_and_scores():
    variance=.249999
    brownian=BrownianReference(AnalyticNoiseSchedule(kind='constant',total_variance=variance))
    linear=LinearGaussianReference(kind='ou',rate=0.,diffusion_scale=variance**.5)
    x0=torch.tensor([[1.,2.],[-3.,.5],[.2,.7]])
    x1=x0*.4
    times=torch.tensor([.2,.5,.8])
    with torch.random.fork_rng():
        torch.manual_seed(912)
        a=brownian.sample_posterior(times,x0,x1)
        torch.manual_seed(912)
        b=linear.sample_posterior(times,x0,x1)
    torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(brownian.cond_score_t0(times,x0,a),linear.cond_score_t0(times,x0,a),rtol=2e-6,atol=2e-6)


def test_component_verifier_rejects_wrong_effective_configuration(tmp_path):
    p=plan('noise',dict(constant=dict(solver='diffusion_euler',noise_shape='constant')))
    cfg,_=configuration(p,tmp_path,'mechanism_smoke','constant',0)
    solver=instantiate(cfg.integrator,sde=instantiate(cfg.base_sde))
    metadata={'kernel':solver.kernel_metadata()}
    report=dict(seed=0,sampling_metadata=metadata,native_cartesian_roundtrip_error=1e-7,
        native_cartesian_roundtrip_diagnostic=roundtrip_report(4),diagnostics=dict(component_contract=dict(
        fc_loss_mode='log_variance',corrector_matching_mode='full',terminal_force_clip_used_by_fc=False,
        sampling_metadata=deepcopy(metadata),base_sde_config=OmegaConf.to_container(cfg.base_sde,resolve=True),
        integrator_config=OmegaConf.to_container(cfg.integrator,resolve=True),kernel=kernel_contract(solver))))
    verify_component_report(report,p,'mechanism_smoke','constant')
    report['diagnostics']['component_contract']['base_sde_config']['noise_schedule']['kind']='linear'
    with pytest.raises(RuntimeError,match='Resolved component configuration'):
        verify_component_report(report,p,'mechanism_smoke','constant')


@pytest.mark.parametrize('family,fields',[
    ('loss_bridge',dict(solver='legacy')),
    ('solver',dict(solver='euler')),
    ('solver',dict(solver='diffusion_euler')),
    ('noise',dict(solver='diffusion_euler',noise_shape='constant')),
    ('noise',dict(solver='diffusion_euler',noise_shape='cosine')),
    ('reference',dict(solver='linear_exponential',noise_shape='constant',reference_kind='ou')),
])
def test_component_initial_observer_uses_real_module_configuration(monkeypatch,tmp_path,family,fields):
    from types import SimpleNamespace
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    from run_aldp_stability_campaign import StabilityObserver

    p=plan(family,dict(cell=fields))
    cfg,_=configuration(p,tmp_path,'mechanism_smoke','cell',0)
    # Exercise the real module's attribute and sampling_metadata contract without
    # initializing a molecular potential or running any training/energy calls.
    module=GammaASBSModule.__new__(GammaASBSModule)
    module.cfg=cfg
    module.fc_loss_mode=cfg.fc_loss_mode
    module.integrator=instantiate(cfg.integrator,sde=instantiate(cfg.base_sde))
    module.terminal_cost=SimpleNamespace(gradient_clip_accumulation_dtype='float64')
    module.potential=SimpleNamespace(data_metadata=None)
    assert not hasattr(module,'corrector_matching_mode')
    monkeypatch.setattr(StabilityObserver,'__call__',lambda self,module,event,payload=None:None)
    observer=ComponentObserver.__new__(ComponentObserver)
    observer.report={}
    observer(module,'initial')
    report=dict(seed=0,sampling_metadata=module.sampling_metadata(),diagnostics=observer.report,
        native_cartesian_roundtrip_error=1e-7,native_cartesian_roundtrip_diagnostic=roundtrip_report(4))
    verify_component_report(report,p,'mechanism_smoke','cell')
    assert observer.report['component_contract']['corrector_matching_mode']==cfg.corrector_matching_mode=='full'


def test_smoke_requires_roundtrip_but_full_pilot_retains_inverse_failure(tmp_path):
    p=plan('loss_bridge',dict(cell=dict(solver='legacy')))
    for stage,count in (('mechanism_smoke',4),('mechanism_pilot',1024)):
        cfg,_=configuration(p,tmp_path,stage,'cell',0)
        solver=instantiate(cfg.integrator,sde=instantiate(cfg.base_sde))
        metadata={'kernel':solver.kernel_metadata()}
        failure=roundtrip_report(count)
        failure.update(status='strict_inverse_rejected',error_nm=None,
            inverse_error='IC inverse chart received a dihedral outside its fundamental interval')
        report=dict(seed=0,sampling_metadata=metadata,native_cartesian_roundtrip_error=None,
            native_cartesian_roundtrip_diagnostic=failure,diagnostics=dict(component_contract=dict(
            fc_loss_mode='log_variance',corrector_matching_mode='full',terminal_force_clip_used_by_fc=False,
            sampling_metadata=deepcopy(metadata),base_sde_config=OmegaConf.to_container(cfg.base_sde,resolve=True),
            integrator_config=OmegaConf.to_container(cfg.integrator,resolve=True),kernel=kernel_contract(solver))))
        if stage=='mechanism_smoke':
            with pytest.raises(RuntimeError,match='smoke Cartesian roundtrip failed'):
                verify_component_report(report,p,stage,'cell')
        else:
            verify_component_report(report,p,stage,'cell')
            failure['samples_filtered']=True
            with pytest.raises(RuntimeError,match='diagnostic incomplete'):
                verify_component_report(report,p,stage,'cell')
