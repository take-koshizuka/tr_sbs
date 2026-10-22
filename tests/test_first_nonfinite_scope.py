import torch,json
from run_aldp_first_nonfinite import safe,audit

def test_nonfinite_loss_saved_without_json_failure():
    x=audit(torch.tensor(float('inf')));assert x['nonfinite']==1;assert json.loads(json.dumps(safe(x),allow_nan=False))['max_abs']=='+inf'

def test_finite_gradient_norm_overflow_is_not_nonfinite_elements():
    x=torch.tensor([1e30]);assert audit(x)['nonfinite']==0;assert not torch.isfinite(x.square().sum());assert torch.isfinite(x.double().square().sum())

def test_capture_stops_before_optimizer_update(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tr_asbs.experiment.train_gamma_asbs_iw as shared
    from run_aldp_first_nonfinite import install,Captured
    import pytest
    model=torch.nn.Linear(1,1);opt=torch.optim.Adam(model.parameters());before=model.weight.detach().clone()
    buffer=SimpleNamespace(storage={},metadata={},max_size=1)
    m=SimpleNamespace(controller=model,corrector=model,prev_integrator=model,prev_corrector=model,controller_optimizer=opt,corrector_optimizer=opt,adjoint_buffer=buffer,corrector_buffer=buffer,fabric=SimpleNamespace(device=torch.device('cpu')),current_epoch=0,global_step=0,outer_step=0,phase='u',phase_epoch=0,gamma=1.,V0=0.,gamma_h=1.,potential=SimpleNamespace(aldp_potential=SimpleNamespace(openmm_configurations_evaluated=0)),sampling_metadata=lambda:{})
    m.adjoint_matching_log_variance_step=lambda b:{'loss':torch.tensor(float('inf'))}
    m.training_step=lambda b,t:m.adjoint_matching_log_variance_step(b)
    install(m,tmp_path,{'stop_epoch':1100,'OpenMM_cap':150000})
    with pytest.raises(Captured):m.training_step({'x':torch.ones(1)},'adjoint_matching')
    assert torch.equal(before,model.weight) and not opt.state
    w=torch.load(tmp_path/'witness.pt',weights_only=False);assert w['reason']=='loss_nonfinite';assert torch.equal(w['batch']['x'],torch.ones(1));assert 'pre_forward_rng' in w
