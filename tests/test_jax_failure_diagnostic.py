"""Tiny numerical/replay checks; no ALDP construction or simulation."""
from pathlib import Path
import sys
from types import MethodType

import jax
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_aldp_jax_failure_diagnostic import (
    finite_number, lv_loss_float64_reduction, run_bounded_u_replay,
)
from test_jax_inner import context, model
from test_jax_inner_curve import gaussian_shared_fit
from tr_asbs.experiment.inner_observation import observe_inner_boundaries, save_inner_snapshot
from tr_asbs.experiment.jax_inner import export_fourier, lv_loss, prepare_lv_batch
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule


def test_finite_extreme_log_weights_identify_variance_range_overflow():
    network=model();module=context(network)
    batch=dict(data_traj=torch.randn(5,5,3)*.2,
        log_Ph_Pref_raw=torch.tensor([0.,1e23,2e23,3e23,4e23]))
    _,params,coeff,spec=export_fourier(network)
    _,old,_,_=export_fourier(module.prev_integrator.sde.u_controller)
    prepared=prepare_lv_batch(module,batch)
    torch_loss=GammaASBSModule.adjoint_matching_log_variance_step(module,batch)['loss']
    original=lv_loss(params,old,coeff,spec,prepared,module.gamma)[0]
    widened=lv_loss_float64_reduction(params,old,coeff,spec,prepared,module.gamma)
    assert torch.isinf(torch_loss) and np.isinf(original) and np.isfinite(widened)
    expected=np.var(module.gamma*batch['log_Ph_Pref_raw'].numpy().astype(np.float64),ddof=1)
    np.testing.assert_allclose(widened,expected,rtol=1e-7)
    assert finite_number(torch_loss)=='inf'


def test_u0_replay_matches_actual_shared_first_epoch_and_rng(tmp_path):
    provenance={'test':'exact u0 diagnostic replay'}
    reference=gaussian_shared_fit(u_epochs=3)
    torch.manual_seed(571)
    def stop_at_one(payload):
        if payload['phase']=='u' and payload['phase_epoch']==1:
            raise RuntimeError('reference boundary')
    observe_inner_boundaries(reference,stop_at_one,{'u':[1],'h':[]})
    with pytest.raises(RuntimeError,match='reference boundary'):
        reference.fit()
    final_rng=torch.get_rng_state().clone()
    start=gaussian_shared_fit(u_epochs=3);torch.manual_seed(571)
    snapshot=tmp_path/'u0.pt'
    def save_zero(payload):
        if payload['phase']=='u' and payload['phase_epoch']==0:
            save_inner_snapshot(start,snapshot,provenance)
            raise RuntimeError('u0 boundary')
    observe_inner_boundaries(start,save_zero,{'u':[],'h':[]})
    with pytest.raises(RuntimeError,match='u0 boundary'):
        start.fit()
    replay=gaussian_shared_fit(u_epochs=3)
    progress=[]
    status=run_bounded_u_replay(replay,snapshot,provenance,1,
        lambda batch:pytest.fail('unexpected rejection'), lambda module:progress.append(module.global_step+1))
    assert status=='no_rejection_within_registered_replay_limit'
    assert progress==[1] and replay.global_step==reference.global_step==2
    assert torch.equal(torch.get_rng_state(),final_rng)
    for left,right in zip(jax.tree.leaves(reference.test_backend.params['u']),
                          jax.tree.leaves(replay.test_backend.params['u'])):
        np.testing.assert_array_equal(left,right)
    for left,right in zip(jax.tree.leaves(reference.test_backend.states['u']),
                          jax.tree.leaves(replay.test_backend.states['u'])):
        np.testing.assert_array_equal(left,right)
    for key in reference.adjoint_buffer.storage:
        torch.testing.assert_close(torch.stack(reference.adjoint_buffer.storage[key]),
            torch.stack(replay.adjoint_buffer.storage[key]),rtol=0,atol=0)


def test_rejection_capture_on_tiny_actual_backend(tmp_path):
    from run_aldp_jax_failure_diagnostic import diagnose_rejection
    from tr_asbs.experiment.jax_inner import prepare_lv_batch
    module=gaussian_shared_fit()
    module.adjoint_matching_log_variance_step=MethodType(
        GammaASBSModule.adjoint_matching_log_variance_step,module)
    module.global_step=0;module.current_epoch=0;module.phase='u';module.phase_epoch=0
    backend=module.test_backend
    batch=dict(data_traj=torch.randn(5,5,3)*.2,
        log_Ph_Pref_raw=torch.tensor([0.,1e23,2e23,3e23,4e23]))
    prepared=prepare_lv_batch(module,batch)
    _,old,_,_=export_fourier(module.prev_integrator.sde.u_controller)
    args=(backend.params['u'],backend.states['u'],prepared,old,module.gamma)
    _,_,values=backend.updates['u'](*args)
    report=diagnose_rejection(module,backend,batch,(*args,values),tmp_path)
    assert report['jax_original']['loss']=='inf'
    assert report['torch_same_jax_weights_batch']['loss']=='inf'
    assert isinstance(report['jax_float64_variance_diagnostic']['loss'],float)
    saved=torch.load(tmp_path/'rejected_batch.pt',weights_only=False)
    assert saved['schema']=='aldp_numerical_rejection_batch_v1' and not saved['exact_mid_epoch_resume']
    assert saved['adam']['count']==0
