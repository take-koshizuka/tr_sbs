"""Exact uninterrupted versus resumed shared JAX fit on a tiny Gaussian."""
from copy import deepcopy
import random

import jax
import numpy as np
import pytest
import torch

from test_jax_inner_curve import gaussian_shared_fit
from tr_asbs.experiment.inner_observation import (
    attach_exact_inner_resume, continuation_training_contract, observe_inner_boundaries,
    restore_inner_snapshot, save_inner_snapshot,
)


@pytest.mark.parametrize('phase', ['u', 'h'])
def test_exact_shared_fit_continuation_preserves_updates_buffers_and_rng(tmp_path, phase):
    full = gaussian_shared_fit()
    torch.manual_seed(9876);np.random.seed(23);random.seed(45)
    full.fit()
    final_rng = torch.get_rng_state().clone()
    split = gaussian_shared_fit()
    torch.manual_seed(9876);np.random.seed(23);random.seed(45)
    path = tmp_path/'prefix.pt'
    provenance = {'source':'tiny-gaussian','config':'same training contract'}

    def checkpoint(payload):
        if payload['outer_loop'] == 1 and payload['phase'] == phase and payload['phase_epoch'] == 1:
            # Save inside a diagnostic with deliberately different RNG.
            torch.manual_seed(991);torch.rand(13);np.random.rand(7);random.random()
            save_inner_snapshot(split,path,provenance)
            raise RuntimeError('unit-test interruption at exact completed boundary')

    observe_inner_boundaries(split,checkpoint,{'u':[1],'h':[1]})
    with pytest.raises(RuntimeError,match='unit-test interruption'):
        split.fit()
    resumed = gaussian_shared_fit()
    resumed.rank_seed = 12345
    marker = attach_exact_inner_resume(resumed,path,expected_provenance=provenance,
        expected_training_contract=continuation_training_contract(full.cfg),
        on_restored=lambda module:(torch.rand(31),np.random.rand(19),random.random()))
    resumed.fit()
    assert marker['restored'] and marker['initial_online_suppressed']
    assert resumed.global_step == full.global_step == 12
    assert resumed.current_epoch == full.current_epoch == 6
    assert torch.equal(torch.get_rng_state(),final_rng)
    assert resumed.rank_seed == full.rank_seed == 42
    for p in ('u','h'):
        for left,right in zip(jax.tree.leaves(full.test_backend.params[p]),
                              jax.tree.leaves(resumed.test_backend.params[p])):
            np.testing.assert_array_equal(left,right)
        for left,right in zip(jax.tree.leaves(full.test_backend.states[p]),
                              jax.tree.leaves(resumed.test_backend.states[p])):
            np.testing.assert_array_equal(left,right)
    for name in ('adjoint_buffer','corrector_buffer'):
        left,right=getattr(full,name),getattr(resumed,name)
        assert left.metadata == right.metadata and left.storage.keys() == right.storage.keys()
        for key in left.storage:
            torch.testing.assert_close(torch.stack(left.storage[key]),torch.stack(right.storage[key]),rtol=0,atol=0)


def test_optimizer_mismatch_is_rejected_before_model_mutation(tmp_path):
    source=gaussian_shared_fit();source.phase='u';source.phase_epoch=1
    path=tmp_path/'prefix.pt';save_inner_snapshot(source,path,{'test':'optimizer guard'})
    target=gaussian_shared_fit()
    target.corrector_optimizer.param_groups[0]['lr']=2e-4
    before=deepcopy(target.controller.state_dict())
    with pytest.raises(ValueError,match='optimizer configuration'):
        restore_inner_snapshot(target,path,expected_provenance={'test':'optimizer guard'})
    for name,value in before.items():
        torch.testing.assert_close(value,target.controller.state_dict()[name],rtol=0,atol=0)


def test_resume_rejects_unregistered_training_changes_and_too_late_prefix(tmp_path):
    source=gaussian_shared_fit();source.phase='u';source.phase_epoch=2
    path=tmp_path/'prefix.pt';save_inner_snapshot(source,path,{'test':'duration guard'})
    contract=continuation_training_contract(source.cfg)
    target=gaussian_shared_fit();target.cfg.train_batch_size=6
    with pytest.raises(ValueError,match='changes settings'):
        attach_exact_inner_resume(target,path,expected_provenance={'test':'duration guard'},
                                  expected_training_contract=contract)
    target=gaussian_shared_fit();target.initial_adjoint_matching_epochs=1
    with pytest.raises(ValueError,match='beyond the requested'):
        attach_exact_inner_resume(target,path,expected_provenance={'test':'duration guard'},
                                  expected_training_contract=contract)
