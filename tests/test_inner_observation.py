"""Actual inner-loop timing/RNG contracts with tiny buffers, without ALDP."""
from copy import deepcopy
import random
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from tr_asbs.data.buffer import BatchBuffer
from tr_asbs.experiment.inner_observation import (
    observe_inner_boundaries, restore_inner_snapshot, save_inner_snapshot,
)
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule


def tiny_loop():
    integrator = torch.nn.Linear(1, 1)
    module = SimpleNamespace(integrator=integrator, prev_integrator=deepcopy(integrator),
        controller=integrator, corrector=torch.nn.Linear(1, 1),
        adjoint_buffer=BatchBuffer(4), clear_adjoint_buffer_per_outer_loop=True,
        fabric=SimpleNamespace(device=torch.device("cpu"), global_rank=0),
        cfg=SimpleNamespace(adjoint_matching_epochs=3, corrector_matching_epochs=2,
            progress_bar=False, log_detail_interval_epoch=2, monitor_values=[]),
        current_epoch=0, global_step=0, outer_step=0, phase="ready", phase_epoch=0,
        total_epochs=5, corrector_gamma_mode="none", early_stop_requested=False,
        adjoint_matching_epochs=3, initial_adjoint_matching_epochs=3, trace=[])
    module.potential = SimpleNamespace()
    module.populate_buffer_adjoint_matching = lambda: module.trace.append(("populate_u", float(torch.rand(()))))
    module.populate_buffer_corrector_matching = lambda: module.trace.append(("populate_h", float(torch.rand(()))))
    module.create_dataloader = lambda task: [{}, {}]
    module.training_step = lambda batch, task: True
    module.log_details = lambda step: module.trace.append(("online", module.current_epoch, float(torch.rand(())))) or {}
    module.save_checkpoint = lambda **kwargs: None
    module._weight_rollback_check = lambda metrics: None
    module._update_divergence_early_stop = lambda metrics: None
    module.inner_loop_u = MethodType(GammaASBSModule.inner_loop_u, module)
    module.inner_loop_h = MethodType(GammaASBSModule.inner_loop_h, module)
    return module


def execute(module):
    module.inner_loop_u()
    module.phase, module.phase_epoch = "h", 0
    module.inner_loop_h()


def test_actual_completed_boundaries_include_online_rng_and_preserve_training():
    torch.manual_seed(142)
    baseline = tiny_loop()
    np.random.seed(23); random.seed(45)
    initial_torch = torch.get_rng_state().clone()
    execute(baseline)
    baseline_rng = torch.get_rng_state().clone()
    observed = tiny_loop()
    torch.set_rng_state(initial_torch)
    np.random.seed(23); random.seed(45)
    numpy_initial, python_initial = np.random.get_state(), random.getstate()
    events = []

    def diagnostic(payload):
        events.append(dict(payload, trace=tuple(observed.trace)))
        torch.rand(13); np.random.rand(11); random.random()
        observed.integrator.eval()

    observe_inner_boundaries(observed, diagnostic, {"u": [1, 2, 3], "h": [1, 2]})
    execute(observed)
    assert observed.trace == baseline.trace
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert np.array_equal(np.random.get_state()[1], numpy_initial[1])
    assert random.getstate() == python_initial
    assert observed.integrator.training
    assert [(e["phase"], e["phase_epoch"], e["epoch"], e["global_step"]) for e in events] == [
        ("u", 0, 0, 0), ("u", 1, 1, 2), ("u", 2, 2, 4), ("u", 3, 3, 6),
        ("h", 0, 3, 6), ("h", 1, 4, 8), ("h", 2, 5, 10)]
    assert events[2]["trace"][-1][:2] == ("online", 2)
    assert events[5]["trace"][-1][:2] == ("online", 4)


def test_only_selected_milestones_and_no_duplicate_attachment():
    module = tiny_loop()
    events = []
    observe_inner_boundaries(module, events.append, {"u": [2], "h": []})
    with pytest.raises(RuntimeError, match="already attached"):
        observe_inner_boundaries(module, events.append, {"u": [], "h": []})
    execute(module)
    assert [(e["phase"], e["phase_epoch"]) for e in events] == [
        ("u", 0), ("u", 2), ("u", 3), ("h", 0), ("h", 2)]


def snapshot_loop():
    module = tiny_loop()
    module.prev_corrector = deepcopy(module.corrector)
    module.controller_optimizer = torch.optim.Adam(module.controller.parameters(), lr=1e-4)
    module.corrector_optimizer = torch.optim.Adam(module.corrector.parameters(), lr=1e-4)
    module.corrector_buffer = BatchBuffer(4)
    module.gamma, module.V0, module.gamma_h = .2, .3, 1.
    module.rank_seed, module.trust_region_bound = 42, 1.5
    module.sampling_metadata = lambda: {"kernel": "fixed Gaussian unit test"}
    original_population = module.populate_buffer_adjoint_matching

    def populate():
        original_population()
        module.adjoint_buffer.bind_metadata(module.sampling_metadata())
        if not len(module.adjoint_buffer):
            module.adjoint_buffer.extend(data_0=torch.tensor([[1.], [2.], [3.]]))

    module.populate_buffer_adjoint_matching = populate
    return module


def test_snapshot_restores_models_adam_buffers_and_training_rng(tmp_path):
    module = snapshot_loop()
    module.controller(torch.ones(3, 1)).square().mean().backward()
    module.controller_optimizer.step()
    module.adjoint_buffer.bind_metadata(module.sampling_metadata())
    module.adjoint_buffer.extend(data_0=torch.randn(3, 1))
    module.corrector_buffer.bind_metadata(module.sampling_metadata())
    module.corrector_buffer.extend(data_0=torch.randn(2, 1))
    snapshots = []

    def diagnostic(payload):
        if payload["phase"] == "u" and payload["phase_epoch"] == 2:
            training_rng = torch.get_rng_state().clone()
            torch.manual_seed(8123); torch.rand(19); np.random.rand(13); random.random()
            path = tmp_path / "prefix.pt"
            save_inner_snapshot(module, path, {"parent": "trusted-unit-test"})
            snapshots.append((path, training_rng, deepcopy(module.controller.state_dict()),
                              deepcopy(module.controller_optimizer.state_dict())))

    observe_inner_boundaries(module, diagnostic, {"u": [2], "h": []})
    module.inner_loop_u()
    path, rng, weights, optimizer = snapshots[0]
    target = snapshot_loop()
    target.rank_seed, target.trust_region_bound = 999, 100.
    restore_inner_snapshot(target, path, expected_provenance={"parent": "trusted-unit-test"})
    assert target.current_epoch == 2 and target.global_step == 4 and target.phase_epoch == 2
    assert target.rank_seed == 42 and target.trust_region_bound == 1.5
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(target.adjoint_buffer) == 3 and len(target.corrector_buffer) == 2
    assert target.adjoint_buffer.metadata == module.sampling_metadata()
    for name, value in target.controller.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    restored = target.controller_optimizer.state_dict()
    for key, values in restored["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, optimizer["state"][key][name], rtol=0, atol=0)
    assert all(not p.requires_grad for p in target.prev_integrator.parameters())
    assert all(not p.requires_grad for p in target.prev_corrector.parameters())
    before = deepcopy(target.controller.state_dict())
    with pytest.raises(ValueError, match="provenance"):
        restore_inner_snapshot(target, path, expected_provenance={"parent": "incorrect"})
    for name, value in before.items():
        torch.testing.assert_close(value, target.controller.state_dict()[name], rtol=0, atol=0)
