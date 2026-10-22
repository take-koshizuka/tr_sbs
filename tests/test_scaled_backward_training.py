"""Shared training-step semantics on tiny CPU models; no target simulation."""
from types import SimpleNamespace

import pytest
import torch
from lightning.fabric import Fabric

from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule


def module(mode='loss_scaled_fp64_clip', factor=1e25):
    fabric = Fabric(accelerator='cpu', devices=1, precision='32-true')
    fabric.launch()
    controller = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        controller.weight.fill_(1e-10)
    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4, eps=1e-8)
    controller, optimizer = fabric.setup(controller, optimizer)
    corrector = torch.nn.Linear(2, 1, bias=False)
    other = torch.optim.Adam(corrector.parameters(), lr=1e-4, eps=1e-8)
    corrector, other = fabric.setup(corrector, other)
    result = GammaASBSModule.__new__(GammaASBSModule)
    result.fabric, result.controller, result.controller_optimizer = fabric, controller, optimizer
    result.corrector, result.corrector_optimizer = corrector, other
    result.fc_loss_mode, result.fc_backward_mode = 'log_variance', mode
    result.fc_backward_statistics = dict(attempts=0, updates=0, loss_nonfinite=0,
        gradient_nonfinite=0, minimum_scale=1., maximum_scale=0.)
    result.logvar_amp_dtype = None
    result.cfg = SimpleNamespace(gradient_clip_val=100., ema_decay=None, log_loss_interval_step=1000)
    result.global_step = 1
    result.adjoint_matching_log_variance_step = lambda batch: dict(loss=(result.controller(batch)*factor).square().mean())
    result.corrector_matching_step = lambda batch: dict(loss=result.corrector(batch).square().mean())
    return result


def test_shared_step_handles_finite_loss_with_native_backward_overflow():
    batch = torch.ones(2, 2)
    m = module()
    initial = next(m.controller.parameters()).detach().clone()
    reference = initial.double().requires_grad_()
    (torch.nn.functional.linear(batch.double(), reference)*1e25).square().mean().backward()
    norm = reference.grad.norm()
    reference_grad = (reference.grad*min(1., 100./float(norm+1e-6))).float()
    expected = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([expected], lr=1e-4, eps=1e-8)
    expected.grad = reference_grad
    opt.step()
    assert m.training_step(batch, 'adjoint_matching')
    torch.testing.assert_close(next(m.controller.parameters()), expected, rtol=0, atol=1e-8)
    assert m.fc_backward_statistics['updates'] == 1
    assert m.fc_backward_statistics['minimum_scale'] < 1
    assert {float(s['step']) for s in m.controller_optimizer.state.values()} == {1.}
    native = module('native')
    assert native.training_step(batch, 'adjoint_matching') is False
    assert not native.controller_optimizer.state


def test_scaling_does_not_apply_to_h_or_change_its_adam_update():
    scaled, native = module(), module('native')
    native.corrector.load_state_dict(scaled.corrector.state_dict())
    batch = torch.ones(2, 2)
    assert scaled.training_step(batch, 'corrector_matching')
    assert native.training_step(batch, 'corrector_matching')
    torch.testing.assert_close(next(scaled.corrector.parameters()), next(native.corrector.parameters()), rtol=0, atol=0)
    assert scaled.fc_backward_statistics['attempts'] == 0


@pytest.mark.parametrize('factor', [float('inf'), float('nan')])
def test_nonfinite_loss_still_rejects_without_optimizer_mutation(factor):
    m = module(factor=factor)
    initial = next(m.controller.parameters()).detach().clone()
    assert m.training_step(torch.ones(2, 2), 'adjoint_matching') is False
    torch.testing.assert_close(next(m.controller.parameters()), initial, rtol=0, atol=0)
    assert not m.controller_optimizer.state
    assert m.fc_backward_statistics['loss_nonfinite'] == 1


def test_true_nonfinite_scaled_gradient_still_rejects():
    m = module(factor=1.)
    handle = next(m.controller.parameters()).register_hook(lambda g: torch.full_like(g, float('inf')))
    assert m.training_step(torch.ones(2, 2), 'adjoint_matching') is False
    assert not m.controller_optimizer.state
    assert m.fc_backward_statistics['gradient_nonfinite'] == 1
    assert all(p.grad is None for p in m.controller.parameters())
    handle.remove()


def test_autocast_or_multiple_processes_cannot_use_unverified_clipping():
    m = module()
    m.logvar_amp_dtype = torch.bfloat16
    with pytest.raises(RuntimeError, match='without autocast'):
        m.training_step(torch.ones(2, 2), 'adjoint_matching')
