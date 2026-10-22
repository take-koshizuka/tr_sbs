"""Finite collision forces must keep their direction after terminal clipping."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from tr_asbs.process.terminal_cost import BaseGradEnergy


@pytest.mark.parametrize('bound', (100., 1000.))
def test_huge_finite_float32_gradient_preserves_direction_and_bound(bound):
    gradient = torch.tensor([[3e30, -4e30]], dtype=torch.float32)
    assert torch.isinf(torch.linalg.vector_norm(gradient))
    result = BaseGradEnergy(None, bound).clip(gradient)
    assert result.dtype == gradient.dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([[.6 * bound, -.8 * bound]]))
    torch.testing.assert_close(result.double().norm(dim=-1), torch.tensor([bound], dtype=torch.double))


def test_unclipped_values_and_nonfinite_failure_are_preserved():
    clip = BaseGradEnergy(None, 100.)
    gradient = torch.tensor([[3., 4.]])
    torch.testing.assert_close(clip.clip(gradient), gradient, rtol=0, atol=0)
    assert not torch.isfinite(clip.clip(torch.tensor([[float('inf'), 0.]]))).all()
    assert not torch.isfinite(clip.clip(torch.tensor([[float('nan'), 0.]]))).all()
    torch.testing.assert_close(BaseGradEnergy(None).clip(gradient), gradient, rtol=0, atol=0)


def test_diagnostic_gradient_norm_does_not_overflow():
    potential = SimpleNamespace(eval=lambda _: torch.zeros(1),
        grad_E=lambda _: torch.tensor([[3e30, 4e30]]))
    result = BaseGradEnergy(potential, 100.).potential_grad(None, return_grad_norm=True)
    torch.testing.assert_close(result['grad_norm'], torch.tensor([5e30], dtype=torch.double))


def test_sampling_metadata_pins_terminal_clip_value_and_accumulation():
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    cfg = OmegaConf.create(dict(source={}, controller={}, corrector={}, potential={},
                               terminal_cost=dict(grad_clip_val=100.)))
    fake = SimpleNamespace(cfg=cfg, fc_loss_mode='cross_entropy',
        integrator=SimpleNamespace(kernel_metadata=lambda: {}),
        terminal_cost=BaseGradEnergy(None,100.), potential=SimpleNamespace())
    first = GammaASBSModule.sampling_metadata(fake)
    assert first['terminal_gradient_clip_accumulation_dtype'] == 'float64'
    fake.cfg.terminal_cost.grad_clip_val = 1000.
    second = GammaASBSModule.sampling_metadata(fake)
    assert first != second
    assert first['terminal_cost']['grad_clip_val'] == 100.
    assert second['terminal_cost']['grad_clip_val'] == 1000.
