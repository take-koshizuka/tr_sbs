import math

import pytest
import torch

from tr_asbs.utils.scaled_backward_diagnostic import power_of_two_scale, unscale_and_clip


@pytest.mark.parametrize('magnitude', [0., .2, 10., 9e30])
def test_power_scale_is_exact_binary_and_keeps_scalar_small(magnitude):
    loss = torch.tensor(magnitude)
    scale = power_of_two_scale(loss)
    assert 0 < scale <= 1 and math.frexp(scale)[0] == .5
    assert abs(float(loss * scale)) <= 1


@pytest.mark.parametrize('gradient', [[3., 4.], [3e39, -4e39]])
@pytest.mark.parametrize('bound', [1., 100.])
def test_unscale_before_clip_matches_independent_float64_vector(gradient, bound):
    reference = torch.tensor(gradient, dtype=torch.float64)
    scale = 2.**-100
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = (reference * scale).float()
    norm = unscale_and_clip([p], scale, bound)
    expected = reference * min(1., bound / (float(reference.norm()) + 1e-6))
    torch.testing.assert_close(norm, reference.norm(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(p.grad, expected.float(), rtol=1e-6, atol=1e-6)


def test_unclipped_gradient_and_adam_preserve_epsilon_and_counter():
    first = torch.nn.Parameter(torch.tensor([.2, -.3]))
    second = torch.nn.Parameter(first.detach().clone())
    a = torch.optim.Adam([first], lr=1e-4, eps=1e-8)
    b = torch.optim.Adam([second], lr=1e-4, eps=1e-8)
    g = torch.tensor([1e-10, -1e-9])
    first.grad = g * 2.**-32
    second.grad = g.clone()
    unscale_and_clip([first], 2.**-32, 100.)
    a.step(); b.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert float(a.state[first]['step']) == 1


@pytest.mark.parametrize('invalid', [float('inf'), float('nan')])
def test_nonfinite_is_still_rejected_without_mutation(invalid):
    with pytest.raises(ValueError):
        power_of_two_scale(torch.tensor(invalid))
    p = torch.nn.Parameter(torch.tensor([1.]))
    p.grad = torch.tensor([invalid])
    with pytest.raises(RuntimeError):
        unscale_and_clip([p], .5, 100.)
    assert float(p.detach()) == 1.


@pytest.mark.parametrize('warm', [False, True])
def test_independent_adam_reference_matches_original_epsilon_and_momentum(warm):
    from run_aldp_scaled_backward_gate import manual_adam
    p = torch.nn.Parameter(torch.tensor([.2, -.3]))
    optimizer = torch.optim.Adam([p], lr=1e-4, eps=1e-8)
    if warm:
        p.grad = torch.tensor([.01, -.02])
        optimizer.step()
    p.grad = torch.tensor([1e-10, -1e-9])
    reference = manual_adam({'p':p.detach().clone()}, {'p':p.grad.clone()}, optimizer.state_dict())
    optimizer.step()
    torch.testing.assert_close(p.double(), reference['p'], rtol=0, atol=2e-8)
