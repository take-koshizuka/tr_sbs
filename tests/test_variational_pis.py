import ast
from copy import deepcopy
import inspect
import math
import textwrap

import pytest
import torch
from torch import nn

from tr_asbs.variational import (
    CosineDiffusionSchedule,
    DiffusionSchedule,
    PISMethod,
    ScoreConditionedControl,
)


class QuadraticTarget:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.score_inputs_require_grad: list[bool] = []
        self.log_prob_inputs_require_grad: list[bool] = []

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        self.log_prob_inputs_require_grad.append(x.requires_grad)
        return -0.5 * x.square().sum(dim=-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        self.score_inputs_require_grad.append(x.requires_grad)
        return -x


class ZeroControl(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        step: int,
        x: torch.Tensor,
        score_hint: torch.Tensor,
    ) -> torch.Tensor:
        del step, score_hint
        return torch.zeros_like(x) + 0.0 * self.anchor


class AffineControl(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.tensor(0.07))
        self.score_hints_require_grad: list[bool] = []

    def forward(
        self,
        step: int,
        x: torch.Tensor,
        score_hint: torch.Tensor,
    ) -> torch.Tensor:
        self.score_hints_require_grad.append(score_hint.requires_grad)
        return self.scale * (x + 0.5 * score_hint) + float(step) * 1e-3


def test_cosine_schedule_is_precomputed_and_positive() -> None:
    sigma_min = 0.01
    sigma_max = 4.0
    offset = 1.0 + 0.008
    schedule = CosineDiffusionSchedule(
        num_steps=8,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        pow=2.0,
    )

    assert schedule.values.shape == (8,)
    assert bool(torch.isfinite(schedule.values).all())
    assert bool((schedule.values > 0).all())
    torch.testing.assert_close(
        schedule.simulation_values,
        schedule.values.flip(0),
    )
    torch.testing.assert_close(
        schedule.integrated_variance(),
        schedule.values.square().mean(),
    )
    assert schedule(1) == schedule.values[0]
    assert schedule(8) == schedule.values[-1]
    assert schedule.pow == 2.0
    expected_first = (
        0.5
        * (sigma_max - sigma_min)
        * math.cos(0.5 * math.pi * (offset - 1.0 / 8.0) / offset) ** 2
        + 0.5 * sigma_min
    )
    expected_last = (
        0.5
        * (sigma_max - sigma_min)
        * math.cos(0.5 * math.pi * (offset - 1.0) / offset) ** 2
        + 0.5 * sigma_min
    )
    torch.testing.assert_close(schedule.values[0], torch.tensor(expected_first))
    torch.testing.assert_close(schedule.values[-1], torch.tensor(expected_last))


def test_fixed_noise_rollout_matches_manual_zero_control() -> None:
    target = QuadraticTarget(dim=2)
    schedule_values = torch.tensor([0.5, 1.0])
    method = PISMethod(
        control=ZeroControl(dim=2),
        target=target,
        num_steps=2,
        schedule=DiffusionSchedule(schedule_values),
    )
    noise = torch.tensor(
        [
            [[1.0, -0.25], [0.5, 0.75]],
            [[-0.5, 0.2], [0.1, -1.0]],
        ]
    )

    result = method.rollout(
        batch_size=2,
        noise=noise,
        return_trajectory=True,
    )

    dt = 0.5
    expected_samples = 1.0 * noise[0] * math.sqrt(dt) + 0.5 * noise[1] * math.sqrt(dt)
    variance = (1.0**2 + 0.5**2) * dt
    reference_log_prob = -0.5 * (
        expected_samples.square().sum(dim=-1) / variance
        + 2 * math.log(2 * math.pi * variance)
    )
    target_log_prob = -0.5 * expected_samples.square().sum(dim=-1)

    torch.testing.assert_close(result.samples, expected_samples)
    torch.testing.assert_close(result.running_cost, torch.zeros(2))
    torch.testing.assert_close(result.stochastic_cost, torch.zeros(2))
    torch.testing.assert_close(
        result.terminal_cost,
        reference_log_prob - target_log_prob,
    )
    torch.testing.assert_close(result.neg_elbo, result.terminal_cost)
    torch.testing.assert_close(result.log_weight, -result.terminal_cost)
    assert result.trajectory is not None
    assert result.trajectory.shape == (2, 2, 2)


def test_rollout_clips_noise_to_reference_bounds() -> None:
    method = PISMethod(
        control=ZeroControl(dim=2),
        target=QuadraticTarget(dim=2),
        num_steps=1,
        schedule=torch.ones(1),
        noise_clip=4.0,
    )

    result = method.rollout(
        batch_size=1,
        noise=torch.tensor([[[20.0, -20.0]]]),
    )

    torch.testing.assert_close(result.samples, torch.tensor([[4.0, -4.0]]))


def test_batched_rollout_matches_per_sample_reference() -> None:
    torch.manual_seed(4)
    target = QuadraticTarget(dim=3)
    control = AffineControl(dim=3)
    method = PISMethod(
        control=control,
        target=target,
        num_steps=4,
        schedule=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    noise = torch.randn(4, 5, 3)

    batched = method.rollout(batch_size=5, noise=noise)
    per_sample = [
        method.rollout(batch_size=1, noise=noise[:, i : i + 1]) for i in range(5)
    ]

    for field in (
        "samples",
        "running_cost",
        "stochastic_cost",
        "terminal_cost",
    ):
        expected = torch.cat([getattr(item, field) for item in per_sample])
        torch.testing.assert_close(getattr(batched, field), expected)
    assert batched.trajectory is None


def test_rollout_has_no_python_loop_over_samples() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(PISMethod.rollout)))
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    ]

    assert len(loops) == 1
    loop = loops[0]
    assert isinstance(loop, ast.For)
    assert "range(self.num_steps, 0, -1)" in ast.unparse(loop.iter)


def test_score_hint_is_detached_but_terminal_cost_is_pathwise() -> None:
    target = QuadraticTarget(dim=2)
    control = AffineControl(dim=2)
    method = PISMethod(
        control=control,
        target=target,
        num_steps=3,
        schedule=torch.tensor([0.2, 0.3, 0.4]),
    )
    noise = torch.full((3, 4, 2), 0.25)

    result = method.rollout(batch_size=4, noise=noise)
    result.neg_elbo.mean().backward()

    assert target.score_inputs_require_grad == [False, False, False]
    assert control.score_hints_require_grad == [False, False, False]
    assert target.log_prob_inputs_require_grad == [True]
    assert control.scale.grad is not None
    assert bool(torch.isfinite(control.scale.grad))


def test_score_conditioned_control_receives_finite_gradients() -> None:
    torch.manual_seed(12)
    target = QuadraticTarget(dim=3)
    control = ScoreConditionedControl(
        dim=3,
        num_layers=2,
        num_hid=8,
        weight_init=1e-8,
        bias_init=1.0,
    )
    method = PISMethod(
        control=control,
        target=target,
        num_steps=3,
        schedule=torch.tensor([0.2, 0.3, 0.4]),
    )

    result = method.rollout(
        batch_size=5,
        noise=torch.randn(3, 5, 3),
    )
    loss = result.neg_elbo.mean()
    assert bool(torch.isfinite(loss))
    loss.backward()

    for name, parameter in control.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_eager_and_compiled_control_are_numerically_equal() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    torch.manual_seed(18)
    eager = ScoreConditionedControl(dim=3, num_hid=8, bias_init=1.0)
    compiled = torch.compile(deepcopy(eager), backend="eager")
    x = torch.randn(5, 3)
    score_hint = torch.randn_like(x)

    expected = eager(torch.tensor(7.0), x, score_hint)
    actual = compiled(torch.tensor(7.0), x, score_hint)

    torch.testing.assert_close(actual, expected)


def test_zero_score_hint_mode_is_invariant_to_score_input() -> None:
    control = ScoreConditionedControl(
        dim=3,
        num_hid=8,
        bias_init=1.0,
        score_hint_mode="zero",
    )
    x = torch.randn(5, 3)
    step = torch.tensor(7.0)

    output_a = control(step, x, torch.randn_like(x))
    output_b = control(step, x, 100.0 * torch.randn_like(x))

    torch.testing.assert_close(output_a, output_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rollout_smoke() -> None:
    target = QuadraticTarget(dim=3)
    control = ScoreConditionedControl(
        dim=3,
        num_hid=8,
        bias_init=1.0,
    ).cuda()
    method = PISMethod(
        control=control,
        target=target,
        num_steps=2,
        max_diffusion=1.0,
    ).cuda()

    result = method.rollout(batch_size=4)
    loss = result.neg_elbo.mean()
    loss.backward()

    assert result.samples.is_cuda
    assert bool(torch.isfinite(loss))


@pytest.mark.parametrize("bad_batch_size", [0, -1])
def test_rollout_rejects_nonpositive_batch_size(bad_batch_size: int) -> None:
    method = PISMethod(
        control=ZeroControl(dim=1),
        target=QuadraticTarget(dim=1),
        num_steps=1,
        schedule=torch.ones(1),
    )

    with pytest.raises(ValueError, match="batch_size"):
        method.rollout(batch_size=bad_batch_size)
