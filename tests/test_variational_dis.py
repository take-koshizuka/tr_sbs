import ast
from copy import deepcopy
import inspect
import math
import textwrap

import pytest
import torch
from torch import nn

from tr_asbs.variational import DISMethod, ScoreConditionedControl


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
        step: torch.Tensor,
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
        step: torch.Tensor,
        x: torch.Tensor,
        score_hint: torch.Tensor,
    ) -> torch.Tensor:
        self.score_hints_require_grad.append(score_hint.requires_grad)
        return self.scale * (x + 0.5 * score_hint) + step * 1e-3


def test_fixed_noise_rollout_matches_manual_zero_control() -> None:
    target = QuadraticTarget(dim=2)
    method = DISMethod(
        control=ZeroControl(dim=2),
        target=target,
        num_steps=2,
        init_std=2.0,
        schedule=torch.tensor([0.5, 1.0]),
    )
    initial_noise = torch.tensor([[0.25, -0.5], [0.75, 0.1]])
    noise = torch.tensor(
        [
            [[1.0, -0.25], [0.5, 0.75]],
            [[-0.5, 0.2], [0.1, -1.0]],
        ]
    )

    result = method.rollout(
        batch_size=2,
        noise=noise,
        initial_noise=initial_noise,
        return_trajectory=True,
    )

    dt = 0.5
    sqrt_dt = math.sqrt(dt)
    initial_x = 2.0 * initial_noise
    first_sigma = math.sqrt(2.0 * 1.0) * 2.0
    first_x = initial_x + 1.0 * initial_x * dt + first_sigma * noise[0] * sqrt_dt
    second_sigma = math.sqrt(2.0 * 0.5) * 2.0
    expected_samples = first_x + 0.5 * first_x * dt + second_sigma * noise[1] * sqrt_dt
    initial_log_prob = -0.5 * (
        initial_x.square().sum(dim=-1) / 4.0 + 2 * math.log(2.0 * math.pi * 4.0)
    )
    target_log_prob = -0.5 * expected_samples.square().sum(dim=-1)

    torch.testing.assert_close(result.samples, expected_samples)
    torch.testing.assert_close(
        result.running_cost,
        torch.full((2,), -2.0 * (1.0 + 0.5) * dt),
    )
    torch.testing.assert_close(result.stochastic_cost, torch.zeros(2))
    torch.testing.assert_close(
        result.terminal_cost,
        initial_log_prob - target_log_prob,
    )
    torch.testing.assert_close(
        result.neg_elbo,
        result.running_cost + result.terminal_cost,
    )
    torch.testing.assert_close(
        result.log_weight,
        -(result.running_cost + result.terminal_cost),
    )
    assert result.trajectory is not None
    torch.testing.assert_close(result.trajectory[0], first_x)
    torch.testing.assert_close(result.trajectory[1], expected_samples)


def test_rollout_clips_initial_and_integration_noise() -> None:
    method = DISMethod(
        control=ZeroControl(dim=2),
        target=QuadraticTarget(dim=2),
        num_steps=1,
        init_std=2.0,
        schedule=torch.ones(1),
        noise_clip=4.0,
    )

    result = method.rollout(
        batch_size=1,
        initial_noise=torch.tensor([[20.0, -20.0]]),
        noise=torch.tensor([[[-20.0, 20.0]]]),
    )

    clipped_initial = torch.tensor([[8.0, -8.0]])
    expected = (
        clipped_initial
        + clipped_initial
        + math.sqrt(2.0) * 2.0 * torch.tensor([[-4.0, 4.0]])
    )
    torch.testing.assert_close(result.samples, expected)


def test_batched_rollout_matches_per_sample_reference() -> None:
    torch.manual_seed(4)
    target = QuadraticTarget(dim=3)
    control = AffineControl(dim=3)
    method = DISMethod(
        control=control,
        target=target,
        num_steps=4,
        init_std=1.5,
        schedule=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    initial_noise = torch.randn(5, 3)
    noise = torch.randn(4, 5, 3)

    batched = method.rollout(
        batch_size=5,
        noise=noise,
        initial_noise=initial_noise,
    )
    per_sample = [
        method.rollout(
            batch_size=1,
            noise=noise[:, i : i + 1],
            initial_noise=initial_noise[i : i + 1],
        )
        for i in range(5)
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
    tree = ast.parse(textwrap.dedent(inspect.getsource(DISMethod.rollout)))
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
    method = DISMethod(
        control=control,
        target=target,
        num_steps=3,
        init_std=1.5,
        schedule=torch.tensor([0.2, 0.3, 0.4]),
    )
    initial_noise = torch.full((4, 2), 0.1)
    noise = torch.full((3, 4, 2), 0.25)

    result = method.rollout(
        batch_size=4,
        noise=noise,
        initial_noise=initial_noise,
    )
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
    method = DISMethod(
        control=control,
        target=target,
        num_steps=3,
        init_std=1.5,
        schedule=torch.tensor([0.2, 0.3, 0.4]),
    )

    result = method.rollout(
        batch_size=5,
        initial_noise=torch.randn(5, 3),
        noise=torch.randn(3, 5, 3),
    )
    loss = result.neg_elbo.mean()
    assert bool(torch.isfinite(loss))
    loss.backward()

    for name, parameter in control.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_generator_reproduces_initial_and_path_noise() -> None:
    method = DISMethod(
        control=ZeroControl(dim=2),
        target=QuadraticTarget(dim=2),
        num_steps=2,
        init_std=1.2,
        schedule=torch.tensor([0.2, 0.3]),
    )

    first = method.rollout(
        batch_size=3,
        generator=torch.Generator().manual_seed(123),
    )
    second = method.rollout(
        batch_size=3,
        generator=torch.Generator().manual_seed(123),
    )

    torch.testing.assert_close(first.samples, second.samples)
    torch.testing.assert_close(first.neg_elbo, second.neg_elbo)


@pytest.mark.parametrize("bad_init_std", [0.0, -1.0, float("inf")])
def test_rejects_invalid_initial_standard_deviation(bad_init_std: float) -> None:
    with pytest.raises(ValueError, match="init_std"):
        DISMethod(
            control=ZeroControl(dim=1),
            target=QuadraticTarget(dim=1),
            num_steps=1,
            init_std=bad_init_std,
            schedule=torch.ones(1),
        )


def test_eager_and_compiled_control_are_numerically_equal_for_dis_hint() -> None:
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
