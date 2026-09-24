import pytest
import torch

from tr_asbs.model.mlp import FourierMLP
from tr_asbs.variational import ASBSFourierControl, DISMethod, PISMethod


class CountingQuadraticTarget:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.score_calls = 0

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * x.square().sum(dim=-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        self.score_calls += 1
        return -x


def test_asbs_fourier_control_uses_exact_gmm_asbs_backbone() -> None:
    control = ASBSFourierControl(dim=10)
    reference = FourierMLP(
        dim=10,
        num_layers=4,
        channels=256,
        clip_norm=100.0,
    )

    assert control.channels == reference.channels == 256
    assert control.clip_norm == reference.clip_norm == 100.0
    assert list(control.state_dict()) == list(reference.state_dict())
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )


def test_asbs_fourier_control_maps_reverse_step_to_asbs_time() -> None:
    control = ASBSFourierControl(dim=3, num_steps=128, channels=8)
    x = torch.randn(4, 3)

    torch.testing.assert_close(control.normalized_time(128, x), torch.zeros(4))
    torch.testing.assert_close(
        control.normalized_time(1, x),
        torch.full((4,), 127.0 / 128.0),
    )
    torch.testing.assert_close(
        control.normalized_time(torch.tensor([128.0, 96.0, 64.0, 32.0]), x),
        torch.tensor([0.0, 0.25, 0.5, 0.75]),
    )


def test_asbs_fourier_control_matches_fourier_mlp_and_ignores_score() -> None:
    torch.manual_seed(12)
    control = ASBSFourierControl(
        dim=3,
        num_steps=8,
        num_layers=4,
        channels=16,
        clip_norm=100.0,
    )
    reference = FourierMLP(
        dim=3,
        num_layers=4,
        channels=16,
        clip_norm=100.0,
    )
    reference.load_state_dict(control.state_dict())
    x = torch.randn(5, 3)

    output_a = control(torch.tensor(6.0), x, torch.randn_like(x))
    output_b = control(torch.tensor(6.0), x, 100.0 * torch.randn_like(x))
    expected = reference(torch.full((5,), 0.25), x)

    torch.testing.assert_close(output_a, output_b)
    torch.testing.assert_close(output_a, expected)


@pytest.mark.parametrize("method_name", ["pis", "dis"])
def test_scoreless_control_prevents_target_score_evaluation(method_name: str) -> None:
    target = CountingQuadraticTarget(dim=2)
    control = ASBSFourierControl(
        dim=2,
        num_steps=2,
        num_layers=2,
        channels=8,
    )
    kwargs = {
        "control": control,
        "target": target,
        "num_steps": 2,
        "schedule": torch.tensor([0.2, 0.3]),
    }
    if method_name == "pis":
        method = PISMethod(**kwargs)
        result = method.rollout(batch_size=4, noise=torch.randn(2, 4, 2))
    else:
        method = DISMethod(**kwargs, init_std=1.5)
        result = method.rollout(
            batch_size=4,
            initial_noise=torch.randn(4, 2),
            noise=torch.randn(2, 4, 2),
        )

    assert target.score_calls == 0
    loss = result.neg_elbo.mean()
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(parameter.grad is not None for parameter in control.parameters())


def test_asbs_fourier_control_rejects_score_conditioning() -> None:
    with pytest.raises(ValueError, match="score_hint_mode"):
        ASBSFourierControl(dim=2, score_hint_mode="provided")
