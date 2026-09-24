import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from tr_asbs.potential.gmm import GMM


def _autograd_score(energy, x: torch.Tensor) -> torch.Tensor:
    x_for_grad = x.detach().clone().requires_grad_(True)
    (score,) = torch.autograd.grad(
        energy.log_prob(x_for_grad).sum(),
        x_for_grad,
    )
    return score


def _make_deterministic_gmm() -> GMM:
    energy = GMM(
        dim=2,
        num_components=3,
        device="cpu",
        dtype=torch.float64,
        uniform_mixing_weights=True,
    )
    energy.locs = torch.tensor(
        [[-1.2, 0.4], [0.7, -0.9], [1.6, 1.3]],
        dtype=torch.float64,
    )
    energy.scale_trils = torch.tensor(
        [
            [[0.6, 0.0], [0.15, 1.1]],
            [[1.3, 0.0], [-0.25, 0.7]],
            [[0.8, 0.0], [0.3, 1.4]],
        ],
        dtype=torch.float64,
    )
    energy.cat_probs = torch.tensor([0.15, 0.55, 0.30], dtype=torch.float64)
    energy.distribution = MixtureSameFamily(
        Categorical(probs=energy.cat_probs),
        MultivariateNormal(
            energy.locs,
            scale_tril=energy.scale_trils,
            validate_args=False,
        ),
        validate_args=False,
    )
    return energy


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.1, -0.3], dtype=torch.float64),
        torch.tensor(
            [[0.1, -0.3], [-1.4, 0.8], [2.0, 1.1], [0.5, -1.7]],
            dtype=torch.float64,
        ),
    ],
)
def test_gmm_analytic_score_matches_autograd(x: torch.Tensor) -> None:
    energy = _make_deterministic_gmm()

    actual = energy.score(x)
    expected = _autograd_score(energy, x)

    assert actual.shape == x.shape
    assert actual.device == x.device
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
