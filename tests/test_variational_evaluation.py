import math

import torch
import wandb

from tr_asbs.variational import RolloutResult
from tr_asbs.variational.evaluation import evaluate_variational_method


class DeterministicMethod:
    def __init__(self) -> None:
        self.chunk_sizes: list[int] = []
        self.offset = 0

    def rollout(
        self,
        batch_size: int,
        *,
        return_trajectory: bool,
    ) -> RolloutResult:
        assert return_trajectory is False
        self.chunk_sizes.append(batch_size)
        values = torch.arange(
            self.offset,
            self.offset + batch_size,
            dtype=torch.float32,
        )
        self.offset += batch_size
        samples = torch.stack((values, -values), dim=-1)
        return RolloutResult(
            samples=samples,
            running_cost=0.1 * values,
            stochastic_cost=-0.05 * values,
            terminal_cost=0.2 * values,
        )


class DetailedTarget:
    dim = 2

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * x.square().sum(dim=-1)

    def log_details(
        self,
        results: dict[str, torch.Tensor],
        *,
        visualize: bool,
    ) -> dict[str, object]:
        assert visualize is True
        assert results["data_1"].shape == (5, 2)
        assert results["log_RN[h|u]"].shape == (5,)
        return {
            "loss/custom": torch.tensor(2.5),
            "figures/custom": [wandb.Image(torch.zeros(3, 4, 4))],
            "ignored/non_scalar": torch.ones(2),
            "ignored/string": "value",
        }


class GroundTruthTarget(DetailedTarget):
    def __init__(self) -> None:
        self.sample_shape: tuple[int, ...] | None = None
        self.visualized_samples: torch.Tensor | None = None
        self.visualized_title: str | None = None

    def sample(self, sample_shape: tuple[int, ...]) -> torch.Tensor:
        self.sample_shape = sample_shape
        return torch.full((*sample_shape, self.dim), 3.0)

    def visualize(self, samples: torch.Tensor, title: str) -> dict[str, object]:
        self.visualized_samples = samples
        self.visualized_title = title
        return {
            f"figures/{title}": [wandb.Image(torch.zeros(3, 4, 4))],
        }


def test_chunked_reverse_evaluation_metrics() -> None:
    method = DeterministicMethod()
    output = evaluate_variational_method(
        method,
        DetailedTarget(),
        num_samples=5,
        chunk_size=2,
    )

    values = torch.arange(5, dtype=torch.float32)
    neg_elbo = 0.3 * values
    log_weights = -0.25 * values
    expected_log_z = torch.logsumexp(log_weights, dim=0) - math.log(5)
    expected_ess = torch.exp(
        2 * torch.logsumexp(log_weights, dim=0)
        - torch.logsumexp(2 * log_weights, dim=0)
        - math.log(5)
    )

    assert method.chunk_sizes == [2, 2, 1]
    assert output.samples.shape == (5, 2)
    torch.testing.assert_close(output.log_weights, log_weights)
    assert output.metrics["eval/neg_elbo"] == float(neg_elbo.mean())
    assert output.metrics["eval/reverse_elbo"] == -float(neg_elbo.mean())
    assert output.metrics["eval/log_z"] == float(expected_log_z)
    assert output.metrics["eval/ess"] == float(expected_ess)
    assert output.metrics["eval/raw_path_ess"] == float(expected_ess)
    assert output.metrics["loss/custom"] == 2.5
    assert len(output.media["figures/custom"]) == 1
    assert "ignored/non_scalar" not in output.metrics
    assert "ignored/string" not in output.metrics
    assert "ignored/non_scalar" not in output.media
    assert "ignored/string" not in output.media


def test_ground_truth_visualization_is_included_in_media() -> None:
    method = DeterministicMethod()
    target = GroundTruthTarget()

    output = evaluate_variational_method(
        method,
        target,
        num_samples=5,
        chunk_size=2,
        include_target_details=True,
        visualize=True,
    )

    assert target.sample_shape == (5,)
    assert target.visualized_samples is not None
    assert target.visualized_samples.shape == (5, 2)
    assert target.visualized_title == "gt_data_1"
    assert len(output.media["figures/gt_data_1"]) == 1
