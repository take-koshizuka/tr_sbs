from __future__ import annotations

from pathlib import Path

import hydra
import torch

from tr_asbs.process.integrator import ExponentialIntegrator, get_timesteps
from tr_asbs.process.sde import ControlledFBSDE, VESDE
from tr_asbs.variational import ScoreConditionedControl, TRIWScoreConditionedControl


class QuadraticTarget:
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return -2.0 * x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -x.square().sum(dim=-1)


def _make_control(dim: int = 3) -> TRIWScoreConditionedControl:
    control = TRIWScoreConditionedControl(dim=dim, num_steps=128)
    control.set_potential(QuadraticTarget())
    return control


def test_time_mapping_and_pis_score_hint_endpoints() -> None:
    control = _make_control()
    x = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 3.0]], requires_grad=True)

    step_0, hint_0 = control.conditioning_inputs(0.0, x)
    step_1, hint_1 = control.conditioning_inputs(1.0, x)

    torch.testing.assert_close(step_0, torch.tensor([[128.0]]))
    torch.testing.assert_close(step_1, torch.tensor([[0.0]]))
    torch.testing.assert_close(hint_0, torch.zeros_like(x))
    torch.testing.assert_close(hint_1, -2.0 * x.detach())
    assert not hint_0.requires_grad
    assert not hint_1.requires_grad


def test_batched_time_mapping_and_output_shape() -> None:
    control = _make_control()
    x = torch.randn(4, 3)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])

    step, hint = control.conditioning_inputs(t, x)
    output = control(t, x)

    torch.testing.assert_close(step.squeeze(-1), 128.0 * (1.0 - t))
    torch.testing.assert_close(hint, t[:, None] * (-2.0 * x))
    assert output.shape == x.shape


def test_backbone_parameter_count_matches_dis_pis_and_all_parameters_train() -> None:
    control = _make_control()
    reference = ScoreConditionedControl(dim=3, num_layers=2, num_hid=64)
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )

    x = torch.randn(8, 3)
    loss = control(torch.full((8,), 0.5), x).square().mean()
    loss.backward()

    missing = [name for name, parameter in control.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(torch.isfinite(parameter.grad).all() for parameter in control.parameters())


def test_matched_configs_use_128_transitions_and_same_backbone() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "src" / "tr_asbs" / "config"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for experiment in ("gmm_mode_score_triw", "gmm_dim_score_triw"):
            cfg = hydra.compose(
                config_name="train",
                overrides=[f"experiment={experiment}"],
            )
            timesteps = hydra.utils.instantiate(cfg.integrator.timesteps)
            controller = hydra.utils.instantiate(cfg.controller)
            corrector = hydra.utils.instantiate(cfg.corrector)

            assert len(timesteps) - 1 == 128
            assert controller.num_steps == 128
            assert corrector.num_steps == 128
            assert sum(p.numel() for p in controller.parameters()) == sum(
                p.numel() for p in corrector.parameters()
            )


def test_forward_backward_and_logweight_paths_all_use_128_transitions() -> None:
    target = QuadraticTarget()
    controller = _make_control(dim=2)
    corrector = _make_control(dim=2)
    sde = ControlledFBSDE(
        base_sde=VESDE(sigma_min=0.1, sigma_max=2.0),
        u_controller=controller,
        h_controller=corrector,
    )
    integrator = ExponentialIntegrator(
        sde=sde,
        timesteps=get_timesteps(0.0, 1.0, steps=128),
    )
    initial = torch.zeros(2, 2)

    forward = integrator.run(
        initial,
        center_every_step=False,
        return_trajectory=True,
        progress_bar=False,
    )
    backward = integrator.run_backward(
        forward[-1],
        center_every_step=False,
        return_trajectory=True,
        progress_bar=False,
    )
    weighted = integrator.run_with_logweights(
        integrator=integrator,
        prior_log_prob_fn=lambda x: -0.5 * x.square().sum(dim=-1),
        target_log_prob_fn=target.log_prob,
        initial_data=initial,
        center_every_step=False,
        return_trajectory=True,
        progress_bar=False,
    )

    assert len(forward) == 129
    assert len(backward) == 129
    assert len(weighted["trajectory"]) == 129
