from __future__ import annotations

from pathlib import Path

import hydra
import torch
from torch import nn

from tr_asbs.process.integrator import ExponentialIntegrator, get_timesteps
from tr_asbs.process.sde import ControlledFBSDE, PISCosineSDE, VESDE
from tr_asbs.variational.schedules import cosine_diffusion_schedule


class ConstantControl(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, time: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def _integrator(base_sde: PISCosineSDE | VESDE, steps: int):
    sde = ControlledFBSDE(base_sde, ConstantControl(0.3), ConstantControl(-0.2))
    return ExponentialIntegrator(sde, get_timesteps(0.0, 1.0, steps=steps))


def test_pis_cosine_schedule_and_exact_integrated_variance():
    steps = 128
    sde = PISCosineSDE(sigma_min=0.1, sigma_max=16.0, num_steps=steps)
    grid = get_timesteps(0.0, 1.0, steps=steps)
    pis_sigmas = cosine_diffusion_schedule(
        steps, sigma_min=0.1, sigma_max=16.0, dtype=torch.float64
    ).flip(0)

    torch.testing.assert_close(sde.diffusion(grid[:-1]).double(), pis_sigmas)
    variance = sde.diffsquare_integral(grid)
    torch.testing.assert_close(variance[0], torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(
        variance[1:] - variance[:-1], pis_sigmas.square() / steps
    )
    torch.testing.assert_close(sde.total_variance, pis_sigmas.square().sum() / steps)

    # Bridge sampling and conditional scores query V(t) away from grid points.
    midway = torch.tensor([16.5, 48.5, 80.5, 112.5], dtype=torch.float64) / steps
    midpoint_variance = sde.diffsquare_integral(midway)
    left = sde.diffsquare_integral(midway - 0.25 / steps)
    right = sde.diffsquare_integral(midway + 0.25 / steps)
    torch.testing.assert_close(midpoint_variance, (left + right) / 2)


def test_cosine_forward_backward_and_iw_use_same_interval_variance():
    steps = 4
    base_sde = PISCosineSDE(sigma_min=0.1, sigma_max=3.0, num_steps=steps)
    integrator = _integrator(base_sde, steps)
    grid = integrator.timesteps
    x = torch.tensor([[0.5, -0.4], [-0.2, 0.1]], dtype=torch.float64)
    for i in range(steps):
        t, t_next = grid[i], grid[i + 1]
        dt = t_next - t
        variance = base_sde.diffsquare_integral(t_next) - base_sde.diffsquare_integral(t)
        expected_forward = x + variance * 0.3
        expected_backward = x + variance * -0.2
        forward, _ = integrator.step(x, t, dt)
        backward, _ = integrator.step_backward(x, t_next, dt)
        torch.testing.assert_close(forward, expected_forward)
        torch.testing.assert_close(backward, expected_backward)

    def log_prior(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * z.square().sum(dim=-1)

    def log_target(z: torch.Tensor) -> torch.Tensor:
        return -(z - 1.0).square().sum(dim=-1)

    torch.manual_seed(4)
    result = integrator.run_with_logweights(
        integrator=integrator,
        prior_log_prob_fn=log_prior,
        target_log_prob_fn=log_target,
        initial_data=x,
        center_every_step=False,
        return_trajectory=True,
        progress_bar=False,
    )
    trajectory = result["trajectory"]
    assert len(trajectory) == steps + 1
    expected_log_weight = log_target(trajectory[-1]) - log_prior(trajectory[0])
    for i in range(steps):
        variance = base_sde.diffsquare_integral(grid[i + 1]) - base_sde.diffsquare_integral(grid[i])
        forward_mean = trajectory[i] + variance * 0.3
        backward_mean = trajectory[i + 1] + variance * -0.2
        fwd_error = (trajectory[i + 1] - forward_mean).square().sum(dim=-1)
        bwd_error = (trajectory[i] - backward_mean).square().sum(dim=-1)
        expected_log_weight += (fwd_error - bwd_error) / (2 * variance)
    torch.testing.assert_close(
        result["log_Ph_Pref"].double(), expected_log_weight, atol=2e-5, rtol=2e-5
    )

    restored = PISCosineSDE(sigma_min=0.1, sigma_max=3.0, num_steps=steps)
    restored.load_state_dict(base_sde.state_dict())
    torch.testing.assert_close(restored.total_variance, base_sde.total_variance)


def test_ve_backward_drift_keeps_existing_right_endpoint_convention():
    sde = VESDE(sigma_min=0.1, sigma_max=2.0)
    integrator = _integrator(sde, 4)
    x = torch.zeros(2, 3)
    time = torch.tensor(0.5)
    dt = torch.tensor(0.25)
    deterministic, _ = integrator.step_backward(x, time, dt)
    torch.testing.assert_close(deterministic, x - 0.2 * sde.diffusion(time).square() * dt)
    torch.testing.assert_close(
        integrator.sde.bwd_drift(time, x),
        integrator.sde.bwd_drift(time, x, interval_dt=dt),
        rtol=0,
        atol=0,
    )


def test_joint_config_selects_cosine_without_changing_ve_configs():
    config_dir = Path(__file__).resolve().parents[1] / "src" / "tr_asbs" / "config"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        dim_joint = hydra.compose("train", overrides=["experiment=gmm_joint_dim_score_triw_cosine"])
        mode_joint = hydra.compose("train", overrides=["experiment=gmm_joint_mode_score_triw_cosine"])
        legacy = hydra.compose("train", overrides=["experiment=gmm_dim"])
    for joint, loc_scaling, source_scale in (
        (dim_joint, 7, 5.0),
        (mode_joint, 40, 30.0),
    ):
        assert joint.potential.dim == 100
        assert joint.potential.num_components == 40
        assert joint.potential.loc_scaling == loc_scaling
        assert joint.source.scale == source_scale
        assert joint.base_sde._target_ == "tr_asbs.process.sde.PISCosineSDE"
        assert joint.base_sde.num_steps == 128
        assert len(hydra.utils.instantiate(joint.integrator.timesteps)) - 1 == 128
    assert legacy.base_sde._target_ == "tr_asbs.process.sde.VESDE"
    assert len(hydra.utils.instantiate(legacy.integrator.timesteps)) - 1 == 200
