"""Analytic, density, and gradient contracts of the modular TR-SBS pipeline."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tr_asbs.process.sde import ControlledFBSDE, VESDE, PISCosineSDE
from tr_asbs.process.noise_schedule import AnalyticNoiseSchedule, LegacyNoiseSchedule
from tr_asbs.process.reference import BrownianReference, LinearGaussianReference
from tr_asbs.process.integrator import (
    EulerMaruyamaIntegrator,
    ExponentialIntegrator,
    DiffusionTimeEulerIntegrator,
    LinearExponentialIntegrator,
    get_timesteps,
)
from tr_asbs.process.transition import GaussianTransition
from tr_asbs.process.source import GaussianPrior
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule


class Control(nn.Module):
    def __init__(self, value=0.3):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.double))

    def forward(self, t, x):
        return self.value * x.new_ones(x.shape)


def integrator(base, cls, **kwargs):
    return cls(ControlledFBSDE(base, Control(), Control(-0.2)), steps=4, **kwargs)


@pytest.mark.parametrize("kind", ["constant", "linear", "geometric"])
def test_noise_integral_inverse_and_normalization(kind):
    noise = AnalyticNoiseSchedule(kind, 0.1, 2.0, total_variance=3.0)
    t = torch.tensor([0.1, 0.4, 0.9], dtype=torch.double, requires_grad=True)
    derivative = torch.autograd.grad(noise.variance(t).sum(), t)[0]
    torch.testing.assert_close(derivative, noise.diffusion(t).square())
    torch.testing.assert_close(
        noise.variance(1.0), torch.tensor(3.0, dtype=torch.double)
    )
    torch.testing.assert_close(noise.inverse_variance(noise.variance(t)), t)


def test_grid_counts_and_tau_keep_noise_knots_fixed():
    for shape in [None, "quad", "cosine"]:
        for dt in [0.13, None]:
            grid = get_timesteps(
                0.0, 1.0, dt=dt, steps=7 if dt is None else None, rescale_t=shape
            )
            assert len(grid) == (8 if dt is None else 9)
            assert grid[0] == 0 and grid[-1] == 1 and (grid.diff() > 0).all()
    ref = BrownianReference(LegacyNoiseSchedule(PISCosineSDE(num_steps=16)))
    solver = integrator(ref, DiffusionTimeEulerIntegrator, grid="uniform_tau")
    var = ref.diffsquare_integral(solver.timesteps.double()).diff()
    torch.testing.assert_close(var, var.mean().expand_as(var), rtol=2e-6, atol=2e-6)
    assert ref.noise_schedule.process.num_steps == 16


def test_constant_noise_matches_all_zero_drift_solvers():
    x = torch.ones(3, 2, dtype=torch.double)
    kernels = [
        integrator(BrownianReference(AnalyticNoiseSchedule(g_max=2.0)), cls).transition(
            x, torch.tensor(0.2), torch.tensor(0.1)
        )
        for cls in [
            EulerMaruyamaIntegrator,
            ExponentialIntegrator,
            DiffusionTimeEulerIntegrator,
        ]
    ]
    for kernel in kernels[1:]:
        torch.testing.assert_close(kernel.mean, kernels[0].mean)
        torch.testing.assert_close(kernel.variance, kernels[0].variance)
        torch.testing.assert_close(kernel.sample(torch.zeros_like(x)), kernel.mean)


def test_ve_solver_variances_and_density_are_not_shared():
    x = torch.randn(4, 2, dtype=torch.double)
    em = integrator(VESDE(0.1, 2.0), EulerMaruyamaIntegrator)
    exact = integrator(VESDE(0.1, 2.0), ExponentialIntegrator)
    t, dt = torch.tensor(0.25), torch.tensor(0.25)
    p, q = em.transition(x, t, dt), exact.transition(x, t, dt)
    assert not torch.isclose(p.variance, q.variance)
    y = p.sample(torch.ones_like(x))
    normal = torch.distributions.Normal(q.mean, q.variance.sqrt())
    torch.testing.assert_close(q.log_prob(y), normal.log_prob(y).sum(-1))
    kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(p.mean, p.variance.sqrt()), normal
    ).sum(-1)
    torch.testing.assert_close(p.kl(q), kl)


@pytest.mark.parametrize("kind", ["ou", "vp"])
def test_linear_reference_forward_reverse_and_bridge(kind):
    ref = LinearGaussianReference(
        kind=kind,
        rate=0.7,
        diffusion_scale=1.2,
        beta_min=0.2,
        beta_max=1.7,
        stationary_scale=1.3,
    )
    s, t = torch.tensor(0.2, dtype=torch.double), torch.tensor(0.7, dtype=torch.double)
    phi, k, q = ref.transition_coefficients(s, t)
    bp, bk, bq = ref.transition_coefficients(s, t, backward=True)
    torch.testing.assert_close(bp, 1 / phi)
    torch.testing.assert_close(bq, q / phi.square())
    if kind == "vp":
        torch.testing.assert_close(phi.square() * 1.3**2 + q, q.new_tensor(1.3**2))
    a = ref.adjoint_transport(s)
    torch.testing.assert_close(a, torch.exp(-ref.integrated_rate(s, 1)))
    torch.manual_seed(2)
    times = torch.full((2048,), 0.4, dtype=torch.double)
    x0, x1 = (
        torch.ones(2048, 1, dtype=torch.double),
        torch.zeros(2048, 1, dtype=torch.double),
    )
    bridge = ref.sample_posterior(times, x0, x1)
    p0, _, q0 = ref.transition_coefficients(0, times[:1])
    p1, _, q1 = ref.transition_coefficients(times[:1], 1)
    gain = q0 * p1 / (p1.square() * q0 + q1)
    expected_mean = p0 - gain * p1 * p0
    expected_var = q0 - gain.square() * (p1.square() * q0 + q1)
    assert abs(bridge.mean() - expected_mean) < 0.04
    assert abs(bridge.var() - expected_var) < 0.04
    score = ref.cond_score_t0(times, x0, bridge)
    torch.testing.assert_close(score, (p0 * x0 - bridge) / q0)
    solver = integrator(ref, LinearExponentialIntegrator, control_evaluation="midpoint")
    assert torch.isfinite(
        solver.transition(x0[:3], s.float(), (t - s).float()).log_prob(x0[:3])
    ).all()


def test_zero_rate_limit_and_projected_covariance_rank():
    ref = LinearGaussianReference(rate=0, diffusion_scale=2.0)
    phi, k, q = ref.transition_coefficients(0.0, 0.2)
    assert phi == 1
    torch.testing.assert_close(k, torch.tensor(0.8, dtype=torch.double))
    torch.testing.assert_close(k, q)
    mean = torch.zeros(3, 4, 2, dtype=torch.double)
    kernel = GaussianTransition(mean, torch.tensor(0.7), mean_free=True)
    noise = kernel.sample()
    torch.testing.assert_close(
        noise.mean(-2), torch.zeros(3, 2, dtype=torch.double), atol=1e-15, rtol=0
    )
    expected = -0.5 * (
        noise.square().sum((-1, -2)) / 0.7
        + 6 * torch.log(mean.new_tensor(2 * torch.pi * 0.7))
    )
    torch.testing.assert_close(kernel.log_prob(noise), expected)
    assert torch.isneginf(kernel.log_prob(noise + 1.0)).all()


def test_gamma_zero_path_kl_has_zero_value_and_gradient():
    model = GammaASBSModule.__new__(GammaASBSModule)
    model.integrator = integrator(
        BrownianReference(AnalyticNoiseSchedule()), DiffusionTimeEulerIntegrator
    )
    model.prev_integrator = deepcopy(model.integrator)
    model.source = GaussianPrior(2, 1.0)
    model.fabric = SimpleNamespace(to_device=lambda x: x)
    model.potential = SimpleNamespace(log_prob=lambda x: -0.5 * x.square().sum(-1))
    model.gamma = 0.0
    loss = model.forward_path_kl_step(
        {"data_0": torch.randn(4, 2, dtype=torch.double)}
    )["loss"]
    assert loss.item() == 0.0
    loss.backward()
    assert model.integrator.sde.u_controller.value.grad.item() == 0.0
    assert model.prev_integrator.sde.u_controller.value.grad is None


def test_invalid_density_and_solver_combinations_fail():
    ref = LinearGaussianReference()
    with pytest.raises(ValueError, match="zero-drift"):
        integrator(ref, ExponentialIntegrator).transition(
            torch.zeros(2, 2), torch.tensor(0.0), torch.tensor(0.1)
        )
    solver = integrator(VESDE(), ExponentialIntegrator)
    with pytest.raises(ValueError, match="Deterministic"):
        solver.run_with_logweights(
            solver,
            lambda x: x[:, 0] * 0,
            lambda x: x[:, 0] * 0,
            torch.zeros(2, 2),
            center_every_step=False,
            zero_last_step_noise=True,
        )


def test_true_stationary_reverse_conditional_differs_from_absolute_h_zero():
    ref = LinearGaussianReference(rate=0.5, diffusion_scale=1.0)
    y = torch.ones(2, 1, dtype=torch.double)
    s, t = y.new_tensor(0.2), y.new_tensor(0.6)
    phi, _, q = ref.transition_coefficients(s, t)
    true_reverse = ref.reverse_conditional(y, s, t, initial_variance=1.0)
    torch.testing.assert_close(true_reverse.mean, phi * y)
    torch.testing.assert_close(true_reverse.variance, q)
    solver = LinearExponentialIntegrator(
        ControlledFBSDE(ref, Control(0.0), Control(0.0)), steps=1
    )
    learned_reverse = solver.transition(y, t, t - s, backward=True)
    assert not torch.allclose(learned_reverse.mean, true_reverse.mean)
    assert not torch.allclose(learned_reverse.variance, true_reverse.variance)


@pytest.mark.parametrize("reference", ["brownian", "ou"])
def test_general_gamma_path_kl_gradient_matches_gaussian_oracle(reference):
    # An antithetic noise pair integrates this quadratic one-step loss exactly.
    from unittest.mock import patch

    ref = (
        BrownianReference(AnalyticNoiseSchedule())
        if reference == "brownian"
        else LinearGaussianReference(rate=0.7, diffusion_scale=1.0)
    )
    cls = (
        DiffusionTimeEulerIntegrator
        if reference == "brownian"
        else LinearExponentialIntegrator
    )
    current = integrator(ref, cls)
    current.timesteps = torch.tensor([0.0, 1.0])
    old = deepcopy(current)
    old.sde.u_controller.value.data.fill_(0.4)
    old.sde.h_controller.value.data.zero_()
    module = GammaASBSModule.__new__(GammaASBSModule)
    module.integrator, module.prev_integrator = current, old
    module.source = GaussianPrior(1, 1.0)
    module.fabric = SimpleNamespace(to_device=lambda x: x)
    module.potential = SimpleNamespace(
        log_prob=lambda x: -0.5 * ((x[:, 0] - 1.2) / 0.8).square()
    )
    module.gamma = 0.35
    if reference == "brownian":
        k, q, backward_phi, backward_q = 1.0, 1.0, 1.0, 1.0
    else:
        _, k, q = ref.transition_coefficients(0, 1)
        backward_phi, _, backward_q = ref.transition_coefficients(0, 1, True)
    theta = current.sde.u_controller.value
    analytic_gradient = (
        module.gamma
        * (
            k * (k * theta - 1.2) / 0.8**2
            + k * backward_phi**2 * k * theta / backward_q
        )
        + (1 - module.gamma) * k**2 * (theta - 0.4) / q
    )
    with patch("torch.randn_like", lambda x: x.new_tensor([[-1.0], [1.0]])):
        loss = module.forward_path_kl_step(
            {"data_0": torch.zeros(2, 1, dtype=torch.double)}
        )["loss"]
    gradient = torch.autograd.grad(loss, theta)[0]
    torch.testing.assert_close(gradient, analytic_gradient)


def test_buffer_metadata_rejects_old_or_changed_kernel():
    from tr_asbs.data.buffer import BatchBuffer

    buffer = BatchBuffer(8)
    buffer.bind_metadata({"grid": [0.0, 1.0]})
    buffer.extend(data_0=torch.zeros(2, 1))
    saved = buffer.state_dict()
    restored = BatchBuffer(8)
    restored.load_state_dict(saved)
    with pytest.raises(ValueError, match="differs"):
        restored.bind_metadata({"grid": [0.0, 0.5, 1.0]})
    restored.clear()
    restored.bind_metadata({"grid": [0.0, 0.5, 1.0]})


def test_graph_kernel_matches_particle_subspace_and_variable_batch_sizes():
    pytest.importorskip("e3nn", reason="Requires the chem extra: uv sync --extra chem")
    from mace.tools.torch_geometric import Batch
    from mace.tools.torch_geometric.data import Data
    from tr_asbs.process.integrator import AtomicExponentialIntegrator

    class GraphControl(nn.Module):
        def forward(self, t, data):
            return data.pos * 0.1

    data = Batch.from_data_list(
        [Data(pos=torch.zeros(3, 3)), Data(pos=torch.zeros(4, 3))]
    )
    ref = VESDE(0.1, 1.0, is_mean_free=True)
    solver = AtomicExponentialIntegrator(
        ControlledFBSDE(ref, GraphControl(), GraphControl()), steps=2
    )
    kernel = solver.transition(data, torch.tensor(0.0), torch.tensor(0.5))
    noise = kernel.sample()
    assert noise.pos.shape == (7, 3)
    assert kernel.log_prob(noise).shape == (2,)
    for graph in noise.to_data_list():
        torch.testing.assert_close(graph.pos.mean(0), torch.zeros(3), atol=1e-7, rtol=0)
    result = solver.run_with_logweights(
        solver,
        lambda x: x.pos.new_zeros(x.num_graphs),
        lambda x: x.pos.new_zeros(x.num_graphs),
        data,
    )
    assert torch.isfinite(result["log_Ph_Pref"]).all()
    torch.testing.assert_close(result["log_Pu_Pref"], torch.zeros(2))


@pytest.mark.parametrize(
    "cls",
    [
        EulerMaruyamaIntegrator,
        DiffusionTimeEulerIntegrator,
        LinearExponentialIntegrator,
    ],
)
def test_vectorized_logvar_value_and_gradient_match_independent_gaussian_densities(cls):
    base = (
        LinearGaussianReference(rate=0.7)
        if cls is LinearExponentialIntegrator
        else BrownianReference(
            AnalyticNoiseSchedule("geometric", 0.1, 2.0, total_variance=1.0)
        )
    )
    new = integrator(base, cls, grid="uniform_tau")
    old = deepcopy(new)
    old.sde.u_controller.value.data.fill_(0.4)
    trajectory = torch.stack(
        old.run(
            torch.randn(4, 2, dtype=torch.double),
            center_every_step=False,
            return_trajectory=True,
            progress_bar=False,
        ),
        dim=1,
    )
    raw = torch.linspace(-2, 1, 4, dtype=torch.double)
    module = GammaASBSModule.__new__(GammaASBSModule)
    module.integrator, module.prev_integrator = new, old
    module.fabric = SimpleNamespace(to_device=lambda x: x)
    module.gamma, module.logvar_amp_dtype = 0.3, None
    actual = module.adjoint_matching_log_variance_step(
        {"data_traj": trajectory, "log_Ph_Pref_raw": raw}
    )["loss"]
    expected_ratio = torch.zeros_like(raw)
    for i, (t, tn) in enumerate(zip(new.timesteps[:-1], new.timesteps[1:])):
        p = new.transition(trajectory[:, i], t, tn - t)
        with torch.no_grad():
            q = old.transition(trajectory[:, i], t, tn - t)
        pn = torch.distributions.Normal(p.mean, p.variance.sqrt())
        qn = torch.distributions.Normal(q.mean, q.variance.sqrt())
        expected_ratio = expected_ratio + (
            pn.log_prob(trajectory[:, i + 1]) - qn.log_prob(trajectory[:, i + 1])
        ).sum(-1)
    expected = (expected_ratio - 0.3 * raw).var(unbiased=True)
    torch.testing.assert_close(actual, expected)
    theta = new.sde.u_controller.value
    torch.testing.assert_close(
        torch.autograd.grad(actual, theta)[0], torch.autograd.grad(expected, theta)[0]
    )


def test_resuming_fc_epoch_preserves_frozen_policy_and_skips_completed_epochs():
    from omegaconf import OmegaConf

    module = GammaASBSModule.__new__(GammaASBSModule)
    module.cfg = OmegaConf.create(
        dict(
            adjoint_matching_epochs=2,
            progress_bar=False,
            log_detail_interval_epoch=100,
            monitor_values=[],
        )
    )
    module.integrator = integrator(VESDE(), ExponentialIntegrator)
    module.prev_integrator = deepcopy(module.integrator)
    module.prev_integrator.sde.u_controller.value.data.fill_(0.4)
    module.phase, module.phase_epoch, module.current_epoch = "u", 1, 5
    module.clear_adjoint_buffer_per_outer_loop = True
    module.global_step, module.total_epochs = 0, 4
    module.fabric = SimpleNamespace(global_rank=0)
    module.potential = SimpleNamespace()
    populated = []
    module.populate_buffer_adjoint_matching = lambda: populated.append(True)
    module.create_dataloader = lambda task: [{}]
    module.training_step = lambda batch, task: True
    module.inner_loop_u()
    assert len(populated) == 1 and module.current_epoch == 6 and module.phase_epoch == 2
    assert module.prev_integrator.sde.u_controller.value.item() == 0.4


def test_pathwise_clipping_uses_actual_state_derivatives():
    from tr_asbs.model.mlp import clip as mlp_clip
    from tr_asbs.model.egnn import clip as egnn_clip

    for clip in (mlp_clip, egnn_clip):
        x = torch.tensor([0.5, 2.0], requires_grad=True)
        actual = clip(x, max_norm=1.0, gradient_mode="exact")
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), x)[0], torch.tensor([1.0, 0.0])
        )
        legacy = clip(x, max_norm=1.0)
        torch.testing.assert_close(
            torch.autograd.grad(legacy.sum(), x)[0], torch.ones(2)
        )
