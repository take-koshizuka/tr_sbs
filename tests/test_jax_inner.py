"""Small cross-framework tests; no ALDP construction or molecular evaluations."""

from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tr_asbs.experiment.jax_inner import (
    AdamSpec, JaxInnerBackend, adam_init, adam_update, bc_loss, copy_to_torch, export_fourier,
    fourier_mlp, from_torch, lv_loss, make_update, prepare_bc_batch, prepare_lv_batch,
)
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
from tr_asbs.model.mlp import FourierMLP
from tr_asbs.process.integrator import LinearExponentialIntegrator
from tr_asbs.process.reference import LinearGaussianReference
from tr_asbs.process.sde import ControlledFBSDE

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def model(clip=100., mode="straight_through"):
    torch.manual_seed(1729)
    result = FourierMLP(dim=3, channels=8, num_layers=3,
                        clip_norm=clip, clip_gradient_mode=mode)
    with torch.no_grad():
        result.out_layer.weight.normal_(0., .15)
        result.out_layer.bias.normal_(0., .15)
    return result


def context(network):
    reference = LinearGaussianReference(kind="ou", rate=1., diffusion_scale=.5)
    process = ControlledFBSDE(reference, network, deepcopy(network))
    integrator = LinearExponentialIntegrator(process, timesteps=torch.linspace(0, 1, 5))
    return SimpleNamespace(controller=network, corrector=network,
        integrator=integrator, prev_integrator=deepcopy(integrator), base_sde=reference,
        fabric=SimpleNamespace(device=torch.device("cpu"), to_device=lambda x: x),
        gamma=.2, logvar_amp_dtype=None, bridge_time_epsilon=.001,
        ones_tensor=torch.tensor(1.), cfg=SimpleNamespace(train_batch_size=5),
        corrector_gamma_mode="none", bc_loss_weight="legacy")


def assert_gradients(network, gradients, *, rtol=5e-4, atol=2e-5):
    for name, value in network.named_parameters():
        np.testing.assert_allclose(np.asarray(gradients[name]), value.grad.numpy(),
                                   rtol=rtol, atol=atol, err_msg=name)


@pytest.mark.parametrize("mode", ["straight_through", "exact"])
def test_forward_and_saturated_output_derivative(mode):
    network = model(.05, mode)
    _, params, coeff, spec = export_fourier(network)
    time, x = torch.linspace(.1, .9, 5), torch.randn(5, 3)
    expected = network(time, x)
    actual = fourier_mlp(params, coeff, spec, from_torch(time), from_torch(x))
    np.testing.assert_allclose(np.asarray(actual), expected.detach().numpy(), atol=1e-6, rtol=1e-5)
    expected.square().sum().backward()
    grads = jax.grad(lambda p: jnp.sum(fourier_mlp(
        p, coeff, spec, from_torch(time), from_torch(x))**2))(params)
    assert_gradients(network, grads)


def test_actual_lv_kernel_loss_and_gradient():
    network = model()
    module = context(network)
    # Distinct old/current policies exercise exact discrete likelihood ratios.
    with torch.no_grad():
        module.prev_integrator.sde.u_controller.out_layer.bias.add_(.02)
    batch = dict(data_traj=torch.randn(5, 5, 3) * .2,
                 log_Ph_Pref_raw=torch.randn(5))
    _, params, coeff, spec = export_fourier(network)
    _, old, _, _ = export_fourier(module.prev_integrator.sde.u_controller)
    prepared = prepare_lv_batch(module, batch)
    expected = GammaASBSModule.adjoint_matching_log_variance_step(module, batch)["loss"]
    (actual, _), grads = jax.value_and_grad(lv_loss, has_aux=True)(
        params, old, coeff, spec, prepared, module.gamma)
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-4, atol=2e-5)
    expected.backward()
    assert_gradients(network, grads, rtol=7e-4, atol=8e-5)


@pytest.mark.parametrize("weight", ["legacy", "reference_variance", "none"])
def test_actual_bc_loss_draws_and_gradient(weight):
    network = model()
    module = context(network)
    module.bc_loss_weight = weight
    batch = dict(data_0=torch.randn(5, 3), data_1=torch.randn(5, 3))
    _, params, coeff, spec = export_fourier(network)
    torch.manual_seed(234)
    expected = GammaASBSModule.corrector_matching_step(module, batch)["loss"]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(234)
    prepared = prepare_bc_batch(module, batch)
    assert torch.equal(expected_rng, torch.get_rng_state())
    (actual, _), grads = jax.value_and_grad(bc_loss, has_aux=True)(params, coeff, spec, prepared)
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=2e-5, atol=1e-5)
    expected.backward()
    assert_gradients(network, grads)


def test_adam_clip_bias_corrections_and_nonfinite_rejection():
    tensor = torch.nn.Parameter(torch.tensor([.4, -.2, .7]))
    optimizer = torch.optim.Adam([tensor], lr=1e-4, weight_decay=.01)
    params = {"p": from_torch(tensor)}
    state = adam_init(params)
    spec = AdamSpec(weight_decay=.01, gradient_clip=.3)
    for i in range(7):
        grad = torch.tensor([.1, -.5, .25]) * (i+1)
        tensor.grad = grad.clone()
        torch.nn.utils.clip_grad_norm_([tensor], .3, error_if_nonfinite=True)
        optimizer.step()
        (params, state), _, accepted = adam_update(params, state, {"p": from_torch(grad)}, 1., spec)
        assert bool(accepted)
        np.testing.assert_allclose(params["p"], tensor.detach().numpy(), rtol=1e-6, atol=1e-7)
    before = params
    (after, next_state), _, accepted = adam_update(params, state,
        {"p": jnp.full_like(params["p"], jnp.nan)}, 1., spec)
    assert not bool(accepted) and int(next_state.count) == 7
    np.testing.assert_array_equal(after["p"], before["p"])


def test_jitted_update_and_weight_roundtrip():
    network = model()
    _, params, coeff, spec = export_fourier(network)
    context_model = context(network)
    prepared = prepare_bc_batch(context_model,
        dict(data_0=torch.randn(5, 3), data_1=torch.randn(5, 3)))
    update = make_update("h", spec, AdamSpec(), coeff)
    params, state, values = update(params, adam_init(params), prepared, params, .2)
    assert int(state.count) == 1 and bool(values["accepted"])
    copy_to_torch(params, network)
    _, restored, _, _ = export_fourier(network)
    for name in params:
        np.testing.assert_array_equal(params[name], restored[name])


def test_unsupported_time_embedding_is_rejected():
    with pytest.raises(ValueError, match="physical-time"):
        export_fourier(FourierMLP(3, channels=8, time_coordinates="variance"))


def test_shared_step_adapter_rng_weights_and_adam_handoff():
    from omegaconf import OmegaConf
    reference = model()
    network = deepcopy(reference)
    module = context(network)
    module.potential = SimpleNamespace(is_mean_free=False)
    module.fc_loss_mode = "log_variance"
    module.cfg = OmegaConf.create(dict(ema_decay=None, logvar_amp_dtype=None, resume=False,
        gradient_clip_val=100., train_batch_size=5, log_loss_interval_step=10))
    module.fabric.world_size = 1
    module.fabric.log_dict = lambda *args, **kwargs: None
    module.global_step = 0
    module.controller_optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    # This test only updates h; u/h share a tiny model solely in this fixture.
    module.corrector_optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    baseline = context(reference)
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4)
    backend = JaxInnerBackend(module, require_gpu=False)
    batch = dict(data_0=torch.randn(5, 3), data_1=torch.randn(5, 3))
    for _ in range(3):
        torch.manual_seed(543)
        expected = GammaASBSModule.corrector_matching_step(baseline, batch)["loss"]
        expected_rng = torch.get_rng_state().clone()
        optimizer.zero_grad()
        expected.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 100.)
        optimizer.step()
        torch.manual_seed(543)
        assert backend.training_step(batch, "corrector_matching")
        assert torch.equal(expected_rng, torch.get_rng_state())
    # Sync h alone because the fixture intentionally shares the u/h model.
    copy_to_torch(backend.params["h"], network)
    for (_, actual), (_, expected) in zip(network.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-7)
    module.controller = deepcopy(network)
    backend.models["u"] = module.controller
    backend.flush(optimizers=True)
    for name, p in network.named_parameters():
        state = module.corrector_optimizer.state[p]
        assert state["step"] == 3
        np.testing.assert_allclose(state["exp_avg"].numpy(), backend.states["h"].m[name], atol=1e-7)


def test_actual_fabric_adam_wrapper_is_unwrapped_without_state_loss():
    from lightning.fabric import Fabric
    from tr_asbs.experiment.jax_inner import _adam_from_torch
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    network = model()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    wrapped_model, wrapped_optimizer = fabric.setup(network, optimizer)
    assert isinstance(wrapped_optimizer, torch.optim.Optimizer)
    assert type(wrapped_optimizer) is not torch.optim.Adam
    loss = wrapped_model(torch.ones(5), torch.ones(5, 3)).square().mean()
    wrapped_optimizer.zero_grad()
    fabric.backward(loss)
    wrapped_optimizer.step()
    raw, params, _, _ = export_fourier(wrapped_model)
    restored, spec, state = _adam_from_torch(wrapped_optimizer, raw, params, 100.)
    assert restored is optimizer and int(state.count) == 1 and spec.lr == 1e-4
    for name, p in raw.named_parameters():
        np.testing.assert_array_equal(state.m[name], optimizer.state[p]["exp_avg"].numpy())
        np.testing.assert_array_equal(state.v[name], optimizer.state[p]["exp_avg_sq"].numpy())
