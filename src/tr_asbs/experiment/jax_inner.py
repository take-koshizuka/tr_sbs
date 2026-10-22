"""Optional JAX inner updates for the existing FourierMLP LV/BC pipeline.

The target, source, discrete kernels, buffers and random draws stay with the
shared runner.  This module owns only the neural loss, gradient and Adam step.
All random inputs are explicit, so paired tests can use identical draws.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FourierSpec:
    hidden_layers: int
    clip_norm: float | None
    clip_gradient_mode: str = "straight_through"


@jax.custom_jvp
def straight_through_clip(value, limit):
    return jnp.clip(value, -limit, limit)


@straight_through_clip.defjvp
def _straight_through_clip_jvp(primals, tangents):
    value, limit = primals
    return straight_through_clip(value, limit), tangents[0]


def fourier_mlp(params, time_coeff, spec, time, x):
    prefix = x.shape[:-1]
    x2 = x.reshape((-1, x.shape[-1]))
    t2 = jnp.broadcast_to(jnp.asarray(time, jnp.float32).reshape((-1, 1)),
                          (x2.shape[0], 1))

    def linear(value, name):
        return value @ params[name + ".weight"].T + params[name + ".bias"]

    phase = time_coeff * t2 + params["timestep_embed.timestep_phase"]
    embedded = jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)
    embedded = jax.nn.gelu(linear(embedded, "timestep_embed.hidden_layer.0"),
                           approximate=False)
    embedded = linear(embedded, "timestep_embed.out_layer")
    embedded = embedded + linear(x2, "input_embed")
    for index in range(spec.hidden_layers):
        embedded = linear(jax.nn.gelu(embedded, approximate=False),
                          f"hidden_layer.{index}")
    output = linear(jax.nn.gelu(embedded, approximate=False), "out_layer")
    if spec.clip_norm is not None:
        clipped = jnp.clip(output, -spec.clip_norm, spec.clip_norm)
        if spec.clip_gradient_mode == "straight_through":
            output = straight_through_clip(output, spec.clip_norm)
        elif spec.clip_gradient_mode == "exact":
            output = clipped
        else:
            raise ValueError("Unsupported output clip derivative")
    return output.reshape((*prefix, output.shape[-1]))


def lv_loss(params, old_params, time_coeff, spec, batch, gamma):
    trajectory = batch["data_traj"]
    size, states, dim = trajectory.shape
    steps = states - 1
    x = trajectory[:, :-1].reshape((-1, dim))
    xn = trajectory[:, 1:].reshape((-1, dim))
    time = jnp.broadcast_to(batch["times"], (size, steps)).reshape(-1)
    phi = jnp.broadcast_to(batch["phi"], (size, steps)).reshape((-1, 1))
    k = jnp.broadcast_to(batch["k"], (size, steps)).reshape((-1, 1))
    q = jnp.broadcast_to(batch["q"], (size, steps)).reshape((-1, 1))
    new = phi * x + k * fourier_mlp(params, time_coeff, spec, time, x)
    old = phi * x + k * jax.lax.stop_gradient(
        fourier_mlp(old_params, time_coeff, spec, time, x))

    def log_density(mean):
        return -.5 * (jnp.sum((xn - mean)**2 / q, axis=-1)
                      + dim * jnp.log(2 * jnp.pi * q[:, 0]))

    ratio = (log_density(new) - log_density(old)).reshape((size, steps)).sum(-1)
    weights = ratio - gamma * batch["log_Ph_Pref_raw"].reshape(-1)
    return jnp.var(weights, ddof=1), {"log_weight_mean": jnp.mean(weights)}


def bc_loss(params, time_coeff, spec, batch):
    # Analytic bridge samples, target scores and weights are supplied by the
    # existing reference. They have no trainable derivative in BC matching.
    size = batch["data_t"].shape[0]
    time = jnp.concatenate((batch["time"], jnp.ones((size,), jnp.float32)))
    data = jnp.concatenate((batch["data_t"], batch["data_1"]), axis=0)
    h = fourier_mlp(params, time_coeff, spec, time, data)
    loss_t = jnp.mean(jnp.mean((h[:size] - batch["score_t"])**2, axis=-1)
                      / batch["coef_t"].reshape(-1))
    loss_1 = jnp.mean((h[size:] - batch["score_1"])**2)
    return loss_t + loss_1, {"loss_t": loss_t, "loss_1": loss_1}


@dataclass(frozen=True)
class AdamSpec:
    lr: float = 1e-4
    beta1: float = .9
    beta2: float = .999
    eps: float = 1e-8
    weight_decay: float = 0.
    gradient_clip: float = 100.


class AdamState(NamedTuple):
    m: dict
    v: dict
    count: object


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamState(zeros, zeros, jnp.asarray(0, jnp.int32))


def adam_update(params, state, grads, loss, spec):
    # Matches torch.nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)
    # before torch.optim.Adam (coupled weight decay, epsilon outside sqrt).
    norm32 = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    if jax.config.x64_enabled:
        def widened():
            norm64=jnp.sqrt(sum(jnp.sum(g.astype(jnp.float64)**2)
                                for g in jax.tree.leaves(grads)))
            scale64=jnp.minimum(1., spec.gradient_clip/(norm64+1e-6))
            clipped=jax.tree.map(lambda g:(g.astype(jnp.float64)*scale64).astype(g.dtype),grads)
            return norm64,clipped
        def ordinary():
            scale=jnp.minimum(1.,spec.gradient_clip/(norm32+1e-6))
            return norm32.astype(jnp.float64),jax.tree.map(lambda g:g*scale,grads)
        norm,clipped=jax.lax.cond(jnp.isfinite(norm32),ordinary,widened)
    else:
        norm=norm32
        scale=jnp.minimum(1.,spec.gradient_clip/(norm+1e-6))
        clipped=jax.tree.map(lambda g:g*scale,grads)
    valid = jnp.isfinite(loss) & jnp.isfinite(norm)
    effective = jax.tree.map(lambda g, p: g + spec.weight_decay * p, clipped, params)
    m = jax.tree.map(lambda a, g: spec.beta1 * a + (1 - spec.beta1) * g,
                     state.m, effective)
    v = jax.tree.map(lambda a, g: spec.beta2 * a + (1 - spec.beta2) * g**2,
                     state.v, effective)
    count = state.count + 1
    # Compute bias corrections in float64 when available, as torch's scalar
    # Python corrections do, and cast them at their tensor use sites.
    scalar_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    c1 = 1 - jnp.asarray(spec.beta1, scalar_dtype)**count
    c2 = 1 - jnp.asarray(spec.beta2, scalar_dtype)**count
    next_params = jax.tree.map(
        lambda p, a, b: p - jnp.asarray(spec.lr / c1, p.dtype) * a
        / (jnp.sqrt(b) / jnp.asarray(jnp.sqrt(c2), p.dtype) + spec.eps), params, m, v)
    return jax.lax.cond(valid, lambda: (next_params, AdamState(m, v, count)),
                        lambda: (params, state)), norm, valid


def make_update(phase, model_spec, adam_spec, time_coeff):
    if phase not in ("u", "h"):
        raise ValueError("JAX inner phase must be u or h")

    @jax.jit
    def update(params, state, batch, old_params, gamma):
        def objective(p):
            if phase == "u":
                return lv_loss(p, old_params, time_coeff, model_spec, batch, gamma)
            return bc_loss(p, time_coeff, model_spec, batch)
        (loss, auxiliary), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        (next_params, next_state), norm, valid = adam_update(
            params, state, gradients, loss, adam_spec)
        return next_params, next_state, dict(loss=loss, gradient_norm=norm,
                                             accepted=valid, **auxiliary)

    return update


def from_torch(tensor):
    """Copy model state; external Torch tensors may subsequently be modified."""
    import torch
    tensor = tensor.detach().contiguous()
    if tensor.is_cuda:
        return jax.dlpack.from_dlpack(tensor, copy=True)
    return jnp.asarray(tensor.cpu().numpy()).copy()


def export_fourier(model):
    import torch
    from tr_asbs.model.mlp import FourierMLP
    # Fabric wrappers expose the actual network through .module.
    while not isinstance(model, FourierMLP) and hasattr(model, "module"):
        model = model.module
    if not isinstance(model, FourierMLP) or model.time_coordinates != "physical":
        raise ValueError("JAX inner backend currently requires physical-time FourierMLP")
    if not isinstance(model.activation, torch.nn.GELU) or model.activation.approximate != "none":
        raise ValueError("JAX Fourier port requires exact GELU")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("JAX Fourier port currently requires float32 weights")
    spec = FourierSpec(len(model.hidden_layer), model.clip_norm, model.clip_gradient_mode)
    params = {key: from_torch(value) for key, value in model.named_parameters()}
    return model, params, from_torch(model.timestep_embed.timestep_coeff), spec


def copy_to_torch(params, model):
    import torch
    names = dict(model.named_parameters())
    if params.keys() != names.keys():
        raise ValueError("JAX/Torch parameter names differ")
    with torch.no_grad():
        for name, value in params.items():
            if value.device.platform == "gpu":
                tensor = torch.utils.dlpack.from_dlpack(value)
            else:
                import numpy as np
                tensor = torch.from_numpy(np.asarray(value).copy())
            names[name].copy_(tensor.to(names[name]))


def prepare_lv_batch(module, batch):
    times = module.prev_integrator.timesteps.to(module.fabric.device)
    phi, k, q = module.base_sde.transition_coefficients(times[:-1], times[1:])
    trajectory = module.fabric.to_device(batch["data_traj"])
    return dict(data_traj=from_torch(trajectory), times=from_torch(times[:-1]),
                phi=from_torch(phi.to(trajectory)), k=from_torch(k.to(trajectory)),
                q=from_torch(q.to(trajectory)),
                log_Ph_Pref_raw=from_torch(module.fabric.to_device(batch["log_Ph_Pref_raw"])))


def prepare_bc_batch(module, batch):
    import torch
    data_0 = module.fabric.to_device(batch["data_0"])
    data_1 = module.fabric.to_device(batch["data_1"])
    size = len(data_0)
    time = module.fabric.to_device(torch.rand(size) * (1 - 2 * module.bridge_time_epsilon)
                                  + module.bridge_time_epsilon)
    data_t = module.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)
    time_1 = torch.ones_like(time)
    score_t = module.base_sde.cond_score_t0(time=time, data_0=data_0, data_t=data_t)
    score_1 = module.base_sde.cond_score_t0(time=time_1, data_0=data_0, data_t=data_1)
    if module.bc_loss_weight == "none":
        coef = torch.ones_like(time)
    elif module.bc_loss_weight == "reference_variance":
        _, _, variance = module.base_sde.transition_coefficients(0, time)
        coef = 1 + module.base_sde.diffusion(time).square() * (1-time) / variance.to(time)
    else:
        coef = 1 + module.base_sde.diffusion(time).square() * (1-time) / time
    return {name: from_torch(value) for name, value in dict(
        data_t=data_t, data_1=data_1, time=time, score_t=score_t, score_1=score_1, coef_t=coef).items()}


def _adam_from_torch(optimizer, model, params, gradient_clip):
    import torch
    # Fabric creates a dynamic FabricAdam subclass which is also an instance
    # of torch.optim.Optimizer. That inheritance does not mean it is unwrapped.
    while hasattr(optimizer, "optimizer") and optimizer.optimizer is not optimizer:
        optimizer = optimizer.optimizer
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise ValueError("JAX inner backend requires one Adam parameter group")
    group = optimizer.param_groups[0]
    if any(group.get(key, False) for key in ("amsgrad", "maximize", "differentiable")):
        raise ValueError("Unsupported Adam variant")
    spec = AdamSpec(group["lr"], *group["betas"], group["eps"],
                    group["weight_decay"], gradient_clip)
    state = adam_init(params)
    if optimizer.state:
        names = dict(model.named_parameters())
        if any(p not in optimizer.state for p in names.values()):
            raise ValueError("Partially populated Adam state is unsupported")
        counts = {int(optimizer.state[p]["step"].item()) for p in names.values()}
        if len(counts) != 1:
            raise ValueError("Different parameter update counts are unsupported")
        state = AdamState(
            {name: from_torch(optimizer.state[p]["exp_avg"]) for name, p in names.items()},
            {name: from_torch(optimizer.state[p]["exp_avg_sq"]) for name, p in names.items()},
            jnp.asarray(counts.pop(), jnp.int32))
    return optimizer, spec, state


class JaxInnerBackend:
    """Attach opt-in neural updates while keeping the shared training loop.

    Torch models are synchronized before sampling, diagnostics and snapshots.
    The existing checkpoint format still contains both Torch Adam states.
    It retains the original runner's limitation: buffers and RNG are not saved.
    """

    def __init__(self, module, *, require_gpu=True):
        from tr_asbs.process.integrator import LinearExponentialIntegrator
        from tr_asbs.process.reference import LinearGaussianReference
        cfg = module.cfg
        if module.fc_loss_mode != "log_variance" or module.corrector_gamma_mode != "none":
            raise ValueError("JAX inner backend currently supports LV/full undamped BC")
        if (not isinstance(module.integrator, LinearExponentialIntegrator)
                or module.integrator.control_evaluation != "left"
                or not isinstance(module.base_sde, LinearGaussianReference)
                or module.potential.is_mean_free):
            raise ValueError("JAX LV port requires full-space exact linear left-control kernels")
        if cfg.ema_decay is not None or cfg.get("logvar_amp_dtype") is not None:
            raise ValueError("JAX inner port currently requires no EMA or mixed precision")
        if cfg.resume:
            raise ValueError("Attach the JAX backend only after an explicitly restored full state")
        if module.fabric.world_size != 1:
            raise ValueError("JAX inner port currently requires a single process")
        if require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("JAX GPU backend unavailable; refusing a CPU training fallback")
        self.module = module
        self.models, self.params, self.optimizers, self.specs, self.states, self.updates = {}, {}, {}, {}, {}, {}
        for phase, name in (("u", "controller"), ("h", "corrector")):
            model, params, coeff, spec = export_fourier(getattr(module, name))
            optimizer, adam_spec, state = _adam_from_torch(
                getattr(module, name + "_optimizer"), model, params, cfg.gradient_clip_val)
            self.models[phase], self.params[phase], self.optimizers[phase] = model, params, optimizer
            self.specs[phase], self.states[phase] = adam_spec, state
            self.updates[phase] = make_update(phase, spec, adam_spec, coeff)
        self.old_params = None
        self.attached = False

    def flush(self, *, optimizers=False):
        import numpy as np
        import torch
        for phase in ("u", "h"):
            copy_to_torch(self.params[phase], self.models[phase])
            if optimizers and int(self.states[phase].count):
                state = self.states[phase]
                for name, param in self.models[phase].named_parameters():
                    target = self.optimizers[phase].state[param]
                    # Adam's non-capturable default keeps its step counter on CPU.
                    target["step"] = torch.tensor(float(state.count), dtype=torch.float32)
                    for key, values in (("exp_avg", state.m), ("exp_avg_sq", state.v)):
                        value = values[name]
                        tensor = (torch.utils.dlpack.from_dlpack(value) if value.device.platform == "gpu"
                                  else torch.from_numpy(np.asarray(value).copy()))
                        target[key] = tensor.to(param).clone()

    def training_step(self, batch, current_task):
        import numpy as np
        phase = {"adjoint_matching": "u", "corrector_matching": "h"}[current_task]
        prepared = (prepare_lv_batch(self.module, batch) if phase == "u"
                    else prepare_bc_batch(self.module, batch))
        if phase == "u" and self.old_params is None:
            _, self.old_params, _, _ = export_fourier(self.module.prev_integrator.sde.u_controller)
        old = self.old_params if phase == "u" else self.params[phase]
        params, state, values = self.updates[phase](self.params[phase], self.states[phase],
                                                  prepared, old, self.module.gamma)
        # Synchronize to detect a failed step, rather than enqueueing an invalid
        # trajectory. A future multi-step scan must preserve this acceptance rule.
        values = {name: np.asarray(value).item() for name, value in values.items()}
        if not values.pop("accepted"):
            return False
        self.params[phase], self.states[phase] = params, state
        self.module.last_training_losses = {k: float(v) for k, v in values.items()}
        if self.module.global_step % self.module.cfg.log_loss_interval_step == 0:
            self.module.fabric.log_dict({f"train/{current_task}_{k}": float(v)
                                         for k, v in values.items()}, step=self.module.global_step)
        return True

    def attach(self):
        if self.attached:
            raise RuntimeError("JAX inner backend already attached")
        self.attached = True
        module = self.module
        module.training_step = self.training_step
        # These are the boundaries at which the surrounding Torch pipeline
        # reads neural parameters or serializes optimizers.
        for name in ("populate_buffer_adjoint_matching", "populate_buffer_corrector_matching",
                     "inner_loop_u", "inner_loop_h", "optimize_gamma", "log_details",
                     "log_final", "save_checkpoint"):
            original = getattr(module, name)

            def wrapped(*args, original=original, name=name, **kwargs):
                self.flush(optimizers=name == "save_checkpoint")
                if name == "inner_loop_u":
                    self.old_params = None
                result = original(*args, **kwargs)
                if name in ("inner_loop_u", "inner_loop_h"):
                    self.flush()
                return result
            setattr(module, name, wrapped)
        module.jax_inner_backend = self
        return self
