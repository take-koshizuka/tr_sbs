"""Observe completed inner epochs without changing the shared training loop."""
from contextlib import contextmanager
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _rng_state(module):
    return dict(torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if module.fabric.device.type == "cuda" else [],
        numpy=np.random.get_state(), python=random.getstate())


@contextmanager
def isolated_inner_diagnostic(module):
    """Preserve training RNG streams and every submodule's training flag."""
    device = module.fabric.device
    devices = [device.index or 0] if device.type == "cuda" else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    modes = {}
    for name in ("controller", "corrector", "integrator", "prev_integrator", "prev_corrector"):
        model = getattr(module, name, None)
        if isinstance(model, torch.nn.Module):
            for part in model.modules():
                modes[id(part)] = (part, part.training)
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for part, flag in modes.values():
            part.training = flag


def observe_inner_boundaries(module, callback, milestones):
    """Emit phase start, selected completed epochs, and phase end.

    Intermediate checkpoints occur before the *next* buffer population, after
    all previous-epoch logging/checkpoint actions. A dataloader-exhaustion hook
    would run before the epoch counters and online-evaluation RNG draws advance.
    These boundaries consequently describe the actual prefix, including those
    draws. The surrounding module continues to own all updates and counters.
    """
    if getattr(module, "_inner_boundaries_attached", False):
        raise RuntimeError("Inner boundary observer already attached")
    module._inner_boundaries_attached = True
    milestones = {phase: frozenset(values) for phase, values in milestones.items()}
    seen = set()

    def emit(phase, event):
        epoch = int(module.phase_epoch)
        key = (int(module.outer_step), phase, epoch)
        if key in seen or (event == "epoch_boundary" and epoch not in milestones[phase]):
            return
        # A snapshot saved *inside* the diagnostic must still contain the
        # training RNG, rather than the seeded diagnostic's temporary RNG.
        module._inner_boundary_training_rng = _rng_state(module)
        try:
            with isolated_inner_diagnostic(module):
                backend = getattr(module, "jax_inner_backend", None)
                if backend is not None:
                    backend.flush()
                callback(dict(event=event, outer_loop=int(module.outer_step)+1, phase=phase,
                              phase_epoch=epoch, epoch=int(module.current_epoch),
                              global_step=int(module.global_step)))
        finally:
            del module._inner_boundary_training_rng
        seen.add(key)

    for phase, populate_name, loop_name in (
        ("u", "populate_buffer_adjoint_matching", "inner_loop_u"),
        ("h", "populate_buffer_corrector_matching", "inner_loop_h"),
    ):
        original_populate = getattr(module, populate_name)
        original_loop = getattr(module, loop_name)

        def populate(*args, phase=phase, original=original_populate, **kwargs):
            emit(phase, "phase_start" if module.phase_epoch == 0 else "epoch_boundary")
            return original(*args, **kwargs)

        def loop(*args, phase=phase, original=original_loop, **kwargs):
            result = original(*args, **kwargs)
            emit(phase, "phase_end")
            return result

        setattr(module, populate_name, populate)
        setattr(module, loop_name, loop)
    return seen


def save_inner_snapshot(module, path, provenance):
    """Save an exact epoch boundary, including the states legacy snapshots omit.

    All JAX randomness is explicit Torch input randomness in the current backend.
    JAX Adam is flushed into the ordinary optimizer states before serialization.
    The caller must invoke this at an observed completed-epoch boundary.
    """
    backend = getattr(module, "jax_inner_backend", None)
    if backend is not None:
        backend.flush(optimizers=True)
    payload = dict(schema="tr_sbs_inner_boundary_snapshot_v1", provenance=provenance,
        sampling_metadata=module.sampling_metadata(),
        models={name: getattr(module, name).state_dict() for name in (
            "controller", "corrector", "prev_integrator", "prev_corrector")},
        optimizers={name: getattr(module, name).state_dict() for name in (
            "controller_optimizer", "corrector_optimizer")},
        counters={name: getattr(module, name) for name in (
            "current_epoch", "global_step", "outer_step", "phase", "phase_epoch",
            "gamma", "V0", "gamma_h", "rank_seed", "trust_region_bound",
            "early_stop_requested") if hasattr(module, name)},
        buffers={name: dict(max_size=getattr(module, name).max_size,
            metadata=getattr(module, name).metadata,
            storage=dict(getattr(module, name).storage)) for name in (
                "adjoint_buffer", "corrector_buffer")},
        rng=getattr(module, "_inner_boundary_training_rng", None) or _rng_state(module),
        last_training_losses=getattr(module, "last_training_losses", {}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _validate_inner_snapshot(module, payload, expected_provenance):
    if payload.get("schema") != "tr_sbs_inner_boundary_snapshot_v1":
        raise ValueError("Snapshot lacks exact inner-boundary RNG/buffer state")
    if payload.get("provenance") != expected_provenance:
        raise ValueError("Inner snapshot provenance mismatch")
    if payload["sampling_metadata"] != module.sampling_metadata():
        raise ValueError("Inner snapshot sampling metadata mismatch")
    if any(payload["buffers"][name]["max_size"] != getattr(module, name).max_size
           for name in payload["buffers"]):
        raise ValueError("Inner snapshot buffer capacity mismatch")
    for name, saved in payload["models"].items():
        current = getattr(module, name).state_dict()
        if saved.keys() != current.keys() or any(saved[key].shape != current[key].shape for key in saved):
            raise ValueError("Inner snapshot model structure mismatch: " + name)
    for name, saved in payload["optimizers"].items():
        current = getattr(module, name).state_dict()
        groups = lambda state: [{key:value for key,value in group.items() if key != "params"}
                                for group in state["param_groups"]]
        if groups(saved) != groups(current):
            raise ValueError("Inner snapshot optimizer configuration mismatch: " + name)


def restore_inner_snapshot(module, path, *, expected_provenance):
    """Restore a validated boundary before continuing the shared fit loop.

    The caller must suppress fit's extra resume-time initial online draw to
    preserve the saved prefix exactly. Restore itself performs no sampling.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_inner_snapshot(module, payload, expected_provenance)
    for name, state in payload["models"].items():
        getattr(module, name).load_state_dict(state)
    module.prev_integrator.requires_grad_(False)
    module.prev_corrector.requires_grad_(False)
    for name, state in payload["optimizers"].items():
        getattr(module, name).load_state_dict(state)
    for name, value in payload["counters"].items():
        setattr(module, name, value)
    for name, state in payload["buffers"].items():
        buffer = getattr(module, name)
        buffer.clear()
        buffer.metadata = state["metadata"]
        buffer.storage = defaultdict(list, state["storage"])
    module.last_training_losses = payload["last_training_losses"]
    backend = getattr(module, "jax_inner_backend", None)
    if backend is not None:
        from .jax_inner import _adam_from_torch, export_fourier
        for phase, name in (("u", "controller"), ("h", "corrector")):
            _, params, _, _ = export_fourier(getattr(module, name))
            _, spec, state = _adam_from_torch(getattr(module, name+"_optimizer"),
                backend.models[phase], params, module.cfg.gradient_clip_val)
            if spec != backend.specs[phase]:
                raise ValueError("Inner snapshot Adam configuration mismatch")
            backend.params[phase], backend.states[phase] = params, state
        backend.old_params = None
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    if module.fabric.device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
    np.random.set_state(payload["rng"]["numpy"])
    random.setstate(payload["rng"]["python"])
    return payload["provenance"]


def continuation_training_contract(cfg):
    """Settings fixed when comparing durations; parent config SHA binds values."""
    keys = ("steps_per_epoch", "train_batch_size", "inference_batch_size",
            "initial_buffer_samples", "buffer_samples_per_epoch", "num_workers",
            "log_detail_interval_epoch", "gradient_clip_val", "ema_decay",
            "logvar_amp_dtype")
    return {key:cfg.get(key) for key in keys}


def attach_exact_inner_resume(module, path, *, expected_provenance,
                              expected_training_contract, on_restored=None):
    """Resume the shared fit once, skipping its extra initial online RNG draw.

    Attach a neural backend first with cfg.resume=False. This adapter then uses
    fit's existing load_checkpoint hook, after fit resets counters and before it
    enters the saved phase. It does not replace the shared outer/inner loops.
    An empty checkpoint-monitor set and disabled divergence/rollback logic are
    required because they otherwise have additional unsaved mutable state.
    """
    cfg = module.cfg
    if getattr(module, "_exact_inner_resume_attached", False):
        raise RuntimeError("Exact inner resume already attached")
    if cfg.resume or cfg.get("pretrained_controller_checkpoint"):
        raise ValueError("Attach exact resume before enabling any legacy/pretrained resume")
    if cfg.monitor_values or (cfg.get("divergence_early_stop") or {}).get("enabled", False):
        raise ValueError("Exact resume requires disabled checkpoint monitors/divergence state")
    if cfg.get("weight_rollback_enabled", False) or getattr(module, "trust_region_adaptive", False):
        raise ValueError("Exact resume does not restore adaptive trust/rollback history")
    if continuation_training_contract(cfg) != expected_training_contract:
        raise ValueError("Continuation changes settings beyond registered phase durations")
    # Reject impossible continuation boundaries before fit or model mutation.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_inner_snapshot(module, payload, expected_provenance)
    counters = payload.get("counters", {})
    phase, epoch = counters.get("phase"), counters.get("phase_epoch", 0)
    if phase not in ("u", "h") or epoch <= 0:
        raise ValueError("Continuation requires a positive completed inner-epoch boundary")
    end = (module.initial_adjoint_matching_epochs if counters["outer_step"] == 0
           else module.adjoint_matching_epochs) if phase == "u" else cfg.corrector_matching_epochs
    if epoch > end or counters["outer_step"] >= module.max_outer_loops:
        raise ValueError("Continuation boundary lies beyond the requested phase/outer budget")
    if counters["early_stop_requested"]:
        raise ValueError("Do not resume a stopped or failed scientific reference")
    original_log, original_load = module.log_details, getattr(module, "load_checkpoint", None)
    marker = dict(restored=False, initial_online_suppressed=False)

    def load():
        if marker["restored"]:
            raise RuntimeError("Exact continuation fit must be invoked once")
        restore_inner_snapshot(module, path, expected_provenance=expected_provenance)
        marker["restored"] = True
        if on_restored is not None:
            with isolated_inner_diagnostic(module):
                on_restored(module)

    def log(step, *args, **kwargs):
        if marker["restored"] and not marker["initial_online_suppressed"]:
            if step != 0:
                raise RuntimeError("Unexpected initial continuation logging boundary")
            marker["initial_online_suppressed"] = True
            return dict(eval_step=0, epoch=module.current_epoch,
                        exact_resume_initial_online_suppressed=True)
        return original_log(step, *args, **kwargs)

    module._exact_inner_resume_attached = True
    module.load_checkpoint, module.log_details = load, log
    cfg.resume = True
    marker["legacy_load_replaced"] = original_load is not None
    module.exact_inner_resume = marker
    return marker
