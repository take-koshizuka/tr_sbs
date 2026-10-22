"""Regression for calling shared buffer APIs before fit; Gaussian-only."""

import pytest

pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
from pathlib import Path
from copy import deepcopy
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_aldp_jax_inner_preflight import initialize_backend_probe
from tr_asbs.data.buffer import BatchBuffer
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
from tr_asbs.process.source import GaussianPrior
from test_jax_inner import context, model


def test_manual_probe_populates_actual_lv_buffer_without_fit():
    module = context(model())
    assert not hasattr(module, "current_epoch")
    module.cfg = SimpleNamespace(train_batch_size=5, adjoint_matching_epochs=2000,
        inference_batch_size=5, initial_buffer_samples=10, buffer_samples_per_epoch=10,
        log_loss_interval_step=10, log_buffer_energy=False, log_buffer_grad_norm=False,
        progress_bar=False)
    module.initial_adjoint_matching_epochs = 2000
    module.fc_loss_mode = "log_variance"
    module.adjoint_buffer_duplicates = 1
    module.adjoint_buffer = BatchBuffer(40)
    module.source = GaussianPrior(dim=3, scale=1.)
    module.potential = SimpleNamespace(is_mean_free=False,
        log_prob=lambda x: -.5 * x.square().sum(-1))
    module.sampling_metadata = lambda: {"test": "frozen Gaussian kernel"}
    module.V0 = 1e-9
    module.fabric.global_rank = 0
    module.fabric.log = lambda *args, **kwargs: None
    module.fabric.barrier = lambda: None
    initialize_backend_probe(module)
    GammaASBSModule.populate_buffer_adjoint_matching(module)
    assert module.current_epoch == 0 and module.global_step == 0
    assert len(module.adjoint_buffer) == 10
    trajectory = torch.stack(module.adjoint_buffer.storage["data_traj"])
    assert trajectory.shape == (10, 5, 3) and torch.isfinite(trajectory).all()


@pytest.mark.parametrize("corrupt_prepared", [False, True])
def test_saved_rejection_probe_preserves_adam_and_restores_state(tmp_path, monkeypatch, corrupt_prepared):
    """Tiny synthetic Gaussian evidence tests plumbing, not molecular equivalence."""
    import run_aldp_jax_inner_preflight as runner
    from tr_asbs.experiment.jax_inner import export_fourier, prepare_lv_batch

    module = context(model())
    module.controller_optimizer = torch.optim.Adam(module.controller.parameters(), lr=1e-4)
    module.adjoint_matching_log_variance_step = lambda batch: (
        GammaASBSModule.adjoint_matching_log_variance_step(module, batch))
    original_controller = deepcopy(module.controller.state_dict())
    teacher = module.prev_integrator.sde.u_controller
    original_teacher = deepcopy(teacher.state_dict())
    original_optimizer = deepcopy(module.controller_optimizer.state_dict())
    original_gamma = module.gamma
    batch = dict(data_traj=torch.randn(5, 5, 3)*.2, log_Ph_Pref_raw=torch.randn(5))
    _, params, coeff, _ = export_fourier(module.controller)
    _, old, _, _ = export_fourier(teacher)
    prepared = {name: torch.utils.dlpack.from_dlpack(value).clone()
                for name, value in prepare_lv_batch(module, batch).items()}
    if corrupt_prepared:
        prepared["phi"][0] += .01
    evidence = dict(schema="aldp_numerical_rejection_batch_v1",
        counters=dict(global_step=8208, gamma=original_gamma),
        params={name: torch.utils.dlpack.from_dlpack(value).clone() for name, value in params.items()},
        old_params={name: torch.utils.dlpack.from_dlpack(value).clone() for name, value in old.items()},
        time_coeff=torch.utils.dlpack.from_dlpack(coeff).clone(), prepared=prepared,
        adam=dict(count=8208,
            m={name: torch.full_like(value, .001) for name, value in module.controller.named_parameters()},
            v={name: torch.full_like(value, .01) for name, value in module.controller.named_parameters()}))
    path = tmp_path / "synthetic_rejected_batch.pt"
    torch.save(evidence, path)
    benchmark_calls = []

    def benchmark(current, phase, actual_batch, iterations, warmup, tolerances):
        assert current is module and phase == "u"
        for name, parameter in current.controller.named_parameters():
            state = current.controller_optimizer.state[parameter]
            assert state["step"] == 8208
            torch.testing.assert_close(state["exp_avg"], evidence["adam"]["m"][name])
            torch.testing.assert_close(state["exp_avg_sq"], evidence["adam"]["v"][name])
        benchmark_calls.append((iterations, warmup))
        return dict(status="passed", phase="u", cpu_fixture_mock=True)

    monkeypatch.setattr(runner, "benchmark_updates", benchmark)
    plan = dict(numerical_repair_evidence_path=str(path), paired_updates=20, warmup_updates=3,
        tolerances=dict(loss_relative=1e-4, gradient_relative_l2=1e-3))
    if corrupt_prepared:
        with pytest.raises(AssertionError):
            runner.numerical_repair_preflight(module, plan)
        assert not benchmark_calls
    else:
        report = runner.numerical_repair_preflight(module, plan)
        assert report["status"] == "passed" and report["prior_adam_updates"] == 8208
        assert report["gradient"]["status"] == "passed"
        assert report["clipping"]["forward_exact_bound"]
        assert not report["tolerance_relaxation"] and not report["target_recomputed"]
        assert benchmark_calls == [(20, 3)]
    for name, value in module.controller.state_dict().items():
        torch.testing.assert_close(value, original_controller[name], rtol=0, atol=0)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original_teacher[name], rtol=0, atol=0)
    assert module.controller_optimizer.state_dict() == original_optimizer
    assert module.gamma == original_gamma


@pytest.mark.parametrize("phase", ["u", "h"])
def test_paired_benchmark_resets_nonempty_adam_without_mutating_saved_baseline(monkeypatch, phase):
    """Real tiny Torch/JAX steps exercise warmup/reset at an8208-update state."""
    from omegaconf import OmegaConf
    import run_aldp_jax_inner_preflight as runner
    from tr_asbs.experiment.jax_inner import JaxInnerBackend

    module = context(model())
    module.corrector = module.integrator.sde.h_controller
    module.cfg = OmegaConf.create(dict(ema_decay=None, logvar_amp_dtype=None, resume=False,
        gradient_clip_val=100., train_batch_size=5, log_loss_interval_step=10))
    module.potential = SimpleNamespace(is_mean_free=False)
    module.fc_loss_mode = "log_variance"
    module.fabric.world_size = 1
    module.fabric.backward = lambda loss: loss.backward()
    module.fabric.clip_gradients = lambda network, optimizer, **kwargs: (
        torch.nn.utils.clip_grad_norm_(network.parameters(), kwargs["max_norm"],
                                      error_if_nonfinite=kwargs["error_if_nonfinite"]))
    module.fabric.log = module.fabric.log_dict = lambda *args, **kwargs: None
    module.fabric.all_reduce = lambda value, **kwargs: value
    module.global_step = 0
    module.adjoint_matching_log_variance_step = lambda batch: (
        GammaASBSModule.adjoint_matching_log_variance_step(module, batch))
    module.corrector_matching_step = lambda batch: GammaASBSModule.corrector_matching_step(module, batch)
    torch_counts, jax_counts = [], []
    for name in ("controller", "corrector"):
        network = getattr(module, name)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        for parameter in network.parameters():
            optimizer.state[parameter] = dict(step=torch.tensor(8208.),
                exp_avg=torch.full_like(parameter, .02), exp_avg_sq=torch.full_like(parameter, .04))
        setattr(module, name+"_optimizer", optimizer)
    optimizer = module.controller_optimizer if phase == "u" else module.corrector_optimizer

    def torch_step(batch, task):
        torch_counts.append(int(next(iter(optimizer.state.values()))["step"]))
        return GammaASBSModule.training_step(module, batch, task)

    module.training_step = torch_step
    saved_weights = [deepcopy(getattr(module, name).state_dict()) for name in ("controller", "corrector")]
    saved_states = [deepcopy(getattr(module, name+"_optimizer").state_dict())
                    for name in ("controller", "corrector")]

    class CPUBackend(JaxInnerBackend):
        def __init__(self, current):
            super().__init__(current, require_gpu=False)

        def training_step(self, batch, task):
            jax_counts.append(int(self.states[phase].count))
            return super().training_step(batch, task)

    monkeypatch.setattr(runner, "JaxInnerBackend", CPUBackend)
    monkeypatch.setattr(runner, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state().clone())
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    batch = (dict(data_traj=torch.randn(5, 5, 3)*.2, log_Ph_Pref_raw=torch.randn(5)) if phase == "u"
             else dict(data_0=torch.randn(5, 3), data_1=torch.randn(5, 3)))
    report = runner.benchmark_updates(module, phase, batch, 4, 2,
        dict(parameter_max_absolute=1e-5, update_relative_l2=.02))
    assert torch_counts == [8208, 8209, 8208, 8209, 8210, 8211]
    assert jax_counts == [8208, 8208, 8209, 8208, 8209, 8210, 8211]
    assert report["status"] == "passed"
    for index, name in enumerate(("controller", "corrector")):
        for key, value in getattr(module, name).state_dict().items():
            torch.testing.assert_close(value, saved_weights[index][key], rtol=0, atol=0)
        state = getattr(module, name+"_optimizer").state_dict()
        for key, values in state["state"].items():
            for field, value in values.items():
                torch.testing.assert_close(value, saved_states[index]["state"][key][field], rtol=0, atol=0)
