"""Normal asymmetric accounting and actual tiny shared-loop JAX lifecycle."""

import pytest

pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
from copy import deepcopy
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_aldp_jax_inner_curve import InnerCurveObserver, expected_cost, validate_contract
from tr_asbs.data.buffer import BatchBuffer
from tr_asbs.experiment.jax_inner import JaxInnerBackend, export_fourier
from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
from tr_asbs.process.integrator import LinearExponentialIntegrator
from tr_asbs.process.sde import ControlledFBSDE
from tr_asbs.process.source import GaussianPrior
from test_jax_inner import context, model


def test_asymmetric_normal_cost_counts_training_and_prefix_probes():
    cfg = SimpleNamespace(adjoint_matching_epochs=2000, corrector_matching_epochs=1000,
        steps_per_epoch=100, train_batch_size=64, inference_batch_size=64,
        integrator=SimpleNamespace(timesteps=SimpleNamespace(steps=200)),
        num_samples_for_dual_loss=[32,8], buffer_samples_per_epoch=128,
        eval_samples=64, log_detail_interval_epoch=250)
    contract = dict(quality_samples=1024, phase_samples=128, validation_samples=256,
        milestones={'u':[250,500,1000,1500,2000], 'h':[125,250,500,750,1000]},
        selection_bank='a', test_bank='closed')
    validate_contract(cfg, contract)
    actual = expected_cost(cfg, contract)
    assert actual == dict(training_and_online_openmm_configurations=257088,
        diagnostic_openmm_configurations=26240,
        final_evaluation_openmm_configurations=2048,
        total_openmm_configurations=285376, u_optimizer_steps=200000,
        h_optimizer_steps=100000, optimizer_steps=300000)


def gaussian_shared_fit(*,u_epochs=2,h_epochs=1):
    from lightning.fabric import Fabric
    from omegaconf import OmegaConf
    module = context(model())
    fabric = Fabric(accelerator='cpu', devices=1)
    fabric.launch()
    u, h = model(), deepcopy(model())
    u_opt, h_opt = torch.optim.Adam(u.parameters(), lr=1e-4), torch.optim.Adam(h.parameters(), lr=1e-4)
    module.controller, module.controller_optimizer = fabric.setup(u, u_opt)
    module.corrector, module.corrector_optimizer = fabric.setup(h, h_opt)
    module.fabric = fabric
    module.integrator = LinearExponentialIntegrator(
        ControlledFBSDE(module.base_sde, u, h),
        timesteps=torch.linspace(0,1,5))
    module.prev_integrator, module.prev_corrector = deepcopy(module.integrator), deepcopy(module.corrector)
    module.potential = SimpleNamespace(is_mean_free=False, log_prob=lambda x:-.5*x.square().sum(-1))
    module.source = GaussianPrior(dim=3, scale=1.)
    module.fc_loss_mode = 'log_variance'
    module.cfg = OmegaConf.create(dict(adjoint_matching_epochs=u_epochs, corrector_matching_epochs=h_epochs,
        steps_per_epoch=2, train_batch_size=5, inference_batch_size=5,
        initial_buffer_samples=10, buffer_samples_per_epoch=10, log_loss_interval_step=10,
        log_buffer_energy=False, log_buffer_grad_norm=False, progress_bar=False,
        log_detail_interval_epoch=1, monitor_values=[], monitor_initial_state=False,
        num_workers=0, ema_decay=None, logvar_amp_dtype=None, resume=False,
        gradient_clip_val=100.))
    module.adjoint_buffer, module.corrector_buffer = BatchBuffer(40), BatchBuffer(40)
    module.adjoint_buffer_duplicates = 1
    module.sampling_metadata = lambda:{'unit_test':'actual Gaussian shared loop'}
    module.clear_adjoint_buffer_per_outer_loop = True
    module.initial_adjoint_matching_epochs = module.adjoint_matching_epochs = u_epochs
    module.max_outer_loops, module.total_epochs, module.rank_seed = 2, 2*(u_epochs+h_epochs), 42
    module.current_epoch = module.global_step = module.outer_step = module.phase_epoch = 0
    module.phase, module.gamma_schedule, module.trust_region_adaptive = 'ready', 'trust_region', False
    module.trust_region_bound, module.V0, module.gamma_h = 1., .3, 1.
    module.early_stop_requested = False
    module.optimize_gamma = lambda:(.2,.3)
    module._weight_rollback_check = module._update_divergence_early_stop = lambda metrics:None
    online, checkpoints, teachers = [], [], []

    def synchronized_models():
        for phase, name in (('u','controller'), ('h','corrector')):
            _, values, _, _ = export_fourier(getattr(module,name))
            for key in values:
                np.testing.assert_array_equal(values[key], backend.params[phase][key])

    def log(step):
        synchronized_models()
        torch.rand(7)  # Real online probes consume training RNG; resume must skip its extra initial call.
        online.append((module.current_epoch, module.global_step))
        return {}

    def checkpoint(**kwargs):
        synchronized_models()
        for phase, opt in (('u',u_opt), ('h',h_opt)):
            counts = [int(state['step']) for state in opt.state.values()]
            count = int(backend.states[phase].count)
            assert (counts and set(counts) == {count}) if count else not counts
        checkpoints.append(module.global_step)

    module.log_details, module.log_final, module.save_checkpoint = log, lambda:None, checkpoint
    for name in ('populate_buffer_adjoint_matching','populate_buffer_corrector_matching',
                 'create_dataloader','inner_loop_u','inner_loop_h','fit'):
        setattr(module,name,MethodType(getattr(GammaASBSModule,name),module))
    population = module.populate_buffer_adjoint_matching

    def check_teacher():
        if module.phase_epoch == 0:
            values = module.prev_integrator.sde.u_controller.state_dict()
            current = module.controller.state_dict()
            for key in values:
                torch.testing.assert_close(values[key],current[key],rtol=0,atol=0)
            teachers.append(module.outer_step)
        return population()

    module.populate_buffer_adjoint_matching = check_teacher
    backend = JaxInnerBackend(module, require_gpu=False).attach()
    module.test_backend, module.test_online = backend, online
    module.test_checkpoints, module.test_teachers = checkpoints, teachers
    return module


def test_actual_shared_fit_jax_syncs_logging_checkpoints_and_next_outer_teacher():
    module = gaussian_shared_fit()
    backend = module.test_backend
    module.fit()
    assert module.global_step == 12 and module.current_epoch == 6
    assert int(backend.states['u'].count) == 8 and int(backend.states['h'].count) == 4
    assert module.test_teachers == [0,1] and module.test_online[-1] == (6,12)
    assert module.test_checkpoints[-1] == 12


def test_boundary_diagnostic_restores_rng_before_shared_caller_checks_it(tmp_path):
    from tr_asbs.experiment.inner_observation import observe_inner_boundaries
    reference=gaussian_shared_fit();torch.manual_seed(1234);reference.fit()
    final_rng=torch.get_rng_state().clone()
    module=gaussian_shared_fit();torch.manual_seed(1234)
    module.adjoint_matching_log_variance_step=MethodType(
        GammaASBSModule.adjoint_matching_log_variance_step,module)
    module.potential.aldp_potential=SimpleNamespace(openmm_configurations_evaluated=0)
    observer=InnerCurveObserver.__new__(InnerCurveObserver)
    observer.contract={'validation_samples':5}
    observer.output,observer.validation,observer.report=tmp_path,{}, {'inner_boundaries':[]}
    observer.provenance={'test':'real boundary observer, no molecular work'}
    observer.started=time.perf_counter()
    observer.policy=lambda *args:(torch.rand(17),{'test':'diagnostic randomness'})[1]

    def shared_caller(payload):
        before=torch.get_rng_state().clone()
        observer(module,'inner_boundary',payload)
        # Check *inside* the boundary hook, matching the failing GPU caller.
        assert torch.equal(before,torch.get_rng_state())

    observe_inner_boundaries(module,shared_caller,{'u':[1],'h':[1]})
    module.fit()
    assert torch.equal(final_rng,torch.get_rng_state())
    for phase in ('u','h'):
        for key,value in reference.test_backend.params[phase].items():
            np.testing.assert_array_equal(value,module.test_backend.params[phase][key])
    assert len(observer.report['inner_boundaries'])==10
