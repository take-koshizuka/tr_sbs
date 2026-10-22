import copy
from pathlib import Path

import pytest
import torch

from run_aldp_scaled_backward_multistep import reference_adam_step, row_gates, validate_plan, state_parameters


@pytest.mark.parametrize('warm', [False, True])
def test_twenty_reference_updates_preserve_adam_moments_and_input(warm):
    p = torch.nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=1e-4, eps=1e-8)
    if warm:
        p.grad = torch.tensor([.01, -.02], dtype=torch.float64)
        optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    frozen = copy.deepcopy(state)
    parameters = {'p': p.detach().clone()}
    for index in range(20):
        g = torch.tensor([1e-10*(index+1), -1e-9/(index+1)], dtype=torch.float64)
        parameters, state = reference_adam_step(parameters, {'p': g}, state)
        p.grad = g.clone()
        optimizer.step()
        torch.testing.assert_close(parameters['p'], p.detach(), rtol=0, atol=2e-16)
        torch.testing.assert_close(state['state'][0]['exp_avg'], optimizer.state[p]['exp_avg'], rtol=1e-14, atol=1e-20)
        torch.testing.assert_close(state['state'][0]['exp_avg_sq'], optimizer.state[p]['exp_avg_sq'], rtol=1e-14, atol=1e-25)
        assert int(state['state'][0]['step']) == int(optimizer.state[p]['step'])
    assert state is not frozen
    if warm:
        assert int(frozen['state'][0]['step']) == 1
        torch.testing.assert_close(frozen['state'][0]['exp_avg'], torch.tensor([.001, -.002], dtype=torch.float64))


def test_cumulative_drift_cannot_hide_behind_passing_local_steps():
    tolerances = dict(loss_relative=1e-4, gradient_relative_l2=1e-3,
                      parameter_max_absolute=1e-5, update_relative_l2=.02)
    row = dict(all_gradients_and_parameters_finite=True, loss_relative_error=0,
               clipped_gradient_error=dict(relative_l2=0), parameter_error=dict(maximum_absolute=0),
               update_error=dict(relative_l2=0), cumulative_parameter_error=dict(maximum_absolute=2e-5),
               cumulative_update_error=dict(relative_l2=.03), Adam_counters=[20],
               reference_Adam_counters=[20], expected_Adam_counters=[20])
    gates = row_gates(row, tolerances)
    assert gates['parameter'] and gates['update']
    assert not gates['cumulative_parameter'] and not gates['cumulative_update']


def valid_plan():
    return dict(updates_per_case=20, cases=['normal', 'witness'], optimizer_updates=40,
                reference_optimizer_updates=40, maximum_backward_passes=120,
                expected_OpenMM=0, OpenMM_cap=0, production_changed=False,
                full_training_submitted=False, optimizer_updates_discarded=True,
                runtime=dict(fresh_sampling=False, GPU_required=True,
                             production_dtype='FP32', matmul_precision='highest'))


@pytest.mark.parametrize('field,value', [('OpenMM_cap', 1), ('updates_per_case', 21),
                                       ('production_changed', True)])
def test_bound_guard_rejects_sampling_or_production_release(field, value):
    plan = valid_plan()
    validate_plan(plan)
    plan[field] = value
    with pytest.raises(AssertionError):
        validate_plan(plan)


def test_controller_state_has_no_persistent_unregistered_parameters():
    from tr_asbs.model.mlp import FourierMLP
    m = FourierMLP(dim=2, num_layers=2, channels=4)
    assert set(m.state_dict()) == set(dict(m.named_parameters()))


def test_fabric_parameter_keys_roundtrip_through_delegated_loader():
    from lightning.fabric import Fabric
    fabric = Fabric(accelerator='cpu', devices=1)
    fabric.launch()
    model = fabric.setup(torch.nn.Linear(2, 1))
    wrapped = dict(model.named_parameters())
    assert set(wrapped) != set(model.state_dict())
    parameters = state_parameters(model)
    assert set(parameters) == set(model.state_dict())
    frozen = {n: p.detach().clone() for n, p in parameters.items()}
    for n, p in parameters.items():
        with torch.no_grad():
            p.add_(1)
    model.load_state_dict(frozen)
    for name, parameter in state_parameters(model).items():
        torch.testing.assert_close(parameter.detach(), frozen[name], rtol=0, atol=0)


def test_worker_uses_only_diagnostic_entrypoint_and_unsets_jax():
    text = (Path(__file__).resolve().parents[1]/'scripts/slurm/exec_aldp_scaled_backward_multistep.sh').read_text()
    assert 'unset JAX_PLATFORMS' in text
    assert 'run_aldp_scaled_backward_multistep.py' in text
    assert 'train_case' not in text and '--advance' not in text
