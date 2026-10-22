"""Curriculum, small-gamma, and diagnostic invariants; no molecular training."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_aldp_stability_campaign import configuration, dual_diagnostics, expected_cost, verify_stability_report


class TimeControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1., dtype=torch.double))

    def forward(self, time, x):
        return self.scale * (x + 2*time[:, None])


def fc_loss(anchor, gamma):
    previous = TimeControl()
    current = deepcopy(previous)
    data_0 = torch.tensor([[0.], [1.]], dtype=torch.double)
    data_1 = torch.tensor([[3.], [4.]], dtype=torch.double)
    obj = SimpleNamespace(cfg=SimpleNamespace(train_batch_size=2),
        fabric=SimpleNamespace(to_device=lambda x: x), gamma=gamma,
        zeros_tensor=torch.zeros(2), ones_tensor=torch.ones(2), bridge_time_epsilon=.001,
        per_time_anchor_target=anchor, prev_integrator=SimpleNamespace(
            sde=SimpleNamespace(u_controller=previous)), controller=current,
        base_sde=SimpleNamespace(sample_posterior=lambda time, data_0, data_1:
            (1-time[:, None])*data_0+time[:, None]*data_1), fc_loss_weight='legacy',
        _compute_iw=lambda batch, gamma: torch.ones(2))
    batch=dict(data_0=data_0, data_1=data_1, terminal_cost=torch.ones_like(data_1),
               control_0=previous(torch.zeros(2), data_0).detach(),
               control_1=previous(torch.ones(2), data_1).detach())
    torch.manual_seed(42)
    loss = GammaASBSModule.adjoint_matching_step(obj, batch)['loss']
    return loss, current


def test_time_anchor_is_no_op_at_zero_gamma():
    anchored, model = fc_loss(True, 0.)
    anchored.backward()
    assert anchored.item() == 0 and model.scale.grad.item() == 0
    legacy, _ = fc_loss(False, 0.)
    assert legacy.item() > 0


def test_anchor_preserves_unit_gamma_target():
    legacy, _ = fc_loss(False, 1.)
    anchored, _ = fc_loss(True, 1.)
    torch.testing.assert_close(anchored, legacy)


def test_only_first_u_phase_gets_extra_updates():
    module = SimpleNamespace(cfg=SimpleNamespace(adjoint_matching_epochs=1,
        progress_bar=False, log_detail_interval_epoch=999),
        initial_adjoint_matching_epochs=3, adjoint_matching_epochs=1,
        integrator=nn.Linear(1, 1), prev_integrator=nn.Linear(1, 1),
        clear_adjoint_buffer_per_outer_loop=False, outer_step=0, phase='ready', phase_epoch=0,
        current_epoch=0, global_step=0, total_epochs=6, potential=SimpleNamespace(),
        fabric=SimpleNamespace(global_rank=0), populate_buffer_adjoint_matching=lambda: None,
        create_dataloader=lambda task: [None], training_step=lambda batch, task: True)
    GammaASBSModule.inner_loop_u(module)
    assert module.global_step == 3 and module.current_epoch == 3
    module.outer_step, module.phase, module.phase_epoch = 1, 'ready', 0
    GammaASBSModule.inner_loop_u(module)
    assert module.global_step == 4 and module.current_epoch == 4


def test_dual_shift_invariance_and_rng_independence():
    values = np.array([[1., -2.], [3., -5.], [-1., 4.]])
    torch.manual_seed(81)
    before = torch.get_rng_state().clone()
    base = dual_diagnostics(values, .2, -3., 1., bootstrap_count=8)
    shifted = dual_diagnostics(values+1000, .2, -3., 1., bootstrap_count=8)
    assert torch.equal(before, torch.get_rng_state())
    assert base['optimizer_success'] and base['additional_openmm_configurations'] == 0
    assert base['independent_solve_gamma'] == pytest.approx(shifted['independent_solve_gamma'], rel=1e-5)
    assert base['bootstrap_gamma_q05_q50_q95'] == pytest.approx(
        shifted['bootstrap_gamma_q05_q50_q95'], rel=1e-5)


@pytest.mark.parametrize('bad', [np.empty((0, 2)), [[np.nan, 0]], [1., 2.]])
def test_dual_rejects_invalid_paths(bad):
    with pytest.raises(ValueError):
        dual_diagnostics(bad, .2, -3., 1., bootstrap_count=8)


def test_candidates_hold_all_other_scientific_fields(tmp_path):
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'reports/aldp_chart_20261007_5a093a95/physical_learning_v1/plan.json').read_text())
    plan['candidates']={name: dict(source_scale=1., terminal_gradient_clip=1000., outer_loops=10,
        chart_key='low_density', **override) for name, override in dict(
            baseline={}, initial_u1000={'initial_adjoint_matching_epochs':500},
            time_anchor={'per_time_anchor_target':True}, bc_variance={'bc_loss_weight':'reference_variance'},
            dual_groups4x={'dual_samples':[128,8]}).items()}
    baseline,_=configuration(plan,tmp_path,'mechanism_pilot','baseline',0)
    baseline=OmegaConf.to_container(baseline,resolve=True)
    keys=('initial_adjoint_matching_epochs','per_time_anchor_target','bc_loss_weight',
        'num_samples_for_dual_loss','max_outer_loops','adjoint_matching_epochs','corrector_matching_epochs',
        'fc_loss_mode','controller_optimizer','corrector_optimizer','potential','source','base_sde','integrator',
        'controller','corrector','train_batch_size','terminal_cost','trust_region_bound')
    for name,key in [('initial_u1000','initial_adjoint_matching_epochs'),('time_anchor','per_time_anchor_target'),
                     ('bc_variance','bc_loss_weight'),('dual_groups4x','num_samples_for_dual_loss')]:
        cfg,_=configuration(plan,tmp_path,'mechanism_pilot',name,0)
        cfg=OmegaConf.to_container(cfg,resolve=True)
        assert [k for k in keys if cfg[k]!=baseline[k]]==[key]


def test_accounting_includes_phase_probes_and_initial_u_cost():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'reports/aldp_chart_20261007_5a093a95/physical_learning_v1/plan.json').read_text())
    plan['budgets']['geometry']['phase_samples']=128
    plan['candidates']={'base':{'outer_loops':10},'warm':{'outer_loops':10,'initial_adjoint_matching_epochs':500}}
    base=expected_cost(plan,'mechanism_pilot','base')
    warm=expected_cost(plan,'mechanism_pilot','warm')
    assert base['training_and_online_openmm_configurations']==80704
    assert base['diagnostic_openmm_configurations']==3456+20*384
    assert base['total_openmm_configurations']==93888
    assert warm['optimizer_steps']==1760 and warm['u_optimizer_steps']==1360 and warm['h_optimizer_steps']==400
    assert warm['training_and_online_openmm_configurations']==266560


def test_preflight_gate_requires_resolved_training_contract():
    plan={'budgets':{'smoke':{'outer_loops':1,'epochs_per_phase':1,'steps_per_epoch':1,
        'dual_samples':[2,2],'phase_samples':4}},'candidates':{'a':{}},'stability_diagnostics':{}}
    with pytest.raises(RuntimeError,match='Resolved FC/BC/dual contract'):
        verify_stability_report({'diagnostics':{}},plan,'mechanism_smoke','a',{})
