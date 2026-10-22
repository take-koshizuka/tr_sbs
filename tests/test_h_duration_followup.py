"""Guard registered selection and cached-control/BC isolation without training."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from monitor_aldp_h_duration_campaign import choose_h_candidate, compare_cached_reference
from test_inner_curve_screening import fixture


def screening():
    return dict(decision_bank='a', test='closed', phases={'h': {'comparisons': [
        dict(phase_epoch=epoch, prefix_gates_passed=epoch != 250)
        for epoch in (125, 250, 500, 750, 1000)]}})


def test_next_trial_skips_failed_confirmation_and_failed_prefix():
    attempts = [dict(h_epochs=125, conditional_gates_passed=False)]
    assert choose_h_candidate(screening(), attempts) == 500
    attempts.append(dict(h_epochs=500, conditional_gates_passed=False))
    assert choose_h_candidate(screening(), attempts) == 750
    attempts.append(dict(h_epochs=750, conditional_gates_passed=False))
    assert choose_h_candidate(screening(), attempts) is None


def test_success_duplicate_unregistered_and_test_open_forbid_more_trials():
    with pytest.raises(ValueError, match='already passed'):
        choose_h_candidate(screening(), [dict(h_epochs=125, conditional_gates_passed=True)])
    bad = [dict(h_epochs=125, conditional_gates_passed=False)] * 2
    with pytest.raises(ValueError, match='Duplicate'):
        choose_h_candidate(screening(), bad)
    bad = screening(); bad['phases']['h']['comparisons'][2]['phase_epoch'] = 375
    with pytest.raises(ValueError, match='registered h grid'):
        choose_h_candidate(bad, [])
    bad = screening(); bad['test'] = 'open'
    with pytest.raises(ValueError, match='closed test'):
        choose_h_candidate(bad, [])


def reports():
    base, contract = fixture()
    policy = base['diagnostics']['inner_boundaries'][-1]['policy']
    loss = dict(heldout_loss=1., batch_path='/frozen/heldout_h.pt', sample_count=256,
                bank='identical normal-outer1 frozen endpoints/bridge draws')
    normal = dict(source_sha256='old', diagnostics=dict(first_outer_complete=policy, trained_policy=policy,
                  first_outer_validation=deepcopy(loss), next_outer_validation=deepcopy(loss)),
                  cost={'total_openmm_configurations': 260224}, new_optimizer_steps=300000,
                  cached_global_step=300000, cached_reference_paid_openmm=285376)
    fresh = deepcopy(normal); fresh['source_sha256'] = 'new'
    plan = dict(source_manifest_sha256='new', comparison_h_validation_path='/frozen/heldout_h.pt',
                cases={'h_short': {'h_epochs': 500}}, normal_cache={'case': {}, 'row': {'job_id': '5212231'}},
                curve_contract=contract, cached_reference_paid_openmm=285376,
                prior_study_paid_openmm_configurations=1004544)
    return fresh, normal, plan


def test_cached_control_does_not_become_a_dispatch_case_or_use_b_for_selection():
    pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
    fresh, normal, plan = reports()
    before = deepcopy(plan)
    result = compare_cached_reference(fresh, normal, plan)
    assert result['candidates']['h_short']['conditional_gates_passed']
    assert result['cached_normal_job_id'] == '5212231'
    assert plan == before and set(plan['cases']) == {'h_short'}
    fresh['diagnostics']['trained_policy']['metrics']['b']['physical_energy_w2'] = 1e30
    result = compare_cached_reference(fresh, normal, plan)
    assert result['candidates']['h_short']['conditional_gates_passed']
    assert result['no_paired_confirmation'] and result['pair_status'] != 'ready_for_one_paired_confirmation'


def test_foreign_bc_or_source_cannot_enter_cached_comparison():
    pytest.importorskip("jax", reason="Requires the jax extra: uv sync --extra jax")
    fresh, normal, plan = reports()
    fresh['diagnostics']['next_outer_validation']['batch_path'] = '/own_u/heldout_h.pt'
    with pytest.raises(ValueError, match='fixed normal BC bank'):
        compare_cached_reference(fresh, normal, plan)
    fresh, normal, plan = reports(); fresh['source_sha256'] = 'foreign'
    with pytest.raises(ValueError, match='source differs'):
        compare_cached_reference(fresh, normal, plan)
