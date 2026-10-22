"""Prefix selection gates and A/B/test separation, without simulations."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_aldp_inner_curve import screen


def fixture():
    contract=json.loads((Path(__file__).resolve().parent/
        'fixtures/aldp_inner_curve_contract.json').read_text())
    boundaries=[]
    for phase in ('u','h'):
        for epoch in [0]+contract['milestones'][phase]:
            metrics=dict(torsion_js_mean=.02,coarse_basin_tvd=.05,
                missing_reference_mass=.02,physical_energy_w2=7.)
            policy=dict(metrics={'a':metrics,'b':deepcopy(metrics)},
                geometry={'fraction_below_reference_q01':.03},
                near_cut_fraction_phi_psi=[.02,.03],raw_log_path_ratio_variance=1.,
                raw_path_ess_fraction=.5,physical_energy={'count':1024,'quantiles':{'0.99':100.}})
            boundaries.append(dict(phase=phase,phase_epoch=epoch,snapshot=f'{phase}{epoch}.pt',
                validation={'sample_count':256,'heldout_loss':1.},policy=policy))
    report=dict(global_step=300000,rejected_training_steps=[],minimum_inner_duration='not_established',
        diagnostics=dict(curve_contract=contract,inner_boundaries=boundaries))
    return report,contract


def test_earliest_passing_grid_candidate_never_claims_a_minimum():
    report,contract=fixture()
    rows=report['diagnostics']['inner_boundaries']
    rows[1]['validation']['heldout_loss']=1.1
    result=screen(report,contract)
    assert result['phases']['u']['proposed_shorter_epoch']==500
    assert result['phases']['h']['proposed_shorter_epoch']==125
    assert result['minimum_inner_duration']=='not_established' and result['new_jobs_submitted']==0


def test_b_disclosure_cannot_change_a_selection():
    report,contract=fixture();baseline=screen(report,contract)
    for row in report['diagnostics']['inner_boundaries']:
        row['policy']['metrics']['b']['physical_energy_w2']=1e30
        row['policy']['metrics']['b']['torsion_js_mean']=.69
    changed=screen(report,contract)
    for phase in ('u','h'):
        assert changed['phases'][phase]['proposed_shorter_epoch']==baseline['phases'][phase]['proposed_shorter_epoch']


def test_h_matching_ess_and_rn_gate_even_when_forward_metrics_match():
    report,contract=fixture()
    for row in report['diagnostics']['inner_boundaries']:
        if row['phase']=='h' and 0<row['phase_epoch']<1000:
            row['policy']['raw_path_ess_fraction']=.1
    result=screen(report,contract)
    assert result['phases']['h']['proposed_shorter_epoch'] is None
    assert result['phases']['h']['forward_quality_is_invariant_at_fixed_u']


def test_incomplete_or_foreign_contract_is_not_screened():
    report,contract=fixture();report['global_step']=299900
    with pytest.raises(ValueError,match='completed normal'):
        screen(report,contract)
    report,contract=fixture();changed=deepcopy(contract);changed['selection_bank']='b'
    with pytest.raises(ValueError,match='A selection'):
        screen(report,changed)
    report,contract=fixture();report['diagnostics']['inner_boundaries'].pop(1)
    with pytest.raises(ValueError,match='Incomplete'):
        screen(report,contract)
