"""Selection respects all A metrics, verification, ties and B disclosure."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_aldp_comparison import midranks,summarize


def cell(name,energy=100.,js=.06,basin=.2,missing=.15,close=.1,ess=.02,verified=True):
    values=dict(phi_psi_tvd=.5,torsion_js_mean=js,physical_energy_w2=energy,
        coarse_basin_tvd=basin,missing_reference_mass=missing)
    return dict(candidate=name,job_id=name,verification=dict(accounting='COMPLETED/0:0',
        log_verified=verified,all_receipt_sha_verified=verified,source_plan_inputs_verified=verified),
        receipt={},report=dict(status='artifact_complete',metrics=dict(a=values,b=deepcopy(values)),
        diagnostics=dict(trained_policy=dict(geometry=dict(fraction_below_reference_q01=close))),
        raw_path_ess=ess,cost=dict(training_and_online_openmm_configurations=100),
        native_cartesian_roundtrip_error=1e-7,
        capacity={},near_cut_fraction_phi_psi=[.1,.1],global_step=800,
        sampling_metadata=dict(source={'scale':1.},target={'energy_mode':'physical'},
            controller={'dim':60},corrector={'dim':60},terminal_cost={'grad_clip_val':1000.})))


THRESHOLDS=dict(phi_psi_tvd=.768310546875,torsion_js_mean=.05,physical_energy_w2=8.,
                coarse_basin_tvd=.15,missing_reference_mass=.1)


def test_low_energy_alone_does_not_win_and_b_cannot_select():
    cells=[cell('energy_only',energy=10.,js=.5,basin=.9,missing=.8,close=.8,ess=.001),
           cell('balanced',energy=100.,js=.055,basin=.16,missing=.11,close=.05,ess=.1)]
    baseline=summarize(cells,THRESHOLDS)
    assert baseline['selected_diagnostic_candidate']=='balanced'
    cells[0]['report']['metrics']['b']={key:0. for key in THRESHOLDS}
    assert summarize(cells,THRESHOLDS)['selected_diagnostic_candidate']=='balanced'
    assert baseline['geometry_status'].startswith('not_admitted')


def test_pending_cells_prevent_premature_selection_and_failures_are_retained():
    cells=[cell('done'),cell('pending',verified=False)]
    result=summarize(cells,THRESHOLDS)
    assert not result['selection_ready'] and result['selected_diagnostic_candidate'] is None
    assert len(result['rows'])==2 and result['rows'][1]['scientific_status']=='not_evaluated'
    cells[1]['terminal_failure_verified']=True
    cells[1]['status']='numerical_failure'
    result=summarize(cells,THRESHOLDS)
    assert result['selection_ready'] and result['selected_diagnostic_candidate']=='done'
    assert result['rows'][1]['retained_failure_or_pending_status']=='numerical_failure'


def test_midranks_are_order_independent_and_preserve_ties():
    assert midranks([1.,1.,3.])=={1.:.5,3.:2.}
    assert midranks([3.,1.,1.])==midranks([1.,1.,3.])


def test_unmatched_source_or_capacity_cannot_be_ranked_as_a_comparison():
    cells=[cell('a'),cell('b')]
    cells[1]['report']['sampling_metadata']['source']['scale']=.5
    with pytest.raises(ValueError,match='sampling contract: source'):
        summarize(cells,THRESHOLDS)
    cells[1]['report']['sampling_metadata']['source']['scale']=1.
    cells[1]['report']['capacity']={'u':42}
    with pytest.raises(ValueError,match='model capacity'):
        summarize(cells,THRESHOLDS)


def test_inverse_failure_is_retained_but_cannot_pass_geometry():
    value=cell('cut_saturation',energy=1.,js=.01,basin=.05,missing=0.)
    value['report']['native_cartesian_roundtrip_error']=None
    value['report']['native_cartesian_roundtrip_diagnostic']={
        'status':'strict_inverse_rejected','error_nm':None,
        'inverse_error':'dihedral outside its fundamental interval'}
    result=summarize([value],THRESHOLDS)
    row=result['rows'][0]
    assert all(row['quality_gate_passes'].values())
    assert not row['pilot_quality_gate_pass'] and not row['coordinate_roundtrip_pass']
    assert row['scientific_status']=='quality_gate_failed'
    assert row['selection_eligible'] and result['selected_diagnostic_candidate']=='cut_saturation'
    assert row['coordinate_roundtrip_diagnostic']['inverse_error'].startswith('dihedral')
