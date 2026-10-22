"""Campaign admission must retain failures and reject altered artifacts."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def monitor_module():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('aldp_monitor', scripts / 'monitor_aldp_chart_campaign.py')
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
    finally:
        sys.path.pop(0)
    return result


def test_geometry_gate_requires_every_seed_and_never_selects_diagnostic(tmp_path):
    monitor = monitor_module()
    (tmp_path / 'preflight').mkdir()
    metrics = dict(phi_psi_tvd=.1, torsion_js_mean=.01, physical_energy_w2=1.,
                   coarse_basin_tvd=.05, missing_reference_mass=0.)
    (tmp_path / 'preflight/calibration_report.json').write_text(json.dumps(
        dict(reference_floor_q95={key: 0. for key in metrics})))
    plan = dict(geometry_gate=dict(minimum_thresholds={key: value * 2 + .01 for key, value in metrics.items()},
                                  floor_multiplier=3))
    rows = [dict(stage='geometry_pilot' if seed == 0 else 'geometry_confirmation',
                 chart=chart, seed=seed, metrics={'a': metrics.copy()}, cost={}, near_cut_fraction_phi_psi=[])
            for chart in ('current', 'low_density', 'high_density') for seed in range(3)]
    rows.append(dict(status='passed', agreement=[{}, {}, {}]))
    rows[0]['metrics']['a']['missing_reference_mass'] = .5
    summary = monitor.summarize_geometry(rows, tmp_path, plan)
    assert summary['selected_chart'] == 'low_density'
    assert len(summary['rows']) == 9
    rows[3]['metrics']['a']['missing_reference_mass'] = .5
    assert monitor.summarize_geometry(rows, tmp_path, plan)['selected_chart'] is None


def test_completed_job_receipt_does_not_accept_modified_output(tmp_path):
    monitor = monitor_module()
    directory = tmp_path / 'preflight'
    directory.mkdir()
    artifact = directory / 'preflight_result.json'
    artifact.write_text(json.dumps(dict(status='passed', agreement=[{}, {}, {}])))
    log = tmp_path / 'worker.out'
    log.write_text('postflight_verified=true\n')
    row = dict(stage='preflight', source_sha256='source', plan_sha256='plan', log_path=str(log))
    receipt = dict(schema='aldp_chart_receipt_v1', postflight_verified=True,
                   inputs=dict(source_sha256='source', plan_sha256='plan'),
                   files={artifact.name: monitor.sha256(artifact)})
    (directory / 'receipt.json').write_text(json.dumps(receipt))
    assert monitor.verify_receipt(directory, row)['status'] == 'passed'
    artifact.write_text('{}')
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        monitor.verify_receipt(directory, row)


def test_final_chart_rejection_keeps_all_raw_outputs_and_receipt(monkeypatch,tmp_path):
    """Exercise train_case's final-report path with the real strict chart, no MD."""
    pytest.importorskip("mdtraj", reason="Requires the molecular extra: uv sync --extra molecular")
    from types import SimpleNamespace
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import tr_asbs.experiment.train_gamma_asbs_iw as trainer
    import tr_asbs.process.chart_source as source_module
    import tr_asbs.utils.md_utils as md
    from tr_asbs.potential.internal_coordinate_chart import InternalCoordinateChart
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_aldp_chart_campaign import train_case,sha256

    transform=InternalCoordinateChart.from_state(dict(
        means=torch.tensor([2.,1.2,.4]),scales=torch.tensor([.3,.2,.6]),
        kinds=torch.tensor([0,1,2])))
    y=torch.tensor([[0.,0.,-1000.],[0.,0.,1000.],[0.,0.,0.]])
    cartesian=transform(y)[0]
    with pytest.raises(ValueError,match='dihedral'):
        transform.inverse(cartesian,canonical=True)
    counters=SimpleNamespace(openmm_configurations_evaluated=0,openmm_batch_calls=0)
    potential=SimpleNamespace(chart=transform,aldp_potential=counters,
        native_to_cartesian=lambda value:transform(value)[0],
        cartesian_to_native=lambda value:transform.inverse(value,canonical=True),
        eval_x=lambda _:torch.zeros(3),log_prob=lambda value:torch.zeros(len(value)))
    module=SimpleNamespace(potential=potential,training_step=lambda *_:True,
        fit=lambda:None,global_step=2,model_capacity={},sampling_metadata=lambda:{},
        source=SimpleNamespace(sample=lambda _:y,log_prob=potential.log_prob),
        integrator=SimpleNamespace(run_with_logweights=lambda *_args,**_kwargs:
            {'data_1':y,'log_Ph_Pref':torch.zeros(3),
             'log_Pref_Pu_det':torch.zeros(3)}))
    bank=dict(x=torch.zeros(3,3),phi_psi=torch.zeros(3,2),energies=torch.zeros(3))
    monkeypatch.setattr(trainer,'GammaASBSModule',lambda _:module)
    monkeypatch.setattr(source_module,'load_calibration_bundle',lambda _:
        {'banks':{'a':bank,'b':bank},'phi_psi_native_indices':[2,2]})
    monkeypatch.setattr(md,'compute_all_dihedrals',lambda *_:torch.zeros(3,2))
    monkeypatch.setattr(md,'extract_phi_psi',lambda value:value)
    calibration=tmp_path/'calibration.pt';calibration.write_bytes(b'bounded fake bank')
    cfg=OmegaConf.create(dict(eval_samples_final=3,potential=dict(calibration_bundle_path=str(calibration))))
    output=tmp_path/'run'
    train_case({'max_openmm_per_geometry_run':100},tmp_path,'mechanism_pilot','current',0,
        'source',config_and_output=(cfg,output))
    report=json.loads((output/'result.json').read_text())
    receipt=json.loads((output/'receipt.json').read_text())
    assert report['global_step']==2 and report['status']=='artifact_complete'
    assert report['native_cartesian_roundtrip_error'] is None
    assert report['native_cartesian_roundtrip_diagnostic']['status']=='strict_inverse_rejected'
    np.testing.assert_array_equal(np.load(output/'raw_y.npy'),y.numpy())
    np.testing.assert_array_equal(np.load(output/'raw_cartesian.npy'),cartesian.numpy())
    for name in ('raw_y.npy','raw_cartesian.npy','raw_phi_psi.npy','raw_physical_energy.npy'):
        assert name in receipt['files'] and receipt['files'][name]==sha256(output/name)
