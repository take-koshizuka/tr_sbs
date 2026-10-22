"""Protect the matched comparison and isolate diagnostics from training RNG."""

from pathlib import Path
import sys

import torch
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_aldp_mechanism_campaign import configuration, diagnostic_rng


def test_mechanism_candidates_change_one_scientific_factor(tmp_path):
    import json
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'reports/aldp_chart_20261007_5a093a95/plan.json').read_text())
    plan['upstream_campaign'] = str(tmp_path / 'old')
    plan['candidates'] = dict(
        baseline=dict(source_scale=1., terminal_gradient_clip=100., outer_loops=2),
        source_half=dict(source_scale=.5, terminal_gradient_clip=100., outer_loops=2),
        clip1000=dict(source_scale=1., terminal_gradient_clip=1000., outer_loops=2),
        training5x=dict(source_scale=1., terminal_gradient_clip=100., outer_loops=10))
    baseline, _ = configuration(plan, tmp_path, 'mechanism_pilot', 'baseline', 0)
    baseline = OmegaConf.to_container(baseline, resolve=True)
    scientific_keys = ('potential', 'source', 'controller', 'corrector', 'base_sde', 'integrator',
        'terminal_cost', 'fc_loss_mode', 'bc_loss_weight', 'max_outer_loops', 'steps_per_epoch',
        'adjoint_matching_epochs', 'corrector_matching_epochs', 'trust_region_bound',
        'controller_optimizer', 'corrector_optimizer', 'train_batch_size', 'eval_samples_final')
    for candidate, expected in [('source_half', 'source'), ('clip1000', 'terminal_cost'),
                                 ('training5x', 'max_outer_loops')]:
        cfg, _ = configuration(plan, tmp_path, 'mechanism_pilot', candidate, 0)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        changed = [key for key in scientific_keys if cfg[key] != baseline[key]]
        assert changed == [expected]


def test_diagnostic_sampling_preserves_training_rng_on_error():
    torch.manual_seed(91)
    before = torch.get_rng_state().clone()
    try:
        with diagnostic_rng(torch.device('cpu')):
            torch.manual_seed(12345)
            torch.randn(16)
            raise ValueError('diagnostic failure')
    except ValueError:
        pass
    assert torch.equal(torch.get_rng_state(), before)


def test_physical_mechanism_uses_the_versioned_bank_for_target_and_source(tmp_path):
    import json
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'reports/aldp_chart_20261007_5a093a95/mechanism_v1/plan.json').read_text())
    plan.update(energy_mode='physical', calibration_path='/frozen/physical/calibration.pt')
    cfg, _ = configuration(plan, tmp_path, 'mechanism_pilot', 'source_half', 0)
    assert cfg.potential.energy_mode == 'physical'
    assert cfg.potential.calibration_bundle_path == plan['calibration_path']
    assert cfg.source.calibration_bundle_path == plan['calibration_path']


def test_submission_reconciliation_blocks_repeat_dispatch(tmp_path):
    from monitor_aldp_mechanism_campaign import submit
    row = dict(stage='preflight', candidate='', seed='0')
    assert submit(tmp_path, {}, '/remote/python', [row], 'preflight') is None
    intent = tmp_path / 'submission_intents/preflight_all_seed0.json'
    intent.parent.mkdir()
    intent.write_text('{"status": "dispatching"}')
    with pytest.raises(RuntimeError, match='needs reconciliation'):
        submit(tmp_path, {}, '/remote/python', [], 'preflight')


def test_fresh_physical_control_allows_preflight_without_cached_baseline(tmp_path, monkeypatch, capsys):
    import json
    import monitor_aldp_mechanism_campaign as monitor
    plan = dict(cached_baseline=None, calibration_path='/physical/calibration.pt',
        calibration_sha256='SHA', diagnostic_report_path='/physical/audit.json',
        diagnostic_report_sha256='SHA', candidates={'baseline': {}})
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(monitor, 'verify_source', lambda: 'SHA')
    monkeypatch.setattr(monitor, 'sha256', lambda _: 'SHA')
    monkeypatch.setattr(sys, 'argv', ['monitor', '--campaign', str(tmp_path),
        '--python', '/remote/python', '--advance', '--dry-run'])
    monitor.main()
    output = json.loads(capsys.readouterr().out)
    assert output['new_submissions'][0]['stage'] == 'preflight'
    assert not (tmp_path / 'submitted_jobs.tsv').exists()


def test_cut_candidate_changes_only_the_target_and_source_chart(tmp_path):
    import json
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'reports/aldp_chart_20261007_5a093a95/physical_interaction_v1/plan.json').read_text())
    plan['candidates'] = dict(
        current=dict(source_scale=1., terminal_gradient_clip=1000., outer_loops=10),
        shifted=dict(source_scale=1., terminal_gradient_clip=1000., outer_loops=10,
                     chart_key='low_density'))
    before, _ = configuration(plan, tmp_path, 'mechanism_pilot', 'current', 0)
    after, _ = configuration(plan, tmp_path, 'mechanism_pilot', 'shifted', 0)
    assert before.potential.chart_key == before.source.chart_key == 'current'
    assert after.potential.chart_key == after.source.chart_key == 'low_density'
    a, b = [OmegaConf.to_container(cfg, resolve=True) for cfg in (before, after)]
    for cfg in (a, b):
        for key in ('name', 'root', 'checkpoint_directory', 'sample_directory'):
            cfg.pop(key)
        cfg['fabric']['loggers'].pop('save_dir')
        cfg['potential'].pop('chart_key')
        cfg['source'].pop('chart_key')
    assert a == b


def test_paired_physical_source_audit_preserves_rng_and_rejects_changed_geometry(tmp_path):
    from types import SimpleNamespace
    from torch import nn
    from tr_asbs.potential.internal_coordinate_chart import InternalCoordinateChart
    from tr_asbs.process.chart_source import ChartMappedGaussianPrior
    from run_aldp_mechanism_campaign import paired_physical_source_audit
    chart = InternalCoordinateChart.__new__(InternalCoordinateChart)
    nn.Module.__init__(chart)
    chart._initialize(torch.tensor([2., 1.2, .4], dtype=torch.double),
                      torch.tensor([.3, .2, .6], dtype=torch.double), torch.tensor([0, 1, 2]))
    bundle = tmp_path / 'calibration.pt'
    torch.save(dict(schema='aldp_chart_calibration_v1', base_chart=chart.state_dict(),
        chart_offsets={'current': torch.zeros(3), 'low_density': torch.tensor([0., 0., 1.1])}), bundle)
    source = ChartMappedGaussianPrior(bundle, 'low_density', dim=3)
    module = SimpleNamespace(source=source, fabric=SimpleNamespace(device=torch.device('cpu')),
        cfg=SimpleNamespace(potential=SimpleNamespace(calibration_bundle_path=str(bundle))),
        potential=SimpleNamespace(aldp_potential=SimpleNamespace(z_to_x=chart.canonicalize)))
    torch.manual_seed(12455)
    y = source.sample((16,))
    x = chart.canonicalize(source.chart(y)[0])
    state = torch.get_rng_state().clone()
    audit = paired_physical_source_audit(module, y, x, 'low_density')
    assert audit['status'] == 'passed' and audit['sample_count'] == 16
    assert audit['max_cartesian_error_nm'] < 1e-6
    assert audit['max_physical_log_density_error'] < 1e-6
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(RuntimeError, match='changed the paired physical source'):
        paired_physical_source_audit(module, y, x + .1, 'low_density')
    assert torch.equal(torch.get_rng_state(), state)


def test_shifted_receipt_requires_consistent_chart_and_source_audit(tmp_path):
    import json
    from monitor_aldp_mechanism_campaign import verify_case
    from run_aldp_chart_campaign import receipt, sha256
    (tmp_path / 'config.yaml').write_text('chart: low_density\n')
    provenance = dict(candidate='shifted', plan_sha256='PLAN')
    report = dict(stage='mechanism_smoke', chart='low_density', seed=0,
        provenance=provenance, status='artifact_complete', rejected_training_steps=[],
        config_sha256=sha256(tmp_path / 'config.yaml'), global_step=2,
        cost=dict(training_and_online_openmm_configurations=40,
                  final_evaluation_openmm_configurations=8,
                  diagnostic_openmm_configurations=24, total_openmm_configurations=72),
        metrics={'a': {'phi_psi_tvd': .5}},
        sampling_metadata=dict(target={'chart_key': 'low_density'}, source={'chart_key': 'low_density'}),
        diagnostics={'physical_source_audit': dict(status='passed', chart='low_density',
            reference_chart='current', source_scale=1., sample_count=4,
            additional_openmm_configurations=0, rng_isolated=True,
            max_cartesian_error_nm=1e-7, max_physical_log_density_error=1e-6)})
    plan = dict(_plan_sha256='PLAN', calibration_sha256='BANK',
        candidates={'shifted': dict(chart_key='low_density', source_scale=1., openmm_cap=100000)},
        budgets={'smoke': dict(outer_loops=1, epochs_per_phase=1, steps_per_epoch=1, final_samples=4)},
        geometry_gate={'minimum_thresholds': {'phi_psi_tvd': .3}})
    def save():
        (tmp_path / 'result.json').write_text(json.dumps(report))
        receipt(tmp_path, dict(stage='mechanism_smoke', chart='low_density', seed=0),
            dict(source_sha256='SOURCE', calibration_sha256='BANK', provenance=provenance))
    save()
    assert verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'shifted', 0)['chart'] == 'low_density'
    report['chart'] = 'current'
    save()
    with pytest.raises(RuntimeError, match='Result identity mismatch'):
        verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'shifted', 0)
    report['chart'] = 'low_density'
    report['diagnostics'].pop('physical_source_audit')
    save()
    with pytest.raises(RuntimeError, match='source audit incomplete'):
        verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'shifted', 0)


def test_configurable_learning_prefix_preserves_rng_at_the_requested_step():
    from types import SimpleNamespace
    from run_aldp_mechanism_campaign import MechanismObserver
    observer = MechanismObserver.__new__(MechanismObserver)
    observer.extended, observer.prefix_outer_loops, observer.prefix_epoch = True, 10, 400
    observer.report = {'online': []}
    observer.controls = lambda _: {}
    labels = []
    def policy(_, label):
        labels.append(label)
        torch.randn(16)
        return {'metrics': {'a': {}, 'b': {}}}
    observer.policy = policy
    module = SimpleNamespace(fabric=SimpleNamespace(device=torch.device('cpu')),
        global_step=800, gamma=.1, adjoint_buffer=SimpleNamespace(storage={}))
    before = torch.get_rng_state().clone()
    observer(module, 'online', {'epoch': 80})
    assert not labels and 'after_outer10' not in observer.report
    observer(module, 'online', {'epoch': 400})
    assert labels == ['outer10']
    assert observer.report['after_outer10']['global_step'] == 800
    assert observer.report['after_outer10']['epoch'] == 400
    assert torch.equal(torch.get_rng_state(), before)


def test_learning_prefix_is_required_at_the_exact_step_and_in_the_receipt(tmp_path):
    import json
    from monitor_aldp_mechanism_campaign import verify_case
    from run_aldp_chart_campaign import receipt, sha256
    (tmp_path / 'config.yaml').write_text('learning: extended\n')
    provenance = dict(candidate='extended', plan_sha256='PLAN')
    metrics = {'a': {'phi_psi_tvd': .5}, 'b': {'phi_psi_tvd': .6}}
    report = dict(stage='mechanism_smoke', chart='current', seed=0, provenance=provenance,
        status='artifact_complete', rejected_training_steps=[], config_sha256=sha256(tmp_path / 'config.yaml'),
        global_step=2, cost=dict(training_and_online_openmm_configurations=40,
            final_evaluation_openmm_configurations=8, diagnostic_openmm_configurations=36,
            total_openmm_configurations=84), metrics=metrics, diagnostics={})
    plan = dict(_plan_sha256='PLAN', calibration_sha256='BANK',
        candidates={'extended': dict(smoke_prefix_outer_loops=1, openmm_cap=200000)},
        budgets={'smoke': dict(outer_loops=1, epochs_per_phase=1, steps_per_epoch=1)},
        geometry_gate={'minimum_thresholds': {'phi_psi_tvd': .3}})
    def save():
        (tmp_path / 'result.json').write_text(json.dumps(report))
        receipt(tmp_path, dict(stage='mechanism_smoke', chart='current', seed=0),
            dict(source_sha256='SOURCE', calibration_sha256='BANK', provenance=provenance))
    save()
    with pytest.raises(RuntimeError, match='Prefix diagnostic incomplete'):
        verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'extended', 0)
    report['diagnostics']['after_outer1'] = dict(epoch=2, global_step=2, metrics=metrics)
    save()
    with pytest.raises(RuntimeError, match='Prefix array absent'):
        verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'extended', 0)
    for suffix in ('cartesian', 'physical_energy'):
        (tmp_path / f'outer1_{suffix}.npy').write_bytes(b'prefix array fixture')
    save()
    verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'extended', 0)
    report['diagnostics']['after_outer1']['global_step'] = 1
    save()
    with pytest.raises(RuntimeError, match='wrong training step'):
        verify_case(tmp_path, plan, 'SOURCE', 'mechanism_smoke', 'extended', 0)
