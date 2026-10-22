#!/usr/bin/env python3
"""Remote ALDP geometry campaign using the shared GammaASBSModule pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ('current', 'low_density', 'high_density')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_source():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError('Source SHA mismatch: ' + relative)
    return sha256(ROOT / 'source_manifest.json')


def receipt(directory, identity, inputs):
    paths = sorted(path for path in directory.rglob('*')
                   if path.is_file() and path.name != 'receipt.json'
                   and not path.name.endswith('.tmp'))
    write_json(directory / 'receipt.json', dict(schema='aldp_chart_receipt_v1',
        identity=identity, inputs=inputs,
        files={str(path.relative_to(directory)): sha256(path) for path in paths},
        postflight_verified=True))


def periodic_delta(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def select_cut_offsets(z, chart, indices):
    offsets = {key: torch.zeros_like(chart.means) for key in CHARTS}
    audit = {}
    edges = np.linspace(-np.pi, np.pi, 73)
    centers = (edges[1:] + edges[:-1]) / 2
    for index in indices:
        angles = (z[:, index] * chart.scales[index] + chart.means[index]).cpu().numpy()
        counts, _ = np.histogram(periodic_delta(angles, 0), bins=edges)
        smooth = sum(weight * np.roll(counts, shift)
                     for shift, weight in zip(range(-2, 3), (1, 2, 3, 2, 1)))
        for key, position in [('low_density', int(np.argmin(smooth))),
                              ('high_density', int(np.argmax(smooth)))]:
            center = centers[position] - np.pi
            offsets[key][index] = float(periodic_delta(center, float(chart.means[index])))
        audit[str(index)] = dict(histogram=counts.tolist(),
            current_cut=float(periodic_delta(float(chart.means[index]) + np.pi, 0)),
            low_density_cut=float(centers[np.argmin(smooth)]),
            high_density_cut=float(centers[np.argmax(smooth)]))
    return offsets, audit


def distribution_metrics(angles, energies, reference_angles, reference_energies):
    # These fixed grids are identical for every chart, seed and budget.
    edges = np.linspace(-np.pi, np.pi, 33)
    def histogram(values, bins):
        h, _ = np.histogramdd(values, bins=[bins, bins])
        return h / max(1, h.sum())
    p, q = histogram(angles, edges), histogram(reference_angles, edges)
    coarse = np.linspace(-np.pi, np.pi, 5)
    pc, qc = histogram(angles, coarse), histogram(reference_angles, coarse)
    js = []
    for dimension in (0, 1):
        ph, _ = np.histogram(angles[:, dimension], bins=edges)
        qh, _ = np.histogram(reference_angles[:, dimension], bins=edges)
        ph, qh = ph / ph.sum(), qh / qh.sum()
        m = (ph + qh) / 2
        def kl(a):
            return float(np.sum(a[a > 0] * np.log(a[a > 0] / m[a > 0])))
        js.append((kl(ph) + kl(qh)) / 2)
    levels = (np.arange(512) + .5) / 512
    ew2 = np.sqrt(np.mean((np.quantile(energies, levels)
                          - np.quantile(reference_energies, levels))**2))
    return dict(phi_psi_tvd=float(np.abs(p - q).sum() / 2),
        torsion_js_mean=float(np.mean(js)), physical_energy_w2=float(ew2),
        coarse_basin_tvd=float(np.abs(pc - qc).sum() / 2),
        missing_reference_mass=float(qc[(qc >= .02) & (pc < .25 * qc)].sum()),
        generated_coarse_occupancy=pc.tolist(), reference_coarse_occupancy=qc.tolist())


def configuration(plan, campaign, stage, chart, seed):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    with initialize_config_dir(str(ROOT / 'src/tr_asbs/config'), version_base=None):
        cfg = compose(config_name='train', overrides=['experiment=ad_ic_gamma_asbs'])
    budget = plan['budgets']['smoke' if stage == 'smoke' else 'geometry']
    output = campaign / stage / f'{chart}_seed{seed}'
    with open_dict(cfg):
        cfg.name, cfg.seed, cfg.root = output.name, seed, str(output)
        cfg.fabric = dict(_target_='lightning.fabric.Fabric', accelerator='cuda',
            devices=1, strategy='auto', num_nodes=1,
            loggers=dict(_target_='lightning.pytorch.loggers.CSVLogger',
                         save_dir=str(output), name='csv'))
        cfg.checkpoint_directory, cfg.checkpoint_group_id = str(output / 'checkpoints'), 'fixed'
        cfg.sample_directory = str(output / 'samples')
        cfg.monitor_values, cfg.monitor_modes = [], {}
        cfg.compile, cfg.resume, cfg.visualize, cfg.progress_bar = False, False, False, False
        cfg.ema_decay = None
        cfg.log_buffer_energy = cfg.log_buffer_grad_norm = cfg.save_samples = False
        cfg.corrector_matching_mode, cfg.corrector_gamma_mode = 'full', 'none'
        cfg.clear_adjoint_buffer_per_outer_loop = True
        cfg.fc_loss_mode, cfg.bc_loss_weight = 'cross_entropy', 'legacy'
        cfg.reference_preset = None
        cfg.controller.time_coordinates = cfg.corrector.time_coordinates = 'physical'
        cfg.max_outer_loops = budget['outer_loops']
        cfg.adjoint_matching_epochs = cfg.corrector_matching_epochs = budget['epochs_per_phase']
        cfg.steps_per_epoch = budget['steps_per_epoch']
        cfg.train_batch_size = cfg.inference_batch_size = budget['batch_size']
        cfg.initial_buffer_samples = cfg.buffer_samples_per_epoch = budget['batch_size'] * 2
        cfg.adjoint_buffer.max_size = cfg.corrector_buffer.max_size = budget['batch_size'] * 8
        cfg.num_samples_for_dual_loss = budget['dual_samples']
        cfg.eval_samples = budget['online_samples']
        cfg.eval_samples_final = budget['final_samples']
        cfg.log_detail_interval_epoch = budget['epochs_per_phase']
        cfg.log_loss_interval_step = 10
        cfg.integrator.timesteps.steps = budget['integration_steps']
        cfg.potential.ref_samples_path = plan['data_path']
        cfg.potential.transform_data_path = plan['data_path']
        for role, selection in [('transform', plan['calibration']), ('reference', plan['bank_a'])]:
            for key, value in selection.items():
                cfg.potential[f'{role}_{key}'] = value
        cfg.potential.coordinate_chart = 'euclidean'
        cfg.potential.calibration_bundle_path = str(campaign / 'preflight/calibration.pt')
        cfg.potential.chart_key = chart
        cfg.source = dict(_target_='tr_asbs.process.chart_source.ChartMappedGaussianPrior',
            dim=60, scale=1., calibration_bundle_path=cfg.potential.calibration_bundle_path,
            chart_key=chart)
    return cfg, output


def prepare(plan, campaign, source_sha):
    from tr_asbs.potential.alanine_dipeptide_energy import AlanineDipeptideEnergyIC
    from tr_asbs.fab.trajectory import load_trajectory_frames
    from tr_asbs.utils.md_utils import compute_all_dihedrals, extract_phi_psi
    directory = campaign / 'preflight'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'calibration.pt').exists():
        raise RuntimeError('Calibration already exists; never overwrite a frozen input')
    data_sha = sha256(plan['data_path'])
    if data_sha != plan['data_sha256']:
        raise RuntimeError('ALDP input data SHA mismatch')
    args = dict(dim=60, ref_samples_path=plan['data_path'], transform_data_path=plan['data_path'],
                coordinate_chart='euclidean', device='cpu')
    args.update({f'transform_{key}': value for key, value in plan['calibration'].items()})
    args.update({f'reference_{key}': value for key, value in plan['bank_a'].items()})
    potential = AlanineDipeptideEnergyIC(**args)
    complete = potential.aldp_potential.p.coordinate_transform.transform
    # Full-IC entries corresponding to [4,6,8,14] (phi), [16,14,8,6] (psi).
    indices = [int(torch.nonzero(complete.permute[9:] == atom * 3 + 2)[0]) + 3
               for atom in (4, 16)]
    calibration_z = potential.aldp_potential.x_to_z(potential.aldp_potential.p.transform_data.float())
    offsets, audit = select_cut_offsets(calibration_z, potential.chart, indices)
    banks = {}
    for key, selection in [('a', plan['bank_a']), ('b', plan['bank_b'])]:
        traj = load_trajectory_frames(plan['data_path'], **selection)
        x = torch.from_numpy(traj.xyz.reshape(-1, 66)).float()
        angles = extract_phi_psi(compute_all_dihedrals(traj, 22, 3))
        energies = potential.eval_x(x).detach().reshape(-1)
        if not torch.isfinite(energies).all():
            raise RuntimeError('Nonfinite MD-bank physical energy')
        banks[key] = dict(x=x, phi_psi=angles, energies=energies)
    base_state = {key: value.cpu() for key, value in potential.chart.state_dict().items()}
    transform = {key: value.cpu() for key, value in potential.aldp_potential.p.coordinate_transform.state_dict().items()}
    bundle = dict(schema='aldp_chart_calibration_v1', coordinate_transform=transform,
        base_chart=base_state, chart_offsets=offsets, phi_psi_native_indices=indices,
        banks=banks, system=potential.aldp_potential.system_metadata,
        provenance=dict(data_sha256=data_sha, calibration=plan['calibration'],
                        bank_a=plan['bank_a'], bank_b=plan['bank_b']))
    torch.save(bundle, directory / 'calibration.pt')
    # A/B floors at the final generated sample count, using frozen subsamples.
    count = plan['budgets']['geometry']['final_samples']
    floors = []
    for seed in range(8):
        rng = np.random.default_rng(11011 + seed)
        ia, ib = [rng.choice(len(banks[key]['x']), count, replace=False) for key in ('a', 'b')]
        floors.append(distribution_metrics(banks['a']['phi_psi'][ia].numpy(),
            banks['a']['energies'][ia].numpy(), banks['b']['phi_psi'][ib].numpy(),
            banks['b']['energies'][ib].numpy()))
    numeric = ('phi_psi_tvd', 'torsion_js_mean', 'physical_energy_w2',
               'coarse_basin_tvd', 'missing_reference_mass')
    floor_q95 = {key: float(np.quantile([row[key] for row in floors], .95)) for key in numeric}
    write_json(directory / 'calibration_report.json', dict(cut_audit=audit,
        phi_psi_native_indices=indices, reference_floor_q95=floor_q95,
        floor_status='finite_MD_bank_comparison_not_reference_convergence',
        setup_openmm_configurations=potential.aldp_potential.openmm_configurations_evaluated,
        source_sha256=source_sha, data_sha256=data_sha))
    del potential


@torch.no_grad()
def cartesian_roundtrip_diagnostic(potential, y, cartesian):
    """Retain strict inverse failures as diagnostics without changing any sample.

    A finite float32 tanh can land exactly on a chart cut. Its strict inverse
    correctly rejects that coordinate; reporting this must not discard the
    completed run or its raw bad samples. No energy call, clamp or path wrap.
    """
    if not torch.isfinite(cartesian).all():
        raise RuntimeError('Nonfinite raw Cartesian samples')
    report = dict(schema='aldp_cartesian_roundtrip_v1', sample_count=len(y),
        tolerance_nm=2e-5, error_nm=None, additional_openmm_configurations=0,
        samples_modified=False, samples_filtered=False)
    chart = getattr(potential, 'chart', None)
    if chart is not None:
        z, _ = chart(y)
        at_boundary = torch.zeros_like(z, dtype=torch.bool)
        for kind in (0, 1, 2):
            indices = chart.kinds == kind
            if kind == 0:
                value = z[..., indices]+chart.means[indices]/chart.scales[indices]
                at_boundary[..., indices] = value <= 0
            elif kind == 1:
                value = (z[..., indices]*chart.scales[indices]+chart.means[indices])/math.pi
                at_boundary[..., indices] = (value <= 0) | (value >= 1)
            else:
                value = (z[..., indices]*chart.scales[indices]-chart.offsets[indices])/math.pi
                at_boundary[..., indices] = value.abs() >= 1
        report['forward_chart_boundary_samples'] = {
            name: int(at_boundary[..., chart.kinds == kind].any(-1).sum())
            for kind, name in enumerate(('bond', 'angle', 'dihedral'))}
        report['forward_chart_boundary_sample_count'] = int(at_boundary.any(-1).sum())
    try:
        reconstructed = potential.cartesian_to_native(cartesian)
    except ValueError as exc:
        report.update(status='strict_inverse_rejected', inverse_error=str(exc))
        return report
    error = float((potential.native_to_cartesian(reconstructed)-cartesian).abs().max())
    if not math.isfinite(error):
        report.update(status='nonfinite_reconstruction')
    else:
        report.update(status='passed' if error <= report['tolerance_nm'] else 'cartesian_mismatch',
                      error_nm=error)
    return report


def train_case(plan, campaign, stage, chart, seed, source_sha, *,
               config_and_output=None, observer=None, provenance=None,
               module_setup=None):
    from omegaconf import OmegaConf
    from tr_asbs.experiment.train_gamma_asbs_iw import GammaASBSModule
    from tr_asbs.process.chart_source import load_calibration_bundle
    from tr_asbs.utils.md_utils import compute_all_dihedrals, extract_phi_psi
    from tr_asbs.utils.eval_logweight import compute_raw_path_ess
    cfg, output = (configuration(plan, campaign, stage, chart, seed)
                   if config_and_output is None else config_and_output)
    if (output / 'result.json').exists():
        raise RuntimeError('Run exists; use a new attempt directory for a retry')
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    bundle = load_calibration_bundle(cfg.potential.calibration_bundle_path)
    started = time.perf_counter()
    module = GammaASBSModule(cfg)
    counters = module.potential.aldp_potential
    failures, online = [], []
    diagnostic_cost = 0

    def observe(event, payload=None):
        nonlocal diagnostic_cost
        if observer is None:
            return None
        backend = getattr(module, 'jax_inner_backend', None)
        if backend is not None:
            backend.flush()
        before = counters.openmm_configurations_evaluated
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(module.fabric.device).clone()
        value = observer(module, event, payload)
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
                cuda_rng, torch.cuda.get_rng_state(module.fabric.device)):
            raise RuntimeError('Diagnostic observer changed the training RNG')
        diagnostic_cost += counters.openmm_configurations_evaluated - before
        return value

    if observer is not None:
        module.campaign_observe = observe
    observe('initial')

    @torch.no_grad()
    def log_online(step, visualize=False, eval_samples=None):
        count = eval_samples or cfg.eval_samples
        result = module.integrator.run_with_logweights(module.prev_integrator,
            module.source.log_prob, module.potential.log_prob, module.source.sample((count,)),
            center_every_step=False, return_trajectory=False, progress_bar=False)
        values = dict(epoch=int(module.current_epoch), gamma=float(module.gamma),
            raw_path_ess=float(compute_raw_path_ess(result['log_Ph_Pref'])),
            achieved_forward_kl=float(result['log_Pref_Pu_det'].mean()),
            openmm_configurations=int(counters.openmm_configurations_evaluated))
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise RuntimeError('Nonfinite online evaluation')
        online.append(values)
        module.fabric.log_dict(values, step=module.global_step)
        observe('online', values)
        return dict(eval_step=step, epoch=module.current_epoch)
    module.log_details, module.log_final = log_online, lambda: None
    # Opt-in backends must wrap the resolved online logger and remain inside
    # the shared failure/budget guard. Default campaigns use the same steps.
    if module_setup is not None:
        module_setup(module)
    original_step = module.training_step
    def checked_step(batch, task):
        success = original_step(batch, task)
        if not success:
            failures.append(dict(task=task, epoch=module.current_epoch, step=module.global_step))
        if counters.openmm_configurations_evaluated > plan['max_openmm_per_geometry_run']:
            raise RuntimeError('Frozen OpenMM configuration budget exceeded')
        return success
    module.training_step = checked_step
    module.fit()
    training_cost = counters.openmm_configurations_evaluated - diagnostic_cost
    if failures:
        raise RuntimeError(f'Numerical training failures retained: {failures}')
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    x0 = module.source.sample((cfg.eval_samples_final,))
    with torch.no_grad():
        result = module.integrator.run_with_logweights(module.integrator,
            module.source.log_prob, module.potential.log_prob, x0,
            center_every_step=False, return_trajectory=False, progress_bar=False)
        y = result['data_1']
        if not torch.isfinite(y).all() or not torch.isfinite(result['log_Ph_Pref']).all():
            raise RuntimeError('Nonfinite raw generated samples/log weights')
        cartesian = module.potential.native_to_cartesian(y)
        physical = module.potential.eval_x(cartesian).detach().cpu().reshape(-1)
        angles = extract_phi_psi(compute_all_dihedrals(cartesian, 22, 3)).numpy()
    np.save(output / 'raw_y.npy', y.detach().cpu().numpy())
    np.save(output / 'raw_cartesian.npy', cartesian.detach().cpu().numpy())
    np.save(output / 'raw_phi_psi.npy', angles)
    np.save(output / 'raw_physical_energy.npy', physical.numpy())
    roundtrip = cartesian_roundtrip_diagnostic(module.potential, y, cartesian)
    final_evaluation_cost = counters.openmm_configurations_evaluated - training_cost - diagnostic_cost
    diagnostics = observe('final', dict(y=y, cartesian=cartesian, physical_energy=physical,
        log_Ph_Pref=result['log_Ph_Pref'], log_Pref_Pu_det=result['log_Pref_Pu_det']))
    if counters.openmm_configurations_evaluated > plan['max_openmm_per_geometry_run']:
        raise RuntimeError('Total OpenMM configuration budget exceeded')
    metrics = {}
    for key in ('a', 'b'):
        bank = bundle['banks'][key]
        indices = np.random.default_rng(22022).choice(len(bank['x']), len(y), replace=False)
        metrics[key] = distribution_metrics(angles, physical.numpy(), bank['phi_psi'][indices].numpy(),
                                             bank['energies'][indices].numpy())
    z, _ = module.potential.chart(y)
    chart_indices = bundle['phi_psi_native_indices']
    delta = ((z[:, chart_indices] * module.potential.chart.scales[chart_indices]
              - module.potential.chart.offsets[chart_indices]).detach().cpu().numpy())
    cut_visits = np.mean(np.pi - np.abs(delta) < .15, axis=0).tolist()
    write_json(output / 'result.json', dict(schema='aldp_chart_result_v1', stage=stage,
        chart=chart, seed=seed, status='artifact_complete', scientific_status='not_evaluated',
        source_sha256=source_sha, calibration_sha256=sha256(cfg.potential.calibration_bundle_path),
        config_sha256=sha256(output / 'config.yaml'), metrics=metrics,
        native_cartesian_roundtrip_error=roundtrip['error_nm'],
        native_cartesian_roundtrip_diagnostic=roundtrip,
        raw_path_ess=float(compute_raw_path_ess(result['log_Ph_Pref'])),
        near_cut_fraction_phi_psi=cut_visits, native_abs_q99=float(torch.quantile(y.abs(), .99)),
        rejected_training_steps=failures, global_step=module.global_step,
        capacity=module.model_capacity, online=online,
        provenance=provenance, diagnostics=diagnostics,
        cost=dict(training_and_online_openmm_configurations=training_cost,
            final_evaluation_openmm_configurations=final_evaluation_cost,
            diagnostic_openmm_configurations=diagnostic_cost,
            total_openmm_configurations=counters.openmm_configurations_evaluated,
            openmm_batch_calls=counters.openmm_batch_calls,
            wall_seconds=time.perf_counter() - started),
        sampling_metadata=module.sampling_metadata()))
    receipt(output, dict(stage=stage, chart=chart, seed=seed),
            dict(source_sha256=source_sha, calibration_sha256=sha256(cfg.potential.calibration_bundle_path),
                 provenance=provenance))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'geometry_pilot', 'geometry_confirmation'), required=True)
    parser.add_argument('--chart', choices=CHARTS, default='current')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    plan = json.loads((args.campaign / 'plan.json').read_text())
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Configured remote GPU unavailable')
    source_sha = verify_source()
    try:
        if args.stage == 'preflight':
            prepare(plan, args.campaign, source_sha)
            for chart in CHARTS:
                train_case(plan, args.campaign, 'smoke', chart, 0, source_sha)
            # Physical source and chart/Jacobian agreement across all three cuts.
            from hydra.utils import instantiate
            previous_x = None
            agreement = []
            for chart in CHARTS:
                cfg, _ = configuration(plan, args.campaign, 'smoke', chart, 0)
                potential, source = instantiate(cfg.potential), instantiate(cfg.source)
                torch.manual_seed(813)
                y = source.sample((8,))
                x = potential.native_to_cartesian(y)
                z, logjac = potential.chart(y)
                torch.testing.assert_close(potential.eval(y) + logjac,
                    -potential.aldp_potential._log_prob(z), atol=.002, rtol=1e-5)
                if previous_x is not None:
                    torch.testing.assert_close(x, previous_x, atol=2e-5, rtol=2e-5)
                previous_x = x
                agreement.append(dict(chart=chart, gradient_finite=bool(torch.isfinite(potential.grad_E(y)).all())))
                if not agreement[-1]['gradient_finite']:
                    raise RuntimeError('Chart target gradient is nonfinite')
                del potential, source
            directory = args.campaign / 'preflight'
            write_json(directory / 'preflight_result.json', dict(source_sha256=source_sha,
                agreement=agreement, status='passed', scientific_status='not_evaluated'))
            receipt(directory, dict(stage='preflight'), dict(source_sha256=source_sha,
                data_sha256=plan['data_sha256'], plan_sha256=sha256(args.campaign / 'plan.json')))
        else:
            train_case(plan, args.campaign, args.stage, args.chart, args.seed, source_sha)
        print('postflight_verified=true', flush=True)
    except Exception as error:
        directory = args.campaign / args.stage / f'{args.chart}_seed{args.seed}'
        write_json(directory / 'failure.json', dict(status='failed', error=repr(error),
            traceback=traceback.format_exc(), source_sha256=source_sha))
        raise


if __name__ == '__main__':
    main()
