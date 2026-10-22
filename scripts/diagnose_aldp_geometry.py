#!/usr/bin/env python3
"""Remote postflight diagnosis of source, clashes, energy tails and learning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


def energy_summary(values, reference=None):
    quantiles = (0, .5, .9, .95, .99, .999, 1)
    summary = dict(count=len(values), quantiles={str(q): float(np.quantile(values, q)) for q in quantiles},
                   mean=float(np.mean(values)))
    if reference is not None:
        summary['fraction_above_reference_q99'] = float(np.mean(values > np.quantile(reference, .99)))
    return summary


def nonbonded_pairs(topology):
    adjacency = {i: set() for i in range(22)}
    for a, b in topology.bonds():
        adjacency[a.index].add(b.index)
        adjacency[b.index].add(a.index)
    pairs = []
    for i in range(22):
        nearby, frontier = {i}, {i}
        for _ in range(3):
            frontier = set().union(*(adjacency[j] for j in frontier)) - nearby
            nearby |= frontier
        pairs.extend((i, j) for j in range(i + 1, 22) if j not in nearby)
    return np.asarray(pairs)


def distance_summary(cartesian, pairs, reference_distances):
    xyz = cartesian.reshape(-1, 22, 3)
    distances = np.linalg.norm(xyz[:, pairs[:, 0]] - xyz[:, pairs[:, 1]], axis=-1)
    minimum = distances.min(axis=1)
    worst = np.unravel_index(np.argmin(distances), distances.shape)
    return dict(minimum_nonbonded_distance_nm=energy_summary(minimum, reference_distances),
                fraction_below_reference_q01=float(np.mean(minimum < np.quantile(reference_distances, .01))),
                smallest_distance_nm=float(distances[worst]),
                closest_pair_atom_indices=pairs[worst[1]].tolist(), sample_index=int(worst[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'scripts'))
    from run_aldp_chart_campaign import configuration, sha256, write_json, receipt, verify_source
    from monitor_aldp_chart_campaign import verify_receipt
    from hydra.utils import instantiate
    from tr_asbs.process.chart_source import load_calibration_bundle
    from openmmtools.testsystems import AlanineDipeptideImplicit

    torch.set_num_threads(1)
    source_sha = verify_source()
    campaign = args.campaign.resolve()
    output = campaign / 'diagnostics'
    if (output / 'receipt.json').exists():
        raise RuntimeError('Diagnostic result exists; preserve it rather than overwrite')
    plan = json.loads((campaign / 'plan.json').read_text())
    with (campaign / 'submitted_jobs.tsv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream, delimiter='\t') if row['stage'].startswith('geometry')]
    if len(rows) != 9:
        raise RuntimeError('Require all 9 planned geometry rows')
    reports = [(row, verify_receipt(Path(row['run_directory']), row)) for row in rows]
    bundle = load_calibration_bundle(campaign / 'preflight/calibration.pt')
    pairs = nonbonded_pairs(AlanineDipeptideImplicit(constraints=None).topology)
    reference = bundle['banks']['a']
    ref_xyz = reference['x'].numpy().reshape(-1, 22, 3)
    ref_distances = np.linalg.norm(ref_xyz[:, pairs[:, 0]] - ref_xyz[:, pairs[:, 1]], axis=-1).min(axis=1)
    ref_energy = reference['energies'].numpy()
    cfg, _ = configuration(plan, campaign, 'geometry_pilot', 'current', 0)
    potential, prior = instantiate(cfg.potential, device='cpu'), instantiate(cfg.source, device='cpu')
    torch.manual_seed(12455)
    native = prior.sample((1024,))
    with torch.no_grad():
        source_xyz = potential.native_to_cartesian(native).numpy()
        source_energy = potential.eval_x(torch.from_numpy(source_xyz)).reshape(-1).numpy()
    gradient = potential.grad_E(native[:128]).double().norm(dim=-1).numpy()
    source = dict(physical_energy=energy_summary(source_energy, ref_energy),
        geometry=distance_summary(source_xyz, pairs, ref_distances),
        true_native_gradient_norm=energy_summary(gradient),
        fraction_gradient_above_terminal_clip100=float(np.mean(gradient > 100)))
    cells, input_hashes = [], {}
    for row, report in reports:
        directory = Path(row['run_directory'])
        xyz = np.load(directory / 'raw_cartesian.npy')
        energy = np.load(directory / 'raw_physical_energy.npy')
        gammas = [item['gamma'] for item in report['online']]
        cells.append(dict(chart=row['chart'], seed=int(row['seed']),
            physical_energy=energy_summary(energy, ref_energy),
            geometry=distance_summary(xyz, pairs, ref_distances),
            gamma_min=float(min(gammas)), gamma_max=float(max(gammas)),
            optimizer_steps=report['global_step'],
            raw_path_ess=report['raw_path_ess'], metrics=report['metrics']['a']))
        for name in ('receipt.json', 'result.json', 'raw_cartesian.npy', 'raw_physical_energy.npy'):
            input_hashes[str((directory / name).relative_to(campaign))] = sha256(directory / name)
    inputs = dict(source_sha256=source_sha, calibration_sha256=sha256(campaign / 'preflight/calibration.pt'),
                  diagnostic_script_sha256=sha256(__file__), geometry_inputs=input_hashes)
    write_json(output / 'diagnostic_report.json', dict(schema='aldp_geometry_diagnosis_v1',
        status='artifact_complete', scientific_status='convergence_not_evaluated',
        reference=energy_summary(ref_energy, ref_energy), source=source, cells=cells,
        nonbonded_definition='exclude topology paths of length <=3',
        units=dict(energy='kT regularized OpenMM energy', distances='nm'),
        diagnostic_source_seed=12455, diagnostic_source_is_new_bank=True,
        diagnostic_openmm_configurations=potential.aldp_potential.openmm_configurations_evaluated,
        diagnostic_cost_separate_from_training=True, inputs=inputs))
    receipt(output, dict(stage='geometry_diagnosis'), inputs)
    print('postflight_verified=true', flush=True)


if __name__ == '__main__':
    main()
