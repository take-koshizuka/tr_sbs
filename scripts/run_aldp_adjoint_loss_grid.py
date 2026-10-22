"""Original TR adjoint-matching loss on the registered noise/solver axes.

This reuses the existing shared trainer, collector, kernels and diagnostics.
The only scientific configuration change is fc_loss_mode=cross_entropy.
"""
from contextlib import contextmanager

from omegaconf import open_dict

import run_aldp_noise_solver_grid as shared

BASE_CONFIGURATION = shared.configuration
BASE_EXPECTED_COST = shared.expected_cost
BASE_VERIFY = shared.verify


def loss_matrix(mode='cross_entropy'):
    if mode not in ('cross_entropy', 'log_variance'):
        raise ValueError('Unregistered FC loss')
    rows = shared.matrix(shifted_chart_source=True)
    for row in rows:
        if mode == 'cross_entropy' and row['preset'] == 'ou' and not row['incompatible_reason']:
            row['incompatible_reason'] = 'Original adjoint CE supports linear reference only at fixed gamma=1; current trust-region gamma is retained'
        row['fc_loss_mode'] = mode
    return rows


def configuration(plan, campaign, stage, candidate):
    mode = plan.get('fc_loss_mode', 'cross_entropy')
    if plan['matrix'] != loss_matrix(mode):
        raise ValueError('Loss/source-aware compatibility matrix changed')
    cfg, output = BASE_CONFIGURATION(plan, campaign, stage, candidate)
    with open_dict(cfg):
        cfg.fc_loss_mode = mode
    return cfg, output


def expected_cost(cfg, phase_samples):
    cost = BASE_EXPECTED_COST(cfg, phase_samples)
    if cfg.fc_loss_mode == 'cross_entropy':
        # CE's terminal force adds 4B evaluations to LV's 2B per FC epoch.
        extra = 2 * int(cfg.max_outer_loops) * int(cfg.adjoint_matching_epochs) * int(cfg.buffer_samples_per_epoch)
        cost['training_and_online_openmm_configurations'] += extra
        cost['total_openmm_configurations'] += extra
    elif cfg.fc_loss_mode != 'log_variance':
        raise ValueError('No registered cost contract for this FC loss')
    return cost


@contextmanager
def shared_api():
    old_configuration, old_cost, old_verify = shared.configuration, shared.expected_cost, shared.verify
    shared.configuration, shared.expected_cost = configuration, expected_cost
    shared.verify = verify
    try:
        yield
    finally:
        shared.configuration, shared.expected_cost, shared.verify = old_configuration, old_cost, old_verify


def verify(directory, plan, source_sha, stage, candidate):
    with shared_api():
        report = BASE_VERIFY(directory, plan, source_sha, stage, candidate)
    if report['sampling_metadata']['fc_loss_mode'] != plan['fc_loss_mode']:
        raise RuntimeError('FC loss metadata mismatch')
    if report['diagnostics']['training_contract']['fc_loss_mode'] != plan['fc_loss_mode']:
        raise RuntimeError('Training collector/FC loss contract mismatch')
    return report


if __name__ == '__main__':
    import json
    import sys
    from pathlib import Path
    from aldp_adjoint_release import require_full_release
    campaign = Path(sys.argv[sys.argv.index('--campaign')+1])
    stage = sys.argv[sys.argv.index('--stage')+1]
    require_full_release(json.loads((campaign/'plan.json').read_text()), stage)
    with shared_api():
        shared.main()
