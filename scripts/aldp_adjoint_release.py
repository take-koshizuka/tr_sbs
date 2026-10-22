"""Read-only release checks and one shared ALDP GPU-slot boundary."""
import getpass
import json
from pathlib import Path

from monitor_aldp_chart_campaign import execute
from run_aldp_chart_campaign import sha256


def combined_active_jobs():
    raw = execute(['squeue', '-u', getpass.getuser(), '-h', '-o', '%i|%j|%T']).stdout
    return [line.split('|') for line in raw.splitlines() if 'aldp' in line.lower()]


def diagnostic_release_state(plan):
    """A retry must be verified before releasing more work after its old bug."""
    if plan['allowed_stages'] != ['full']:
        return dict(status='smoke_only', ready=True)
    guard = plan.get('diagnostic_release_guard')
    if not guard:
        raise RuntimeError('Full release requires the registered diagnostic guard')
    campaign = Path(guard['campaign'])
    diagnostic_plan = json.loads((campaign/'plan.json').read_text())
    assert sha256(campaign/'plan.json') == guard['plan_sha256']
    manifest = campaign.parent/'code/source_manifest.json'
    assert sha256(manifest) == guard['source_sha256'] == diagnostic_plan['source_manifest_sha256']
    for name, expected in json.loads(manifest.read_text())['files'].items():
        assert sha256(manifest.parent/name) == expected, name
    for name, expected in diagnostic_plan['frozen_inputs'].items():
        assert sha256(name) == expected, name
    assert diagnostic_plan['tolerances'] == guard['tolerances']
    submission = json.loads((campaign/'submission.json').read_text())
    assert submission['job_id'] == guard['job_id']
    raw = execute(['sacct', '-j', guard['job_id'], '-P', '-n',
                   '-o', 'JobIDRaw,State,ExitCode']).stdout
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    if not rows or any(row[1].split()[0] in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING') for row in rows):
        return dict(status='diagnostic_pending_verification', ready=False, accounting=rows)
    if any(row[1] != 'COMPLETED' or row[2] != '0:0' for row in rows):
        return dict(status='diagnostic_failure_requires_classification', ready=False, accounting=rows)
    assert {row[0] for row in rows} == {guard['job_id'], guard['job_id']+'.batch', guard['job_id']+'.extern'}
    output = campaign/'diagnostic'
    gate = json.loads((output/'gate.json').read_text())
    receipt = json.loads((output/'receipt.json').read_text())
    for name, expected in receipt['files'].items():
        assert sha256(output/name) == expected, name
    assert receipt['inputs']['source_sha256'] == guard['source_sha256']
    assert receipt['inputs']['plan_sha256'] == guard['plan_sha256']
    assert gate['status'] == 'passed' and gate['tolerances'] == guard['tolerances']
    assert gate['source_sha256'] == guard['source_sha256'] and gate['plan_sha256'] == guard['plan_sha256']
    assert gate['OpenMM'] == 0 and gate['optimizer_updates'] == gate['reference_optimizer_updates'] == 40
    assert gate['matmul_precision'] == 'highest' and gate['optimizer_updates_discarded']
    assert not gate['production_changed'] and not gate['full_training_submitted']
    assert len(gate['rows']) == 40 and all(all(row['gates'].values()) for row in gate['rows'])
    for case in ['normal', 'witness']:
        case_rows = [row for row in gate['rows'] if row['case'] == case]
        assert [row['update'] for row in case_rows] == list(range(1, 21))
        for row in case_rows:
            expected = [v+row['update'] for v in diagnostic_plan['initial_Adam_counters'][case]]
            assert row['Adam_counters'] == row['reference_Adam_counters'] == row['expected_Adam_counters'] == expected
    return dict(status='verified_saved_batch_gate', ready=True, accounting=rows,
                gate_sha256=sha256(output/'gate.json'), receipt_sha256=sha256(output/'receipt.json'))


def require_full_release(plan, stage=None):
    if stage is not None and stage not in plan['allowed_stages']:
        raise RuntimeError('Requested worker stage is not released by immutable plan')
    if plan['allowed_stages'] != ['full']:
        return
    state = diagnostic_release_state(plan)
    if not state['ready']:
        raise RuntimeError('Full release blocked: '+state['status'])
    cached = plan['cached_smokes']
    parent = Path(cached['campaign'])
    for name, expected in cached['evidence_files'].items():
        assert sha256(parent/name) == expected, name
    source = parent.parent/'code/source_manifest.json'
    assert sha256(source) == cached['source_sha256']
    for name, expected in json.loads(source.read_text())['files'].items():
        assert sha256(source.parent/name) == expected, name
    parent_plan = json.loads((parent/'plan.json').read_text())
    assert parent_plan['fc_loss_mode'] == plan['fc_loss_mode'] == 'cross_entropy'
    assert parent_plan['frozen_inputs'] == plan['frozen_inputs']
    required = {cell['candidate'] for cell in plan['matrix'] if not cell['incompatible_reason']}
    assert len(cached['rows']) == len(required) == 15
    assert {row['candidate'] for row in cached['rows']} == required
    for row in cached['rows']:
        assert row['stage'] == 'smoke' and row['source_sha256'] == cached['source_sha256']
        assert row['plan_sha256'] == sha256(parent/'plan.json')
        directory = Path(row['run_directory'])
        assert directory == parent/'smoke'/row['candidate']
        report = json.loads((directory/'result.json').read_text())
        assert report['sampling_metadata']['fc_loss_mode'] == 'cross_entropy'
        assert report['global_step'] == 4 and not report['rejected_training_steps']
        assert report['cost']['total_openmm_configurations'] == 584
        account = json.loads((parent/'accounting'/(row['job_id']+'.json')).read_text())['raw'].strip().splitlines()
        assert len(account) == 3 and all(line.split('|')[1:] == ['COMPLETED', '0:0'] for line in account)
        assert 'postflight_verified=true' in Path(row['log_path']).read_text()
