"""Pinned LV continuation proof and health checks for the two loss campaigns."""
import json
from pathlib import Path

from aldp_adjoint_release import combined_active_jobs, diagnostic_release_state
from monitor_aldp_chart_campaign import ACTIVE, execute
from run_aldp_chart_campaign import sha256
from run_aldp_noise_solver_grid import matrix


FAILED_JOBS = {'5263587', '5263588', '5278815', '5278816'}


def validate_plan(plan):
    if (plan['allowed_stages'] != ['full'] or
        plan['controller_backward_mode'] != 'loss_scaled_fp64_clip' or
        plan['fc_loss_mode'] != 'log_variance' or plan['matrix'] != matrix(True)):
        raise RuntimeError('Changed registered LV continuation contract')
    if plan['outer_loops'] != 5 or plan['combined_ALDP_gpu_limit'] != 2:
        raise RuntimeError('Changed outer/concurrency boundary')
    failed = plan['classified_terminal_failures']['rows']
    if {r['job_id'] for r in failed} != FAILED_JOBS:
        raise RuntimeError('All four historical numerical failures must remain')
    if any(r['retry_allowed'] or r['scientific_status'] != 'not_evaluated' for r in failed):
        raise RuntimeError('Historical failure retry/admission prohibited')
    required = {c['candidate'] for c in plan['matrix'] if not c['incompatible_reason']}
    remaining = required - {r['candidate'] for r in failed}
    if len(remaining) != 14 or set(plan['remaining_candidates']) != remaining:
        raise RuntimeError('Changed remaining grid or silently pruned failure')
    return remaining


def verify_source_contract(code, expected):
    code = Path(code)
    if sha256(code/'source_manifest.json') != expected:
        raise RuntimeError('Pinned source manifest changed: '+str(code))
    files = json.loads((code/'source_manifest.json').read_text())['files']
    for name, value in files.items():
        if sha256(code/name) != value:
            raise RuntimeError('Pinned source file changed: '+name)
    return len(files)


def verify_static_release(plan, verify_preflights=True):
    """No new experiment or polling: verify immutable proof and parent artifacts."""
    from monitor_aldp_noise_solver_grid import cached_smokes, classified_terminal_failures
    remaining = validate_plan(plan)
    guard = plan['diagnostic_release_guard']
    output = Path(guard['campaign'])/'diagnostic'
    if (sha256(output/'gate.json') != guard['gate_sha256'] or
        sha256(output/'receipt.json') != guard['receipt_sha256']):
        raise RuntimeError('Verified numerical proof changed')
    state = diagnostic_release_state(plan)
    if not state['ready']:
        raise RuntimeError('Registered numerical gate is not verified')
    # A monitor delegates these expensive immutable parent reads to its one
    # shared pass. A direct worker checks them here before target evaluation.
    smokes = cached_smokes(plan) if verify_preflights else plan['cached_smokes']['rows']
    failures = classified_terminal_failures(plan) if verify_preflights else plan['classified_terminal_failures']['rows']
    for code, expected in plan['classified_failure_sources'].items():
        verify_source_contract(code, expected)
    peer = plan['peer_CE_campaign']
    campaign = Path(peer['campaign'])
    if sha256(campaign/'plan.json') != peer['plan_sha256']:
        raise RuntimeError('Peer CE immutable plan changed')
    peer_plan = json.loads((campaign/'plan.json').read_text())
    if peer_plan['fc_loss_mode'] != 'cross_entropy' or peer_plan['allowed_stages'] != ['full']:
        raise RuntimeError('Wrong peer CE loss or stage')
    verify_source_contract(campaign.parent/'code', peer['source_sha256'])
    if peer_plan['frozen_inputs'] != plan['frozen_inputs']:
        raise RuntimeError('Cross-loss source/calibration input mismatch')
    if len(smokes) != 18 or len(failures) != 4:
        raise RuntimeError('Incomplete cached or failed parent evidence')
    return dict(status='verified_scaled_LV_release', cached_smokes=18,
                retained_failed_fulls=4, remaining_fulls=len(remaining),
                numerical_gate_sha256=guard['gate_sha256'])


def campaign_health(campaign, source, plan_sha):
    """Read current attempts without advancing a monitor or its active streak."""
    campaign = Path(campaign)
    if sha256(campaign/'plan.json') != plan_sha:
        raise RuntimeError('Campaign plan changed')
    ledger = json.loads((campaign/'ledger.json').read_text()) if (campaign/'ledger.json').exists() else []
    if not ledger:
        return dict(active=[], failures=[])
    for row in ledger:
        if row['source_sha256'] != source or row['plan_sha256'] != plan_sha:
            raise RuntimeError('Ledger source/plan changed')
    raw = execute(['sacct', '-j', ','.join(r['job_id'] for r in ledger), '-P', '-n',
                   '-o', 'JobIDRaw,State,ExitCode']).stdout
    account = [line.split('|') for line in raw.splitlines() if line.strip()]
    active, failures = [], []
    for row in ledger:
        job = row['job_id']
        parts = [p for p in account if p[0] == job or p[0].startswith(job+'.')]
        directory = Path(row['run_directory'])
        failure = directory/'failure.json'
        if failure.exists():
            failures.append(dict(row, reason='unclassified_failure_artifact', failure_sha256=sha256(failure)))
        elif any(p[1].split()[0] not in ACTIVE and p[1] != 'COMPLETED' for p in parts):
            failures.append(dict(row, reason='unclassified_scheduler_failure', accounting=parts))
        elif not parts or any(p[1].split()[0] in ACTIVE for p in parts):
            active.append(row)
        elif ({p[0] for p in parts} != {job, job+'.batch', job+'.extern'} or
              any(p[1] != 'COMPLETED' or p[2] != '0:0' for p in parts)):
            failures.append(dict(row, reason='unclassified_scheduler_failure', accounting=parts))
        else:
            receipt = directory/'receipt.json'
            result = directory/'result.json'
            if not receipt.exists() or not result.exists():
                failures.append(dict(row, reason='missing_terminal_receipt_or_result'))
                continue
            document = json.loads(receipt.read_text())
            report = json.loads(result.read_text())
            if (not document.get('postflight_verified') or report.get('rejected_training_steps') or
                report.get('global_step') != 1500000):
                failures.append(dict(row, reason='unclassified_terminal_numerical_contract'))
    return dict(active=active, failures=failures)


def combined_health(campaign, plan):
    peer = plan['peer_CE_campaign']
    own = campaign_health(campaign, plan['source_manifest_sha256'], sha256(Path(campaign)/'plan.json'))
    ce = campaign_health(peer['campaign'], peer['source_sha256'], peer['plan_sha256'])
    return dict(LV=own, CE=ce, failures=own['failures']+ce['failures'])


def require_scaled_release(plan, stage, candidate):
    remaining = validate_plan(plan)
    if stage != 'full' or candidate not in remaining:
        raise RuntimeError('Only unsubmitted registered LV full cells are released')
    return verify_static_release(plan)
