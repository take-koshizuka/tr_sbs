"""Cheap status polling; full original verification at execution boundaries."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=15).stdout


def tail(path, limit=8192):
    p = Path(path)
    if not p.exists():
        return dict(path=str(p), exists=False)
    with p.open('rb') as stream:
        stream.seek(max(0, p.stat().st_size-limit))
        data = stream.read(limit)
    return dict(path=str(p), exists=True, bytes=p.stat().st_size,
                sha256_scope='last_8192_bytes_snapshot', sha256=hashlib.sha256(data).hexdigest(),
                text=data.decode(errors='replace'))


def inspect(registration, accounting, queue):
    """Never submit or infer new completion from queue disappearance."""
    parts = [line.split('|') for line in accounting.splitlines() if line.strip()]
    active, failures, changed_terminal, logs, jobs = [], [], [], [], []
    for label, spec in registration['pools'].items():
        root = Path(spec['campaign'])
        if digest(root/'plan.json') != spec['plan_sha256']:
            raise RuntimeError(label+' immutable plan changed')
        if digest(root.parent/'code/source_manifest.json') != spec['source_sha256']:
            raise RuntimeError(label+' source manifest changed')
        rows = load(root/'ledger.json') if (root/'ledger.json').exists() else []
        intents = [load(p) for p in (root/'intents').glob('*.json')]
        if any(i['status'] != 'submitted' for i in intents):
            failures.append(dict(loss=label, reason='uncertain_submission_intent'))
        row_ids = {r['job_id'] for r in rows}
        if len(row_ids) != len(rows) or {i['row']['job_id'] for i in intents if i['status']=='submitted'} != row_ids:
            failures.append(dict(loss=label, reason='ledger_intent_mismatch'))
        for row in rows:
            if row['source_sha256'] != spec['source_sha256'] or row['plan_sha256'] != spec['plan_sha256']:
                raise RuntimeError('Ledger provenance changed')
            jobs.append(row['job_id'])
            records = [p for p in parts if p[0] == row['job_id'] or p[0].startswith(row['job_id']+'.')]
            directory = Path(row['run_directory'])
            log = tail(row['log_path']); logs.append(dict(log, job_id=row['job_id'], loss=label))
            failure = directory/'failure.json'
            if failure.exists():
                failures.append(dict(row, loss=label, reason='failure_artifact', failure_sha256=digest(failure)))
                continue
            if not records:
                failures.append(dict(row, loss=label, reason='missing_accounting'))
                continue
            states = [p[1].split()[0].rstrip('+') for p in records]
            if any(s not in ACTIVE|{'COMPLETED'} for s in states):
                failures.append(dict(row, loss=label, reason='scheduler_failure', accounting=records))
                continue
            if any(s in ACTIVE for s in states):
                active.append(dict(row, loss=label, state=states[0]))
                if 'Traceback (most recent call last)' in log.get('text', ''):
                    failures.append(dict(row, loss=label, reason='active_log_traceback'))
                continue
            if ({p[0] for p in records} != {row['job_id'], row['job_id']+'.batch', row['job_id']+'.extern'}
                    or any(p[2] != '0:0' for p in records)):
                failures.append(dict(row, loss=label, reason='terminal_accounting_incomplete', accounting=records))
                continue
            expected = registration['previously_verified_completed'].get(row['job_id'])
            if not expected:
                changed_terminal.append(dict(row, loss=label))
            else:
                for name, sha in expected.items():
                    if not (directory/name).exists() or digest(directory/name) != sha:
                        failures.append(dict(row, loss=label, reason='verified_terminal_metadata_changed', file=name))
    queued = [line.split('|') for line in queue.splitlines() if 'aldp' in line.lower()]
    if len(queued) > 2 or any(p[0] not in set(jobs) for p in queued):
        failures.append(dict(reason='untracked_or_excess_combined_ALDP_jobs', queue=queued))
    return dict(active=active, failures=failures, newly_terminal=changed_terminal,
                combined_ALDP_queue=queued, log_snapshots=logs,
                needs_full_validation=bool(changed_terminal or failures or len(queued)<2),
                complete=False, total_registered_cells=70,
                validation_scope='status_only; old large artifacts are not rehashed; no new quality admission')


def pass_once(registration, output, python, advance=False, check_only=False):
    started = time.monotonic()
    state_path = output.parent.parent/'monitor_state.json'
    previous = load(state_path) if state_path.exists() else {}
    registration = dict(registration, previously_verified_completed=dict(
        registration['previously_verified_completed'], **previous.get('verified_completed', {})))
    ids = [row['job_id'] for spec in registration['pools'].values()
           for row in (load(Path(spec['campaign'])/'ledger.json') if (Path(spec['campaign'])/'ledger.json').exists() else [])]
    accounting = command(['sacct','-j',','.join(ids),'-P','-n','-o','JobIDRaw,State,ExitCode']) if ids else ''
    queue = command(['squeue','-u',getpass.getuser(),'-h','-o','%i|%j|%T'])
    result = inspect(registration, accounting, queue)
    result.update(mode='light_status', accounting=accounting, new_submissions=[],
                  dispatch_blocked=bool(result['failures']), full_verification_executed=False)
    if result['needs_full_validation'] and not check_only:
        lv = registration['pools']['LV']
        # Original full checks, original job roots and original release guards.
        if digest(Path(lv['campaign']).parent/'code/scripts/monitor_aldp_loss_campaigns.py') != registration['parent_monitor_sha256']:
            raise RuntimeError('Original full monitor changed')
        env = dict(os.environ); env.pop('JAX_PLATFORMS',None)
        code = Path(lv['campaign']).parent/'code'
        env['PYTHONPATH'] = str(code/'scripts')+':'+str(code/'src')
        args = [python, str(Path(__file__).with_name('full_coordinator.py')),
                '--campaign', lv['campaign'], '--python', python]
        if advance and not result['failures']:
            args.append('--advance')
        run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=480)
        (output/'full_stdout.txt').write_text(run.stdout)
        (output/'full_stderr.txt').write_text(run.stderr)
        if run.returncode:
            raise RuntimeError('Full validation failed; no replay: '+run.stderr[-5000:])
        full = json.loads(run.stdout)
        boundary_failures = result['failures'] + [dict(row, loss=label)
            for label, report in full['reports'].items() for row in report['failures']]
        result.update(mode='full_boundary_validation', full_verification_executed=True,
                      full_report=full, active=full['active'], new_submissions=full['new_submissions'],
                      dispatch_blocked=bool(boundary_failures) or full['dispatch_blocked'],
                      complete=full['complete'] and not boundary_failures, failures=boundary_failures)
        if not boundary_failures:
            verified = dict(previous.get('verified_completed', {}))
            for report in full['reports'].values():
                for row in report['verified']:
                    if row['stage']=='full':
                        directory = Path(row['run_directory'])
                        verified[row['job_id']] = {name:digest(directory/name)
                            for name in ('receipt.json','result.json','config.yaml')}
            previous['verified_completed'] = verified
    result['elapsed_seconds'] = time.monotonic()-started
    streak = 0 if result['new_submissions'] or not result['active'] else previous.get(
        'consecutive_active', registration.get('initial_active_streak', 0))+1
    result['consecutive_active'] = streak
    result['recommended_monitor_minutes'] = 30 if streak>=2 else 10
    if not check_only:
        temporary=state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(previous,consecutive_active=streak),indent=2)+'\n')
        temporary.replace(state_path)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--advance',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    registration=load(args.registration)
    code=Path(__file__).parent
    for name, sha in registration['monitor_code_sha256'].items():
        if digest(code/name)!=sha: raise RuntimeError('Light monitor source changed')
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Preserve the previous monitor output; use a new pass directory')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output.parent/'.light_dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Share the old common dispatch lock, so old and new entry points cannot race.
        lv=Path(registration['pools']['LV']['campaign'])
        with (lv/'.pair_monitor.lock').open('a') as parent_lock:
            if args.check_only:
                fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Full coordinator takes this lock itself; holding it here would deadlock.
            result=pass_once(registration,args.output,args.python,args.advance,args.check_only)
        (args.output/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
