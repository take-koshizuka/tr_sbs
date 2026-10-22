#!/usr/bin/env python3
"""One bounded scheduler pass; individually gated cells, at most two GPU jobs."""
import argparse
import fcntl
import json
from pathlib import Path

from monitor_aldp_chart_campaign import ACTIVE, execute, scheduler
from run_aldp_chart_campaign import ROOT, receipt, sha256, verify_source, write_json
from run_aldp_noise_solver_grid import verify


def compare(reports):
    """A-only physical screen; cross-reference RN/ESS are disclosures."""
    control=reports.get('ou__linear_exponential')
    output=[]
    for name, r in reports.items():
        a=r['metrics']['a']
        contact=r['diagnostics']['trained_policy']['geometry']['fraction_below_reference_q01']
        row=dict(candidate=name,A=a,B=r['metrics']['b'],contact_fraction=contact,
            near_cut=r['near_cut_fraction_phi_psi'],raw_path_ess=r['raw_path_ess'],
            cost=r['cost'],phase_curve=r['diagnostics']['phase_end'],
            reference_specific_path_quantities=r['diagnostics']['dual'],
            physical_screen='control_not_complete',geometry_admission='not_established')
        if control:
            b=control['metrics']['a']
            refcontact=control['diagnostics']['trained_policy']['geometry']['fraction_below_reference_q01']
            gates=dict(js=a['torsion_js_mean']<=b['torsion_js_mean']+.01,
                basin=a['coarse_basin_tvd']<=b['coarse_basin_tvd']+.02,
                missing_mass=a['missing_reference_mass']<=b['missing_reference_mass']+.02,
                energy=a['physical_energy_w2']<=b['physical_energy_w2']*1.1,
                contact=contact<=refcontact+.02,
                near_cut_phi=r['near_cut_fraction_phi_psi'][0]<=control['near_cut_fraction_phi_psi'][0]+.02,
                near_cut_psi=r['near_cut_fraction_phi_psi'][1]<=control['near_cut_fraction_phi_psi'][1]+.02)
            improvement=(a['torsion_js_mean']<=b['torsion_js_mean']-.01 or a['coarse_basin_tvd']<=b['coarse_basin_tvd']-.02)
            row.update(physical_gates=gates,meaningful_A_improvement=improvement,
                physical_screen='conditional_candidate' if all(gates.values()) and improvement else 'not_admitted')
        output.append(row)
    return dict(rows=output,control='ou__linear_exponential',backend='torch',selection_bank='a',
                disclosure_bank='b',test_bank='closed',cross_reference_RN_ESS_used_for_selection=False,
                limits='One seed, finite grid/outer5, physical non-deterioration is relative to an imperfect control; no geometry or global optimum claim')


def submit(campaign, plan, python, stage, candidate, rows):
    key=stage+'__'+candidate
    intent=campaign/'intents'/f'{key}.json'
    if intent.exists(): raise RuntimeError('Existing intent needs reconciliation; duplicate prohibited')
    output=campaign/stage/candidate
    if output.exists(): raise RuntimeError('Existing output; immutable new attempt required')
    log=campaign/'logs'/f'{key}_%j.log'
    log.parent.mkdir(exist_ok=True)
    command=['sbatch','--parsable','--partition=gpu','--gres=gpu:1','--cpus-per-task=4',
        '--mem=32G','--exclude=htc-gpu024','--time='+('00:45:00' if stage=='smoke' else '3-00:00:00'),
        '--job-name=aldp-'+key,'--output='+str(log),str(ROOT/'exec_aldp_noise_solver_grid.sh'),
        str(ROOT),str(python),str(campaign),stage,candidate]
    write_json(intent,dict(status='dispatching',command=command))
    job=execute(command).stdout.strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Uncertain Slurm submission; inspect intent')
    row=dict(job_id=job,stage=stage,candidate=candidate,run_directory=str(output),
        log_path=str(log).replace('%j',job),source_sha256=plan['source_manifest_sha256'],
        plan_sha256=sha256(campaign/'plan.json'),python=str(python))
    write_json(intent,dict(status='submitted',row=row,command=command))
    rows.append(row)
    write_json(campaign/'ledger.json',rows)
    return row


def cached_smokes(plan):
    """Verify immutable parent evidence, without scheduling or retraining it."""
    cached=plan.get('cached_smokes')
    if not cached: return []
    parent=Path(cached['campaign'])
    for relative, expected in cached['evidence_files'].items():
        if sha256(parent/relative)!=expected:
            raise RuntimeError('Cached evidence changed: '+relative)
    parent_plan=json.loads((parent/'plan.json').read_text())
    manifest=parent.parent/'code'/'source_manifest.json'
    if sha256(manifest)!=cached['source_sha256']:
        raise RuntimeError('Cached source manifest changed')
    for relative, expected in json.loads(manifest.read_text())['files'].items():
        if sha256(manifest.parent/relative)!=expected:
            raise RuntimeError('Cached source file changed: '+relative)
    if parent_plan['frozen_inputs']!=plan['frozen_inputs']:
        raise RuntimeError('Cached input contract changed')
    required={c['candidate'] for c in plan['matrix'] if not c['incompatible_reason']}
    rows=cached['rows']
    if len(rows)!=len(required) or {r['candidate'] for r in rows}!=required:
        raise RuntimeError('Cached smoke set incomplete or duplicated')
    for row in rows:
        if row['stage']!='smoke' or row['source_sha256']!=cached['source_sha256'] or row['plan_sha256']!=sha256(parent/'plan.json'):
            raise RuntimeError('Cached ledger identity changed')
        if Path(row['run_directory'])!=parent/'smoke'/row['candidate']:
            raise RuntimeError('Cached directory mismatch')
        account=json.loads((parent/'accounting'/(row['job_id']+'.json')).read_text())['raw'].splitlines()
        if len(account)!=3 or any('COMPLETED|0:0' not in line for line in account):
            raise RuntimeError('Cached accounting is not successful')
        if 'postflight_verified=true' not in Path(row['log_path']).read_text():
            raise RuntimeError('Cached postflight marker missing')
        verify(row['run_directory'],parent_plan,cached['source_sha256'],'smoke',row['candidate'])
    return [dict(row,cached_read_only=True) for row in rows]


def classified_terminal_failures(plan):
    """Read-only pinned terminal failures; never resubmit or claim causal replay."""
    contract=plan.get('classified_terminal_failures')
    if not contract: return []
    for name, expected in contract['evidence_files'].items():
        if sha256(name)!=expected: raise RuntimeError('Terminal evidence changed: '+name)
    parent=Path(contract['parent_code'])
    if sha256(parent/'source_manifest.json')!=contract['parent_source_sha256']:
        raise RuntimeError('Terminal source changed')
    for name, expected in json.loads((parent/'source_manifest.json').read_text())['files'].items():
        if sha256(parent/name)!=expected: raise RuntimeError('Terminal source file changed: '+name)
    required={c['candidate'] for c in plan['matrix'] if not c['incompatible_reason']}
    rows=contract['rows']
    if len({r['candidate'] for r in rows})!=len(rows): raise RuntimeError('Duplicate classified failure')
    for row in rows:
        if row['candidate'] not in required or row['stage']!='full' or row['classification']!='terminal_numerical_failure':
            raise RuntimeError('Invalid classified failure')
        if row['retry_allowed'] or row['scientific_status']!='not_evaluated':
            raise RuntimeError('Failure retry/admission prohibited')
    return rows


def pass_once(campaign, python, advance=False):
    plan=json.loads((campaign/'plan.json').read_text())
    source=verify_source()
    if source != plan['source_manifest_sha256']: raise RuntimeError('Plan/source mismatch')
    for path, expected in plan['frozen_inputs'].items():
        if sha256(path)!=expected: raise RuntimeError('Frozen input changed '+path)
    rows=json.loads((campaign/'ledger.json').read_text()) if (campaign/'ledger.json').exists() else []
    terminal_failures=classified_terminal_failures(plan)
    verified, failures, active=cached_smokes(plan),[],[]
    reports={}
    for row in rows:
        state, code=scheduler(row['job_id'])
        if state in ACTIVE: active.append(dict(row,state=state)); continue
        account=execute(['sacct','-n','-P','-j',row['job_id'],'--format=JobIDRaw,State,ExitCode']).stdout
        write_json(campaign/'accounting'/f"{row['job_id']}.json",dict(raw=account))
        parts=[r.split('|') for r in account.splitlines() if r.strip()]
        terminal=[r for r in parts if r[0]==row['job_id'] or r[0].startswith(row['job_id']+'.')]
        if state!='COMPLETED' or code!='0:0' or any(r[1].split()[0]!='COMPLETED' or r[2]!='0:0' for r in terminal):
            failures.append(dict(row,state=state,exit_code=code,
                log_tail=Path(row['log_path']).read_text(errors='replace')[-4000:] if Path(row['log_path']).exists() else 'missing'))
            continue
        try:
            if row['source_sha256']!=source or row['plan_sha256']!=sha256(campaign/'plan.json'):
                raise RuntimeError('Ledger provenance mismatch')
            if 'postflight_verified=true' not in Path(row['log_path']).read_text(errors='replace'):
                raise RuntimeError('Missing postflight log marker')
            report=verify(row['run_directory'],plan,source,row['stage'],row['candidate'])
            if row['stage']=='full': reports[row['candidate']]=report
            verified.append(row)
        except Exception as error: failures.append(dict(row,reason=str(error)))
    submitted=[]
    keys={(r['stage'],r['candidate']) for r in rows+verified+terminal_failures}
    passed={(r['stage'],r['candidate']) for r in verified}
    if advance and not failures:
        failedkeys={(r['stage'],r['candidate']) for r in failures}
        all_smoke_terminal=all(('smoke',c['candidate']) in passed or ('smoke',c['candidate']) in failedkeys
                              for c in plan['matrix'] if not c['incompatible_reason'])
        for cell in plan['matrix']:
            if cell['incompatible_reason']: continue
            c=cell['candidate']
            stage='full' if ('smoke',c) in passed else 'smoke'
            if stage=='full' and not all_smoke_terminal: continue
            if (stage,c) in keys: continue
            if len(active)+len(submitted)>=plan['max_concurrent_gpu_jobs']: break
            submitted.append(submit(campaign,plan,python,stage,c,rows))
            keys.add((stage,c))
    cells=[]
    for cell in plan['matrix']:
        c=cell['candidate']
        status='incompatible' if cell['incompatible_reason'] else ('complete' if ('full',c) in passed else ('failed' if any(r['candidate']==c for r in failures+terminal_failures) else 'not_evaluated'))
        cells.append(dict(cell,status=status))
    complete=all(r['status'] in ('complete','failed','incompatible') for r in cells) and not active and not submitted
    previous=json.loads((campaign/'monitor_state.json').read_text()) if (campaign/'monitor_state.json').exists() else {}
    streak=0 if submitted or not active else previous.get('consecutive_active',0)+1
    result=dict(active=active,verified=verified,failures=failures,new_submissions=submitted,
        classified_terminal_failures=terminal_failures,complete=complete,consecutive_active=streak,recommended_monitor_minutes=30 if streak>=2 else 10)
    write_json(campaign/'aggregate/grid_status.json',dict(cells=cells,scientific_status='not_admitted',
        selection_bank='a',disclosure_bank='b',test_bank='closed',complete=complete, historical_failures=plan.get('historical_failures',[])+terminal_failures))
    write_json(campaign/'aggregate/comparison.json',compare(reports))
    write_json(campaign/'monitor_state.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--advance',action='store_true')
    args=parser.parse_args()
    with (args.campaign/'.monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(json.dumps(pass_once(args.campaign,args.python,args.advance),indent=2))


if __name__=='__main__': main()
