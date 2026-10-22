"""One coordinated pass for both loss campaigns, with a shared stop boundary."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from aldp_scaled_release import combined_health, verify_static_release
from run_aldp_chart_campaign import write_json
from analyze_aldp_loss_pairs import compare_loss_pairs


def publish_comparison(campaign, plan, reports):
    cells, complete_reports = [], {}
    roots = dict(LV=campaign, CE=Path(plan['peer_CE_campaign']['campaign']))
    for label, root in roots.items():
        grid = json.loads((root/'aggregate/grid_status.json').read_text())
        cells.extend(dict(row,loss=label) for row in grid['cells'])
        complete_reports[label] = {row['candidate']:json.loads((Path(row['run_directory'])/'result.json').read_text())
            for row in reports[label]['verified'] if row['stage']=='full'}
    if len(cells) != 70 or len({(r['loss'],r['candidate']) for r in cells}) != 70:
        raise RuntimeError('Loss aggregate lost a registered/failed/incompatible cell')
    paired = compare_loss_pairs(complete_reports['CE'],complete_reports['LV'])
    by_candidate = {r['candidate']:r for r in paired['rows']}
    status = {(r['loss'],r['candidate']):r['status'] for r in cells}
    for cell in plan['matrix']:
        c=cell['candidate']
        if c not in by_candidate:
            by_candidate[c]=dict(candidate=c,status='pending_verified_full_pair',geometry_admission='not_established')
        row=by_candidate[c]
        row['cell_status']=dict(CE=status['CE',c],LV=status['LV',c])
        if 'failed' in row['cell_status'].values():
            row['status']='terminal_failure_no_complete_pair'
        elif 'incompatible' in row['cell_status'].values():
            row['status']='incompatible_loss_pair'
    paired['rows']=[by_candidate[c['candidate']] for c in plan['matrix']]
    write_json(campaign/'aggregate/loss_pairs.json',paired)
    write_json(campaign/'aggregate/loss_grid_status.json',dict(cells=cells,total_registered_cells=70,
        scientific_status='not_admitted',selection_bank='a',disclosure_bank='b',test_bank='closed'))


def pass_once(campaign, python, advance=False):
    plan = json.loads((campaign/'plan.json').read_text())
    verify_static_release(plan, verify_preflights=False)
    initial = combined_health(campaign, plan)
    blocked = bool(initial['failures'])
    reports = {}
    # At most one new LV job leaves a slot for CE when both pools have work.
    pairs = [('LV',campaign,'monitor_aldp_scaled_grid.py'),
             ('CE',Path(plan['peer_CE_campaign']['campaign']),'monitor_aldp_adjoint_loss_grid.py')]
    for label, root, script in pairs:
        code = root.parent/'code'
        env = dict(os.environ)
        env.pop('JAX_PLATFORMS',None)
        env['PYTHONPATH'] = str(code/'scripts')+':'+str(code/'src')
        command = [str(python),str(code/'scripts'/script),'--campaign',str(root),'--python',str(python)]
        if advance and not blocked:
            command.append('--advance')
        run = subprocess.run(command,env=env,capture_output=True,text=True,timeout=55)
        if run.returncode:
            raise RuntimeError(label+' monitor failed: '+run.stderr[-5000:])
        report = json.loads(run.stdout)
        reports[label] = report
        blocked = blocked or bool(report['failures']) or bool(report.get('cross_loss_health',{}).get('failures'))
    new = [dict(row,loss=label) for label,report in reports.items() for row in report['new_submissions']]
    active = [dict(row,loss=label) for label,report in reports.items() for row in report['active']]
    previous = json.loads((campaign/'pair_monitor_state.json').read_text()) if (campaign/'pair_monitor_state.json').exists() else {}
    streak = 0 if new or not active else previous.get('consecutive_active',0)+1
    result = dict(reports=reports,initial_health=initial,new_submissions=new,active=active,
        dispatch_blocked=blocked,complete=all(r['complete'] for r in reports.values()),
        consecutive_active=streak,recommended_monitor_minutes=30 if streak>=2 else 10,
        total_registered_cells=70,combined_GPU_limit=2)
    publish_comparison(campaign,plan,reports)
    write_json(campaign/'pair_monitor_state.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--advance',action='store_true')
    args=parser.parse_args()
    with (args.campaign/'.pair_monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(json.dumps(pass_once(args.campaign,args.python,args.advance),allow_nan=False))
