import json
from pathlib import Path
from types import SimpleNamespace

import monitor_aldp_loss_campaigns as monitor


def test_lv_failure_prevents_advancing_ce_in_same_bounded_pass(tmp_path,monkeypatch):
    ce=tmp_path/'ce/results';ce.mkdir(parents=True)
    (tmp_path/'plan.json').write_text(json.dumps(dict(peer_CE_campaign=dict(campaign=str(ce)))))
    monkeypatch.setattr(monitor,'verify_static_release',lambda *a,**kw:None)
    monkeypatch.setattr(monitor,'combined_health',lambda *a:dict(failures=[]))
    monkeypatch.setattr(monitor,'publish_comparison',lambda *a:None)
    commands=[]
    def run(command,**kw):
        commands.append(command)
        failed='monitor_aldp_scaled_grid.py' in command[1]
        return SimpleNamespace(returncode=0,stdout=json.dumps(dict(failures=[dict(job_id='lv1')] if failed else [],
            complete=False,active=[],new_submissions=[])),stderr='')
    monkeypatch.setattr(monitor.subprocess,'run',run)
    result=monitor.pass_once(tmp_path,'python',advance=True)
    assert result['dispatch_blocked']
    assert '--advance' in commands[0] and '--advance' not in commands[1]
    assert result['consecutive_active']==0


def test_four_failed_cells_stay_in_70_cell_aggregate(tmp_path):
    from run_aldp_noise_solver_grid import matrix
    from run_aldp_adjoint_loss_grid import loss_matrix
    ce=tmp_path/'ce/results';ce.mkdir(parents=True)
    (tmp_path/'aggregate').mkdir()
    (ce/'aggregate').mkdir()
    failed={'brownian_constant__euler','brownian_constant__legacy','brownian_constant__diffusion_euler','brownian_linear__euler'}
    lv=[dict(r,status='incompatible' if r['incompatible_reason'] else 'failed' if r['candidate'] in failed else 'not_evaluated') for r in matrix(True)]
    original=[dict(r,status='incompatible' if r['incompatible_reason'] else 'not_evaluated') for r in loss_matrix()]
    (tmp_path/'aggregate/grid_status.json').write_text(json.dumps(dict(cells=lv)))
    (ce/'aggregate/grid_status.json').write_text(json.dumps(dict(cells=original)))
    p=dict(matrix=matrix(True),peer_CE_campaign=dict(campaign=str(ce)))
    monitor.publish_comparison(tmp_path,p,dict(LV=dict(verified=[]),CE=dict(verified=[])))
    grid=json.loads((tmp_path/'aggregate/loss_grid_status.json').read_text())
    assert len(grid['cells'])==70 and sum(r['status']=='failed' for r in grid['cells'])==4
    pairs=json.loads((tmp_path/'aggregate/loss_pairs.json').read_text())
    assert len(pairs['rows'])==35
    assert sum(r['status']=='terminal_failure_no_complete_pair' for r in pairs['rows'])==4
