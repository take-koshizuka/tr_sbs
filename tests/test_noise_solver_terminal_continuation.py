import pytest
from monitor_aldp_noise_solver_grid import classified_terminal_failures

def test_no_classification():
    assert classified_terminal_failures({})==[]

def test_changed_evidence_blocks_before_scheduling(tmp_path):
    p=tmp_path/'evidence';p.write_text('changed')
    with pytest.raises(RuntimeError,match='Terminal evidence changed'):
        classified_terminal_failures({'classified_terminal_failures':{'evidence_files':{str(p):'wrong'}}})

def test_failure_not_used_as_success():
    from pathlib import Path
    s=Path('scripts/monitor_aldp_noise_solver_grid.py').read_text()
    assert 'rows+verified+terminal_failures' in s
    assert 'failures+terminal_failures' in s
    assert 'advance and not failures' in s
