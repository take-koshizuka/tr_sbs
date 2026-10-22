"""The Slurm worker must discard a CPU selection inherited from submission."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('inherited_platform', ['cpu', None])
def test_worker_discards_submission_platform(tmp_path, inherited_platform):
    root = Path(__file__).resolve().parents[1]
    mock_bin = tmp_path / 'bin'
    mock_bin.mkdir()
    gpu_info = mock_bin / 'nvidia-smi'
    gpu_info.write_text('#!/bin/sh\nprintf "mock GPU\\n"\n')
    gpu_info.chmod(0o755)
    mock_python = mock_bin / 'worker-python'
    mock_python.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["WORKER_ENV_CAPTURE"]).write_text(json.dumps(dict(\n'
        'platform=os.environ.get("JAX_PLATFORMS"), \n'
        'x64=os.environ.get("JAX_ENABLE_X64"), args=sys.argv[1:])))\n'
    )
    mock_python.chmod(0o755)
    capture = tmp_path / 'capture.json'
    env = dict(os.environ, PATH=str(mock_bin) + os.pathsep + os.environ['PATH'],
               WORKER_ENV_CAPTURE=str(capture))
    env.pop('JAX_PLATFORMS', None)
    if inherited_platform is not None:
        env['JAX_PLATFORMS'] = inherited_platform
    result = subprocess.run(['bash', str(root / 'exec_aldp_inner_continuation.sh'),
        str(root), str(mock_python), str(tmp_path / 'campaign'), 'u_short'],
        env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    saved = json.loads(capture.read_text())
    assert saved['platform'] is None
    assert saved['x64'] == 'true'
    assert saved['args'] == ['-u', 'scripts/run_aldp_inner_continuation.py',
                            '--campaign', str(tmp_path / 'campaign'), '--candidate', 'u_short']
