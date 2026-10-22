"""A budget-only continuation can reuse smoke; a scientific change cannot."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from monitor_aldp_matched_mechanism import smoke_equivalence


def test_preflight_reuse_requires_identical_scientific_smoke_settings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / 'reports/aldp_chart_20261007_5a093a95/mechanism_v1/plan.json').read_text())
    new = deepcopy(old)
    new['candidates'] = {key: dict(old['candidates'][key], outer_loops=10, openmm_cap=100000)
                         for key in ('source_half', 'clip1000')}
    assert set(smoke_equivalence(old, new, tmp_path / 'old', tmp_path / 'new')) == {'source_half', 'clip1000'}
    new['candidates']['source_half']['source_scale'] = .75
    with pytest.raises(ValueError, match='settings differ'):
        smoke_equivalence(old, new, tmp_path / 'old', tmp_path / 'new')


def test_new_candidate_cannot_reuse_preflight(tmp_path):
    old = dict(candidates={'source_half': {}})
    new = dict(candidates={'new_source': {}})
    with pytest.raises(ValueError, match='requires a new preflight'):
        smoke_equivalence(old, new, tmp_path, tmp_path)
