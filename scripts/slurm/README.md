# Remote experiment recipes

These launchers retain the existing experiment settings. Their location changed
from the repository root; Python trainer and Hydra preset names are unchanged.
The pre-cleanup layout is available in commit `94c93ab`.

Run submitters from the repository root. Inspect commands before submission:

```bash
bash scripts/slurm/submit_gmm_mode_full_grid.sh --dry-run
bash scripts/slurm/submit_gmm_dim_full_grid.sh --dry-run
bash scripts/slurm/submit_gmm_variational_asbs_control_grid.sh --dry-run
```

These generate 60, 48, and 54 jobs, respectively. They preserve the tuned,
method-specific GMM settings and historical result keys. See the
[frozen comparison](../../reports/gmm_scorefree_six_methods/README.md) for seeds,
checkpoint selection, evaluation counts, and unequal training budgets.

`TR_ASBS_REPO_ROOT` selects the checkout, `TR_ASBS_PYTHON` the Python interpreter,
`TR_ASBS_ROOT` the output directory, and `TR_ASBS_LOG_ROOT` the submission-log
directory where supported by the selected recipe. Each submitter records its
commands and scheduler identifiers. Keep those records with the run outputs.

The retained resource requests and node exclusions describe the original z3
environment. Adapt scheduler resources for another cluster without changing the
scientific overrides. Full experiments belong on remote compute. Campaign
monitors in `scripts/monitor_*.py` retain their separate admission and failure
checks; run them periodically rather than holding an interactive session open.

The `general/` directory contains the existing backward-loss comparison recipes.
Fixed personal W&B sweep launchers were removed; their source remains in the
preservation commit.
