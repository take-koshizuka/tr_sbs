## Running command

```bash
# lj13 dataset experiment 
uv run python -u -m tr_asbs.experiment.train_asbs experiment=lj13 # with asbs
uv run python -u -m tr_asbs.experiment.train_gamma_asbs experiment=lj13
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=lj13 gamma_schedule=fixed
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=lj13 gamma_schedule=trust_region
```



# gmm varying-dimension experiment
```bash
uv run python -u -m tr_asbs.experiment.train_asbs experiment=gmm_dim
uv run python -u -m tr_asbs.experiment.train_gamma_asbs experiment=gmm_dim
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=gmm_dim gamma_schedule=fixed
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=gmm_dim gamma_schedule=trust_region
```

## GMM PIS/DIS and six-method comparison

PIS and DIS use a shared PyTorch variational-training entrypoint. Their GMM
defaults use the same FourierMLP backbone as the ASBS controller, without a
target-score hint. They can be run against the same `gmm_mode` and `gmm_dim`
targets as the ASBS methods:

```bash
uv run python -u -m tr_asbs.experiment.train_variational experiment=gmm_mode algorithm=pis
uv run python -u -m tr_asbs.experiment.train_variational experiment=gmm_mode algorithm=dis
uv run python -u -m tr_asbs.experiment.train_variational experiment=gmm_dim algorithm=pis
uv run python -u -m tr_asbs.experiment.train_variational experiment=gmm_dim algorithm=dis
```

PIS starts its paths at zero. DIS starts from the isotropic Gaussian specified
by `source.scale`. Both methods use `base_sde.sigma_min/max` for their fixed
cosine schedules; they do not instantiate the ASBS source or SDE objects.

The score-free PIS/DIS grid contains 54 Slurm jobs (30 mode-grid jobs and 24
dimension-grid jobs). Inspect the commands before submitting:

```bash
bash submit_gmm_variational_asbs_control_grid.sh --dry-run
```

The target seed is fixed at zero in every grid job, while seeds 0, 1, and 2
change model initialization and rollout randomness. Submission manifests and
Slurm logs are written below
`slurm_logs/gmm_variational_asbs_control_scoreless_v1/`.

The completed ASBS, Gamma-ASBS, IW fixed, IW TR, PIS, and DIS mode/dimension
comparison, including frozen aggregate data and reproducible figures, is in
[`reports/gmm_scorefree_six_methods/`](reports/gmm_scorefree_six_methods/README.md).

On z3, the historical repository path
`/home/htc/tkoshizuka/SCRATCH/tr_asbs_experiments_lorenz_gmm` is on the
home filesystem. The six-method Slurm runners now default their output to
`/scratch/htc/$USER/tr_asbs_gmm_runs/<campaign>` when that directory is
writable. Set `TR_ASBS_SCRATCH_ROOT` for a different scratch parent, or
`TR_ASBS_ROOT` to override the complete run root. `/scratch/htc` is the
actual scratch filesystem; check its retention and backup policy before
relying on it as the only copy of irreplaceable results.
