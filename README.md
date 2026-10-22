# TR-SBS

Code for controlled-diffusion sampling from unnormalized target distributions,
including Trust-Region Schrödinger Bridge Sampler (TR-SBS), ASBS variants,
Path Integral Sampler (PIS), and Time-Reversed Diffusion Sampler (DIS).

The repository retains the existing training implementations and experiment
configurations for GMM, Many-well, Lennard–Jones, Morse, and alanine dipeptide.
The public launcher gives these implementations a common interface and uses
the method names below. Historical entrypoints and stored result keys remain
available for reproducibility.

## Methods

| Paper name | Launcher name | Update |
| --- | --- | --- |
| ASBS | `asbs` | Adjoint and corrector matching |
| Gamma-ASBS | `gamma-asbs` | ASBS with fixed interpolation parameter γ |
| TR-ASBS | `tr-asbs` | Trust-region variant of Gamma-ASBS, without IW |
| ASBS-IW | `asbs-iw` | Importance-weighted update with fixed γ |
| TR-SBS | `tr-sbs` | Importance-weighted update with trust-region γ |
| PIS | `pis` | Variational path-integral sampling |
| DIS | `dis` | Variational reverse-OU diffusion sampling |

TR-ASBS belongs to the Gamma-ASBS family. The former names “IW-ASBS: TR”
and “IW-ASBS (fixed γ)” correspond to TR-SBS and ASBS-IW, respectively.
See [method definitions and compatibility](docs/methods.md).

## Installation

Use Python 3.11 or 3.12 and run commands from the repository root:

```bash
uv sync --locked
```

The base environment supports GMM, Many-well, Lennard–Jones, and Morse.
Alanine dipeptide in internal coordinates requires molecular dependencies and
external data:

```bash
uv sync --locked --extra molecular
```

The Cartesian atomic models additionally require `--extra chem`. For OpenMM
CUDA support on Linux, add `--extra cuda` alongside `--extra molecular`.
Select a PyTorch build appropriate to your compute environment. Dataset
requirements and evaluation boundaries are listed in
[reproducibility](docs/reproducibility.md).

## Run an experiment

List methods and inspect a command before starting training:

```bash
uv run python -m tr_asbs.train --list-methods
uv run python -m tr_asbs.train --method tr-sbs --experiment gmm_mode --dry-run
uv run python -m tr_asbs.check --experiment lj13
```

The checker composes the configuration and checks configured input files and
installed dependencies without creating a model or training. Its method
defaults to `tr-sbs`; missing files or dependencies return exit status 1. Use
`--skip-data` to check configuration and dependencies without checking data.
The launcher accepts Hydra overrides after the method and experiment:

```bash
uv run python -m tr_asbs.train --method tr-sbs --experiment gmm_mode \
  seed=0 name=gmm_tr_sbs_seed0 root=/path/to/results
```

Run full training on your remote compute environment. The launcher does not
submit scheduler jobs. Give each run a distinct `name` and an explicit output
`root`; existing research presets may contain cluster-specific defaults.
W&B logging is enabled in the training presets. Configure your account or use
`WANDB_MODE=offline` for local logging.

Other target presets include:

| Target | Experiment presets | External data |
| --- | --- | --- |
| GMM | `gmm_mode`, `gmm_dim` | None |
| Many-well | `mw5`, `mw5_delta` | None |
| Lennard–Jones | `lj13`, `lj38`, `lj55` | Reference samples for LJ13/LJ55 |
| Morse | `morse13_asbs`, `morse13_ce_fast`, `morse13_ce_mlp` | None; no equilibrium reference |
| Alanine dipeptide, internal coordinates | `ad_ic_gamma_asbs`, `ad_ic_reference` | ALDP trajectory |
| Alanine dipeptide, Cartesian atomic backend | `ala2_gamma_asbs_iw` | Topology, pretraining and evaluation trajectories; `molecular` + `chem` |
| Alanine dipeptide, flattened Cartesian backend | `ala2_mlp_gamma_asbs_iw` | Same molecular assets; `molecular` + `chem` |

A preset specifies the target and historical hyperparameters; the method is
selected separately. A runnable configuration does not establish that every
method–target combination has been validated. Use the retained experiment
recipes when reproducing a reported result. `ala2` remains a legacy base
configuration rather than a supported public launcher preset.

Launcher runs save `paper_method`, `method_id`, and `method_variant_of` in their
resolved configuration, preserving paper names alongside historical module and
result keys.

## Reproduce the GMM comparison

The [frozen six-method GMM report](reports/gmm_scorefree_six_methods/README.md)
contains committed aggregate data, figures, seeds, checkpoint-selection rules,
and source hashes. Recreate its figures with:

```bash
uv run python scripts/plot_gmm_six_method_grid.py
```

The historical Slurm recipes are retained in
[`scripts/slurm/`](scripts/slurm/):

- [ASBS-family mode grid](scripts/slurm/submit_gmm_mode_full_grid.sh).
- [ASBS-family dimension grid](scripts/slurm/submit_gmm_dim_full_grid.sh).
- [Score-free PIS/DIS grid](scripts/slurm/submit_gmm_variational_asbs_control_grid.sh).

Inspect each recipe with `--dry-run` before submitting it on the configured
cluster. These recipes preserve method-specific tuned parameters; a generic
launcher command is not a replacement for the full recipe.

This comparison uses a common score-free FourierMLP backbone and fixed GMM
targets. Path-step counts and training budgets differ across methods; the
report does not establish equal-compute performance.

## Repository layout

- `src/tr_asbs/`: training, samplers, targets, models, evaluation, and Hydra configurations.
- `tests/`: unit and bounded smoke checks.
- `scripts/`: experiment utilities, scheduler recipes, and result processing.
- `reports/gmm_scorefree_six_methods/`: frozen GMM summaries and figures.
- `docs/methods.md`: method names and historical entrypoint mapping.
- `docs/reproducibility.md`: data, output provenance, and comparison requirements.

Research extensions, development notes, and historical configurations are
retained when needed to interpret existing results. Alanine chart/reference
variants include diagnostic configurations whose sampling quality remains
unvalidated. The pre-cleanup source and historical launcher locations are
preserved in commit `94c93ab`.

## License

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Bundled components retain their source notices and separately included licenses.
