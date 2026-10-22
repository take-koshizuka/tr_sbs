# Reproducibility

Run commands from the repository root. Use the committed lockfile, retain the
resolved Hydra configuration and logs, and record the Git revision plus any
uncommitted changes. The public launcher chooses a method and forwards Hydra
overrides to the existing trainer; it does not replace the historical
experiment recipes or standardize their budgets.

## Inspect before training

```bash
uv sync --locked
uv run python -m tr_asbs.train --list-methods
uv run python -m tr_asbs.train --method tr-sbs --experiment gmm_mode --dry-run
uv run python -m tr_asbs.check --experiment lj13
```

`train --dry-run` inspects the selected launch command without training. The
separate `tr_asbs.check` command composes the configuration, checks installed
modules, and checks the existence of configured input files. It defaults to
`--method tr-sbs`; pass a method explicitly to inspect another method. Relative
input paths are resolved from the current directory, or from `--data-root`.

Missing files, missing dependencies, or an unsupported method–target combination
return exit status 1. A configuration/argument error returns exit status 2.
Use `--skip-data` for configuration and dependency checks without input-file
checks:

```bash
uv run python -m tr_asbs.check --experiment lj13 --skip-data
```

Neither command creates a model, trains, evaluates an energy, or verifies dataset
contents, OpenMM execution, numerical stability, or convergence. For a resolved
Hydra configuration, use the trainer's Hydra inspection flags,
for example:

```bash
uv run python -m tr_asbs.experiment.train_gamma_asbs_iw \
  experiment=gmm_mode gamma_schedule=trust_region --cfg job --resolve
```

Submit full experiments through your remote scheduler. Record the job ID,
command, revision, output root, and log path together. Give each run a unique
name and use an explicit `root=/path/to/results`; several retained presets
were written for a particular cluster. Periodic monitoring should distinguish
queue state, process completion, successful artifact generation, and scientific
evaluation.

W&B is enabled in the training presets. Set its account/project fields for your
environment or set `WANDB_MODE=offline`. Keep offline logs with the other run
artifacts. PIS/DIS and ASBS-family trainers have different output layouts, so
retain the actual output path printed by the run.

## Target presets and external data

Large datasets, trajectories, checkpoints, and run outputs are excluded from
Git. A preset referring to these files requires the corresponding external
asset. The public source does not include a general downloader for the data
listed below; obtain the original datasets and record their source, version,
size, checksum, and split before using them. Matching a filename alone does not
establish matching data.

| Target | Presets | Required external files or evaluation boundary |
| --- | --- | --- |
| GMM | `gmm_mode`, `gmm_dim` | No dataset; record `potential.seed` separately from the training `seed` |
| Many-well | `mw5`, `mw5_delta` | No external dataset; preserve dimension, double-well coordinate count, and separation |
| LJ13 | `lj13` and variants | `data/test_split_LJ13-1000.npy` for reference-distribution metrics |
| LJ38 | `lj38` and variants | Uses `LennardJonesEnergyNoGT`; energy/path diagnostics, without an equilibrium reference |
| LJ55 | `lj55` and variants | `data/test_split_LJ55-1000-part1.npy` for reference-distribution metrics |
| Morse13 | `morse13_asbs`, `morse13_ce_fast`, `morse13_ce_mlp` | Uses `MorseEnergyNoGT`; no equilibrium reference is supplied |
| Alanine IC60 | `ad_ic_gamma_asbs`, `ad_ic_reference` | `data/alanine_dipeptide/val.h5` for transform calibration and validation; separate test trajectory for final evaluation |
| Alanine Cartesian | `ala2_gamma_asbs_iw`, `ala2_mlp_gamma_asbs_iw` | `md_data/ala2/ala2_ref.pdb`, `md_data/ala2/ala2_0.1ns.pdb`, and `data/val_split_AD.h5`; `molecular` + `chem` |

The LJ13/LJ55 reference arrays and Cartesian alanine assets are not distributed
with this repository. The configurations retain their expected paths rather
than substituting another target or silently omitting reference metrics.

### Many-well

`mw5` currently specifies dimension 64, `m=16`, and `delta=4`;
`mw5_delta` specifies dimension 8, `m=5`, and `delta=8`. They are distinct
experiments despite their related names. `mw5_delta_gl` uses a coupled
Ginzburg–Landau potential, so it is also a separate target rather than a
renamed independent Many-well experiment.

### Lennard–Jones and Morse

Preserve particle count, oscillator confinement, energy scaling, clipping,
soft-core options, mean removal, and the controller architecture. The retained
LJ presets include several such variants; their filenames are not sufficient
to establish the same equilibrium target.

Morse13 is a different target from LJ13. LJ13 reference samples cannot be used
as Morse equilibrium samples. LJ38 and Morse diagnostics without ground truth
do not establish accurate equilibrium coverage. Reference-based and
ground-truth-free evaluations should remain identifiable in aggregated results.

### Alanine dipeptide

Install the molecular extra before using the OpenMM/trajectory pipeline:

```bash
uv sync --locked --extra molecular
```

The Cartesian atomic models also require the `chem` extra:

```bash
uv sync --locked --extra molecular --extra chem
```

Add `--extra cuda` for the optional Linux OpenMM CUDA plugin. Experimental JAX
inner-loop diagnostics require `--extra jax`; standard PyTorch training does
not require that extra.

Record OpenMM platform, device, precision, molecular system, temperature, energy
definition, coordinate transform, units, and external trajectory provenance.
The IC path uses 60 native internal coordinates for a 22-atom molecule with 66
Cartesian coordinates. The default FAB system configuration uses 300 K and an
implicit environment. Physical and regularized energy modes are distinct;
their calibration banks and results must not be mixed.

`ad_ic_gamma_asbs` uses 4,096 transform-calibration frames from `val.h5`
(stride 100, offset 0), and 4,096 reference frames from a disjoint range
(stride 100, offset 500,000). Retain these ranges or explicitly record a changed
protocol. Keep `test.h5` for frozen final evaluation rather than transform
calibration or hyperparameter selection.

`ad_ic_reference` adds a smooth coordinate chart and modular reference/solver
configuration. The retained chart/reference diagnostics establish pipeline
execution for tested cases; their sampling quality remains unvalidated. Preserve
failed quality gates and `not_evaluated` results when summarizing these variants.

The Cartesian path uses topology and short pretraining trajectories
separately from its evaluation trajectory. The 0.1 ns PDB is warm-start data,
not an equilibrium evaluation reference. Confirm atom ordering, topology, and
coordinate units before supplying or overriding any trajectory path. The IC
and Cartesian implementations must not be treated as interchangeable recipes.
The public atomic launcher accepts `ala2_gamma_asbs_iw` with ASBS-IW/TR-SBS.
`ala2_mlp_gamma_asbs_iw` uses the flat backend. The legacy `ala2` base config and
older `ala4` preset are retained but are not public launcher targets.

## Frozen GMM six-method comparison

The [report](../reports/gmm_scorefree_six_methods/README.md) is the authoritative
record of the committed aggregate inputs and figure reproduction. Preserve
those files and their historical method keys. Method display names are
documented separately in [methods.md](methods.md).

- Mode sweep: component counts 5, 10, 20, 30, 40 at dimension 10.
- Dimension sweep: dimensions 10, 25, 50, 100 at 10 components.
- Common target seed: `potential.seed=0`; training seeds: 0, 1, 2.
- Main checkpoint selection: minimum online SD; reporting uses an independent
  4,000-sample evaluation with `report_seed=12345`.
- Error bars: sample standard deviation across training seeds.
- PIS/DIS use the score-free `ASBSFourierControl`; ASBS variants with a corrector
  retain an additional network.
- Historical ASBS-family runs use 200 path steps, while PIS/DIS use 128.
  Optimizer-update budgets also differ.

Reproduce the figures without retraining:

```bash
uv run python scripts/plot_gmm_six_method_grid.py
```

The report contains hashes for the aggregate inputs. Full training reruns
require the retained submission recipes and their method-specific overrides;
a generic launcher command using only a target preset is not an exact
reproduction of the tuned grid. The figure script also supports
`--selection final` and `--selection max_online_raw_ess` for the retained
alternatives. The scheduler recipes now live in `scripts/slurm/`; pre-cleanup
paths and code are preserved in commit `94c93ab`.

## Run artifacts and comparisons

Retain the resolved configuration, environment/lockfile, revision, seeds,
scheduler record, logs, final and selected checkpoints, evaluation samples when
saved, and checkpoint-evaluation JSON files. Keep target construction,
training randomness, and final-report randomness identifiable. A completed job
or a saved checkpoint is not evidence that sampling-quality criteria passed.
Launcher runs save `paper_method`, `method_id`, and `method_variant_of` with
the configuration, so display names and method-family relationships are recorded
without replacing historical checkpoint or aggregate keys.

Before comparing runs, align the target and energy definition, input features,
coordinate system, reference bank, evaluation sample count, checkpoint-selection
rule, and statistical unit. Report path steps, training updates, device,
runtime/cost boundaries, and any extra score evaluations. Preserve failed or
missing evaluations rather than dropping them from a headline mean.
