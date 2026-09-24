# GMM mode and dimension sweep: six score-free methods

This is the frozen three-seed comparison used by
`scripts/plot_gmm_six_method_grid.py`. Each row uses the same GMM target
(`potential.seed=0`), model seeds 0/1/2, and an independent 4,000-sample
checkpoint evaluation (`report_seed=12345`). The main figures select the
checkpoint with the lowest *online* SD and show the independent evaluation
mean ± sample standard deviation across seeds. Smaller SD, TVD, W2, and
sliced W2 are better; larger raw path ESS is better.

| Suite | Values | Fixed counterpart | Methods |
| --- | --- | --- | --- |
| mode | K = 5, 10, 20, 30, 40 | d = 10 | ASBS, TR-ASBS, ASBS-IW, TR-SBS, PIS, DIS |
| dim | d = 10, 25, 50, 100 | K = 10 | Same six methods |

PIS and DIS use `ASBSFourierControl`, which inherits the ASBS `FourierMLP`
backbone (four layers, width 256) and does not evaluate or ingest the target
score. The forward controller therefore has the same parameter count as the
ASBS controller at a given dimension. The ASBS variants with a corrector have
an additional network. The historical ASBS-family grid uses 200 path steps;
PIS and DIS use 128, and their optimizer-update budgets also differ. The
figures compare the completed experiments; they do not claim equal total
compute.

The colors and markers follow `tr_asbs_wandb/make_plot.py` where that program
fixes them: ASBS blue/circle, ASBS-IW orange/square, TR-SBS red/diamond.
TR-ASBS, PIS, and DIS use distinct tab10 colors and markers. The plot labels
TR-ASBS, ASBS-IW, and TR-SBS correspond to the stored method keys
`gamma_asbs`, `gamma_asbs_iw_fixed`, and `gamma_asbs_iw_trust_region_*`.
SD and raw path ESS use logarithmic y axes; if a lower error bar would cross zero, the
displayed lower endpoint is clipped to the smallest positive value. Means and
sample standard deviations in the CSVs are unchanged.

## Reproduce the plots

From the repository root:

```bash
.venv/bin/python scripts/plot_gmm_six_method_grid.py
```

The script requires Matplotlib and writes five metric plots for each suite as
PDF and PNG under `figures/min_online_sd/` (boxed legend) and
`figures/min_online_sd/legend_no_box/` (frameless legend). Both legends are
placed inside the axes. SD/TVD panels use compact layouts in open bands where a
lower-right legend would obscure TR-SBS. To inspect stability or ESS-based
selection, pass `--selection final` or `--selection max_online_raw_ess`.

## Frozen input provenance

- `data/asbs_mode_aggregate.csv`: z3
  `outputs/gmm_mode_full_grid_best_v1_aggregate.csv`, SHA-256
  `60b8c4c241158c5ffb98a1aa2498f60141375cf3e577444932db4378793c0a9c`.
- `data/asbs_dim_aggregate.csv`: z3
  `outputs/gmm_dim_full_grid_best_v1_aggregate.csv`, SHA-256
  `dc810e3fec7d1ecb1bbbc859ea0bf80760c8bdf29d20a3a1ecc883df610ff43b`.
- `data/pis_dis_aggregate.csv`: z3
  `runs/gmm_variational_asbs_control_scoreless_v1/aggregation/aggregate_metrics.csv`,
  SHA-256
  `3766a746123ed73f9a454b6f61cca2b90510c4adbf3662a0895264bb28e4ea02`.

The full checkpoint evaluations and Slurm logs remain in the remote run
directory. The committed CSVs and figures are small, reproducible summaries;
model checkpoints and W&B media are not committed.
