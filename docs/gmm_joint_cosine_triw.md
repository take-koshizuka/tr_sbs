# Joint K=40, d=100 score-conditioned TR-IW cosine tuning

This campaign changes only the reference noise schedule. The controller and
corrector remain the 2-layer, width-64 score-conditioned networks, and all
paths use 128 transitions with the existing `ExponentialIntegrator`.
`PISCosineSDE` reuses PIS's cosine coefficients in reverse step order, with
zero reference drift. It integrates their squared values exactly as a
piecewise-linear cumulative variance. The same interval variance is used for
forward and backward noise and for both Gaussian path transition densities
in the IW ratio. The backward drift uses that interval's coefficient while
its controller is evaluated at the right endpoint. `VESDE` and its existing
configs retain their previous behavior.

PIS clips standard-normal noise at 4; TR-IW deliberately does not clip it.
Clipped noise would no longer have the Gaussian transition density assumed by
its importance weights. The schedule parameters, not clipped-noise laws, are
matched. PIS's `sigma_max` parameter is approximately twice the attained
peak coefficient because its schedule contains a factor of 0.5.

Two distinct targets are tuned separately, both with target seed 0:

| Suite | K | d | loc_scaling | source.scale | sigma_max grid | TR bound grid |
|---|---:|---:|---:|---:|---|---|
| dim | 40 | 100 | 7 | 5 | 8, 16, 32 | 0.03, 0.1, 0.3 |
| mode | 40 | 100 | 40 | 30 | 32, 64, 128 | 0.1, 0.3, 1.0 |

The first stage is the 18-run seed-0 Cartesian grid. The two best candidates
per suite by minimum-SD checkpoint advance to seeds 1 and 2 (8 further runs).
The final selection uses mean SD over three seeds; differences within the
combined standard error are decided by TVD-soft, W2, then raw path ESS.
Every completed run re-evaluates final, minimum-SD, and maximum-ESS
checkpoints with 4,000 samples and report seed 12345. Existing divergence
early stopping remains enabled for full runs.

On z3, use `scripts/monitor_gmm_joint_cosine_triw.py --advance` with explicit
`--repo-root`, `--run-root`, `--log-root`, and `--python`. Its first invocation
submits two bounded GPU smokes; subsequent invocations advance only after
their reports and sample artifacts validate, then after every stage-1 run
validates. Manifests in `--log-root` record Slurm IDs, configurations, output
directories, logs, and commit. Any failed job stops advancement; inspect its
log before deciding whether to launch a separate recovery run.
