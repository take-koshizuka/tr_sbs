# Method names and implementation mapping

Public method names describe the update rule. Experiment names describe the
target and a saved configuration. Historical Python module names, configuration
filenames, checkpoint metadata, and aggregate keys are retained so that existing
results can still be loaded and traced to their original implementations.

The launcher adds `paper_method`, `method_id`, and `method_variant_of` to the
saved configuration. For TR-ASBS, the family metadata identifies Gamma-ASBS;
historical result keys are unchanged.

| Public name | `--method` | Historical entrypoint | Method setting |
| --- | --- | --- | --- |
| ASBS | `asbs` | `tr_asbs.experiment.train_asbs` | Baseline adjoint/corrector matching |
| Gamma-ASBS | `gamma-asbs` | `tr_asbs.experiment.train_gamma_asbs` | `gamma_schedule=fixed` |
| TR-ASBS | `tr-asbs` | `tr_asbs.experiment.train_gamma_asbs` | `gamma_schedule=trust_region` |
| ASBS-IW | `asbs-iw` | `tr_asbs.experiment.train_gamma_asbs_iw` | `gamma_schedule=fixed` |
| TR-SBS | `tr-sbs` | `tr_asbs.experiment.train_gamma_asbs_iw` | `gamma_schedule=trust_region` |
| PIS | `pis` | `tr_asbs.experiment.train_variational` | `algorithm=pis` |
| DIS | `dis` | `tr_asbs.experiment.train_variational` | `algorithm=dis` |

## ASBS family

ASBS alternates the learning of a forward control by adjoint matching and a
backward corrector by corrector matching. Gamma-ASBS introduces an interpolation
parameter γ for the update. Its fixed-γ form is called **Gamma-ASBS** here;
**TR-ASBS is its trust-region variant**, using adaptive γ without the
importance-weighted regression used by TR-SBS.

ASBS-IW uses importance weights with a fixed γ. TR-SBS uses importance weights
and chooses γ from the trust-region dual problem. Thus the naming changes are:

- `IW-ASBS (fixed γ)` → **ASBS-IW**.
- `IW-ASBS: TR`, `TR-IW`, or the trust-region IW trainer → **TR-SBS**.
- `Gamma-ASBS` remains the family name; **TR-ASBS** explicitly identifies its
  trust-region variant.

For fixed-γ methods, provide `gamma=<value>` when the intended value differs
from the target preset. Some historical presets set `gamma: 1.0`; the method
name alone does not imply a smaller interpolation parameter. Preserve the
original γ and other hyperparameters when reproducing a run.

The IW implementation also contains optional forward losses, corrector losses,
reference processes, solvers, and damping settings. These are additional
configuration choices, not implied by the TR-SBS name. In particular,
`cross_entropy`, `log_variance`, and `path_kl` are distinct forward objectives;
record the resolved configuration when comparing them. General backward-loss
experiments retain their separate `train_general_asbs_iw` entrypoint.

## PIS and DIS

PIS and DIS share a variational trainer and minimize their path-space ELBO
objectives. PIS starts at zero. DIS uses a Gaussian initial distribution with
scale taken from the experiment configuration and a reverse-OU formulation.
Their fixed cosine schedules reuse scalar diffusion bounds from the target
presets; they do not instantiate the ASBS source and SDE objects.

The retained GMM six-method recipe uses `ASBSFourierControl` with no target-score
hint for PIS and DIS. Other variational presets may use different controller
inputs or architectures. The method label alone does not guarantee input,
parameter-count, or compute parity.

## Historical results

The frozen GMM report maps stored keys as follows:

| Stored key | Reported name |
| --- | --- |
| `asbs` | ASBS |
| `gamma_asbs` | TR-ASBS in the retained trust-region GMM grid |
| `gamma_asbs_iw_fixed` | ASBS-IW |
| `gamma_asbs_iw_trust_region_*` | TR-SBS |
| `pis`, `dis` | PIS, DIS |

Do not rename frozen CSV keys, run directories, or checkpoint fields to change
their display names. The key `gamma_asbs` identifies a trainer family; confirm
`gamma_schedule` in the run configuration before calling an arbitrary run
TR-ASBS. See the [frozen report](../reports/gmm_scorefree_six_methods/README.md)
for the exact GMM mapping and comparison scope.

The public atomic backend supports `ala2_gamma_asbs_iw` with ASBS-IW or TR-SBS
and dispatches to `train_gamma_asbs_iw_ala2`. `ala2` remains the legacy base
configuration; the public launcher does not support it or the older `ala4`
preset. Historical entrypoints remain available.

`ala2_mlp_gamma_asbs_iw` uses flattened Cartesian coordinates and the flat
trainer. Both Cartesian paths need the `molecular` and `chem` extras and differ
from the IC60 `ad_ic_*` presets. Preserve those distinctions when reporting or
reproducing alanine results.
