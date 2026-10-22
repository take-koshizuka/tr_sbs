#!/bin/bash

# One seed-0 hardest-setting pilot for Gamma-ASBS or Gamma-ASBS-IW fixed.

#SBATCH --job-name=gmm-pilot
#SBATCH --mem=64000
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=htc-cmp[101-148,501-532]
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source ~/.profile

REPO_ROOT="${TR_ASBS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
EXPERIMENT="${GMM_EXPERIMENT:-}"
METHOD="${GMM_METHOD:-}"
SIGMA_MAX="${GMM_SIGMA_MAX:-}"
TR_BOUND="${GMM_TRUST_REGION_BOUND:-}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_gamma_fixed_pilot_seed0}"

case "${SIGMA_MAX}" in 1|2|4) ;; *) echo "GMM_SIGMA_MAX must be 1, 2, or 4." >&2; exit 2;; esac
case "${EXPERIMENT}" in
  gmm_mode)
    LABEL="k40"
    CONFIG_OVERRIDES=("potential.num_components=40" "potential.dim=10")
    ;;
  gmm_dim)
    LABEL="d100"
    CONFIG_OVERRIDES=("potential.num_components=10" "potential.dim=100")
    ;;
  *) echo "GMM_EXPERIMENT must be gmm_mode or gmm_dim." >&2; exit 2;;
esac

case "${METHOD}" in
  gamma_asbs)
    case "${TR_BOUND}" in 0.1|0.3) ;; *) echo "Gamma-ASBS requires bound 0.1 or 0.3." >&2; exit 2;; esac
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs"
    METHOD_LABEL="gamma_tr_${TR_BOUND/./p}"
    METHOD_OVERRIDES=("gamma_schedule=trust_region" "trust_region_bound=${TR_BOUND}")
    ;;
  iw_fixed)
    [[ -z "${TR_BOUND}" ]] || { echo "IW fixed does not use a trust-region bound." >&2; exit 2; }
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs_iw"
    METHOD_LABEL="iw_fixed"
    METHOD_OVERRIDES=("gamma_schedule=fixed")
    ;;
  *) echo "GMM_METHOD must be gamma_asbs or iw_fixed." >&2; exit 2;;
esac

RUN_NAME="pilot_${EXPERIMENT}_${LABEL}_${METHOD_LABEL}_smax${SIGMA_MAX}_seed0"
TAGS="[gmm,pilot_tuning,${LABEL},${METHOD_LABEL},sigma${SIGMA_MAX},seed0]"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_COMMAND=("${TR_ASBS_PYTHON:-${REPO_ROOT}/.venv/bin/python}")
[[ -x "${PYTHON_COMMAND[0]}" ]] || { echo "Project Python not found." >&2; exit 2; }

echo "experiment=${EXPERIMENT} method=${METHOD_LABEL} sigma_max=${SIGMA_MAX}"
"${PYTHON_COMMAND[@]}" -u -m "${TRAINING_MODULE}" \
  "experiment=${EXPERIMENT}" \
  "name=${RUN_NAME}" \
  "seed=0" \
  "+potential.seed=0" \
  "base_sde.sigma_max=${SIGMA_MAX}" \
  "max_outer_loops=150" \
  "visualize=true" \
  "monitor_initial_state=false" \
  "divergence_early_stop.enabled=true" \
  "fabric.loggers.tags=${TAGS}" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}" \
  "progress_bar=false" \
  "${CONFIG_OVERRIDES[@]}" \
  "${METHOD_OVERRIDES[@]}" \
  "$@"
