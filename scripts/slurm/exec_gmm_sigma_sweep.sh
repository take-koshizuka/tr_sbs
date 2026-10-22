#!/bin/bash

# One sigma_max tuning run for ASBS or the best prior IW-TR variant.
# Existing sigma_max=2 comparison runs are reused as the baseline.

#SBATCH --job-name=gmm-sigma
#SBATCH --mem=64000
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=htc-cmp[101-148,501-532]
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source ~/.profile

REPO_ROOT="${TR_ASBS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
EXPERIMENT="${GMM_EXPERIMENT:-}"
METHOD="${GMM_METHOD:-}"
SIGMA_MAX="${GMM_SIGMA_MAX:-}"
GLOBAL_SEED="${GMM_SEED:-0}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_SUFFIX="${GMM_RUN_SUFFIX:-sigma_tune}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_sigma_max_tuning_seed0}"

case "${SIGMA_MAX}" in
  1|4|8) ;;
  *) echo "GMM_SIGMA_MAX must be one of: 1, 4, 8." >&2; exit 2 ;;
esac

case "${EXPERIMENT}" in
  gmm_mode)
    DIFFICULTY_LABEL="k40"
    CONFIG_OVERRIDES=("potential.num_components=40" "potential.dim=10")
    IW_BOUND="0.3"
    ;;
  gmm_dim)
    DIFFICULTY_LABEL="d100"
    CONFIG_OVERRIDES=("potential.dim=100" "potential.num_components=10")
    IW_BOUND="0.1"
    ;;
  *) echo "GMM_EXPERIMENT must be gmm_mode or gmm_dim." >&2; exit 2 ;;
esac

case "${METHOD}" in
  asbs)
    TRAINING_MODULE="tr_asbs.experiment.train_asbs"
    METHOD_LABEL="asbs"
    METHOD_OVERRIDES=()
    ;;
  iw_tr)
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs_iw"
    METHOD_LABEL="iw_tr_${IW_BOUND/./p}"
    METHOD_OVERRIDES=("gamma_schedule=trust_region" "trust_region_bound=${IW_BOUND}")
    ;;
  *) echo "GMM_METHOD must be asbs or iw_tr." >&2; exit 2 ;;
esac

RUN_NAME="sigma_${EXPERIMENT}_${DIFFICULTY_LABEL}_${METHOD_LABEL}_smax${SIGMA_MAX}_seed${GLOBAL_SEED}_${RUN_SUFFIX}"
TAGS="[gmm,sigma_max_tuning,${DIFFICULTY_LABEL},${METHOD_LABEL},sigma${SIGMA_MAX},seed${GLOBAL_SEED}]"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${TR_ASBS_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("${TR_ASBS_PYTHON}")
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_COMMAND=("${REPO_ROOT}/.venv/bin/python")
else
  echo "Project Python not found." >&2; exit 2
fi

echo "experiment=${EXPERIMENT} method=${METHOD_LABEL} sigma_max=${SIGMA_MAX} iw_bound=${IW_BOUND}"
echo "Early stop: min_evals=20, SD>2x best and TVD>best+0.15 for 8 evaluations"

"${PYTHON_COMMAND[@]}" -u -m "${TRAINING_MODULE}" \
  "experiment=${EXPERIMENT}" \
  "name=${RUN_NAME}" \
  "seed=${GLOBAL_SEED}" \
  "+potential.seed=${TARGET_SEED}" \
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
