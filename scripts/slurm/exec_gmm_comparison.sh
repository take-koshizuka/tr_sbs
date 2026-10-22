#!/bin/bash

# One run from the hardest-setting GMM comparison grid.
# Use scripts/slurm/submit_gmm_comparison.sh to launch all ten runs.

#SBATCH --job-name=gmm-comparison
#SBATCH --mem=64000
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=htc-cmp[101-148,501-532]
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

source ~/.profile

if [[ -n "${TR_ASBS_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${TR_ASBS_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Repository not found at ${REPO_ROOT}." >&2
  exit 2
fi

EXPERIMENT="${GMM_EXPERIMENT:-}"
METHOD="${GMM_METHOD:-}"
GLOBAL_SEED="${GMM_SEED:-0}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_SUFFIX="${GMM_RUN_SUFFIX:-}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_comparison_150_seed0}"
TRUST_REGION_BOUND="${GMM_TRUST_REGION_BOUND:-}"

if [[ ! "${GLOBAL_SEED}" =~ ^[0-9]+$ ]]; then
  echo "GMM_SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "${TARGET_SEED}" =~ ^[0-9]+$ ]]; then
  echo "GMM_TARGET_SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "${RUN_SUFFIX}" && ! "${RUN_SUFFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "GMM_RUN_SUFFIX must contain only letters, digits, underscores, or hyphens." >&2
  exit 2
fi

case "${EXPERIMENT}" in
  gmm_mode)
    DIFFICULTY_LABEL="k40"
    EXPERIMENT_TAG="mode40"
    CONFIG_OVERRIDES=(
      "potential.num_components=40"
      "potential.dim=10"
    )
    ;;
  gmm_dim)
    DIFFICULTY_LABEL="d100"
    EXPERIMENT_TAG="dim100"
    CONFIG_OVERRIDES=(
      "potential.dim=100"
      "potential.num_components=10"
    )
    ;;
  *)
    echo "GMM_EXPERIMENT must be one of: gmm_mode, gmm_dim." >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  asbs)
    TRAINING_MODULE="tr_asbs.experiment.train_asbs"
    METHOD_LABEL="asbs"
    METHOD_TAG="asbs"
    METHOD_OVERRIDES=()
    ;;
  gamma_asbs)
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs"
    METHOD_LABEL="gamma_asbs"
    METHOD_TAG="gamma_asbs"
    METHOD_OVERRIDES=("gamma_schedule=trust_region")
    ;;
  gamma_asbs_iw_fixed)
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs_iw"
    METHOD_LABEL="gamma_asbs_iw_fixed"
    METHOD_TAG="iw_fixed"
    METHOD_OVERRIDES=("gamma_schedule=fixed")
    ;;
  gamma_asbs_iw_trust_region)
    if [[ "${TRUST_REGION_BOUND}" != "0.1" && "${TRUST_REGION_BOUND}" != "0.3" ]]; then
      echo "GMM_TRUST_REGION_BOUND must be 0.1 or 0.3 for IW trust-region." >&2
      exit 2
    fi
    BOUND_LABEL="${TRUST_REGION_BOUND/./p}"
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs_iw"
    METHOD_LABEL="gamma_asbs_iw_trust_region_${BOUND_LABEL}"
    METHOD_TAG="iw_trust_region_${BOUND_LABEL}"
    METHOD_OVERRIDES=("gamma_schedule=trust_region" "trust_region_bound=${TRUST_REGION_BOUND}")
    ;;
  *)
    echo "GMM_METHOD must be one of: asbs, gamma_asbs, gamma_asbs_iw_fixed, gamma_asbs_iw_trust_region." >&2
    exit 2
    ;;
esac

RUN_NAME="${EXPERIMENT}_${DIFFICULTY_LABEL}_${METHOD_LABEL}_seed${GLOBAL_SEED}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
fi
TAGS="[gmm,comparison,${EXPERIMENT_TAG},${METHOD_TAG},seed${GLOBAL_SEED}]"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${TR_ASBS_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("${TR_ASBS_PYTHON}")
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_COMMAND=("${REPO_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_COMMAND=(uv run python)
else
  echo "No project Python environment was found." >&2
  exit 2
fi

echo "Starting GMM comparison run"
echo "experiment=${EXPERIMENT} difficulty=${DIFFICULTY_LABEL} method=${METHOD}"
echo "global_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED}"
echo "run_name=${RUN_NAME}"
echo "repository=${REPO_ROOT}"
echo "run_root=${RUN_ROOT}"
hostname
nvidia-smi

"${PYTHON_COMMAND[@]}" -u -m "${TRAINING_MODULE}" \
  "experiment=${EXPERIMENT}" \
  "name=${RUN_NAME}" \
  "seed=${GLOBAL_SEED}" \
  "+potential.seed=${TARGET_SEED}" \
  "fabric.loggers.tags=${TAGS}" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}" \
  "progress_bar=false" \
  "visualize=true" \
  "max_outer_loops=150" \
  "monitor_initial_state=false" \
  "${CONFIG_OVERRIDES[@]}" \
  "${METHOD_OVERRIDES[@]}" \
  "$@"
