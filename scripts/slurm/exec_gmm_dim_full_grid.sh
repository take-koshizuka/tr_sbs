#!/bin/bash

# One run from the tuned gmm_dim grid. Use scripts/slurm/submit_gmm_dim_full_grid.sh to
# launch all 4 dimensions x 3 model seeds x 4 methods.

#SBATCH --job-name=gmm-dim-grid
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
DIMENSION="${GMM_DIMENSION:-}"
METHOD="${GMM_METHOD:-}"
GLOBAL_SEED="${GMM_SEED:-}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
source "${REPO_ROOT}/scripts/gmm_storage.sh"
RUN_ROOT="$(gmm_run_root "${REPO_ROOT}" gmm_dim_full_grid_best_v1)"

case "${DIMENSION}" in
  10|25|50|100) ;;
  *) echo "GMM_DIMENSION must be one of: 10, 25, 50, 100." >&2; exit 2 ;;
esac
case "${GLOBAL_SEED}" in
  0|1|2) ;;
  *) echo "GMM_SEED must be one of: 0, 1, 2." >&2; exit 2 ;;
esac
if [[ "${TARGET_SEED}" != "0" ]]; then
  echo "GMM_TARGET_SEED must remain 0 for a common target across model seeds." >&2
  exit 2
fi

case "${METHOD}" in
  asbs)
    TRAINING_MODULE="tr_asbs.experiment.train_asbs"
    METHOD_LABEL="asbs"
    SIGMA_MAX="1"
    METHOD_OVERRIDES=()
    ;;
  gamma_asbs)
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs"
    METHOD_LABEL="gamma_asbs"
    SIGMA_MAX="2"
    METHOD_OVERRIDES=("gamma_schedule=trust_region" "trust_region_bound=0.1")
    ;;
  gamma_asbs_iw_fixed)
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs_iw"
    METHOD_LABEL="gamma_asbs_iw_fixed"
    SIGMA_MAX="1"
    METHOD_OVERRIDES=("gamma_schedule=fixed")
    ;;
  gamma_asbs_iw_trust_region)
    TRAINING_MODULE="tr_asbs.experiment.train_gamma_asbs_iw"
    METHOD_LABEL="gamma_asbs_iw_trust_region_0p1"
    SIGMA_MAX="2"
    METHOD_OVERRIDES=("gamma_schedule=trust_region" "trust_region_bound=0.1")
    ;;
  *)
    echo "Unknown GMM_METHOD: ${METHOD}." >&2
    exit 2
    ;;
esac

RUN_NAME="gmm_dim_d${DIMENSION}_${METHOD_LABEL}_seed${GLOBAL_SEED}"
TAGS="[gmm,gmm_dim,full_grid,best_params,d${DIMENSION},${METHOD_LABEL},seed${GLOBAL_SEED},target_seed0]"

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
else
  echo "Project Python was not found at ${REPO_ROOT}/.venv/bin/python." >&2
  exit 2
fi

echo "Starting tuned gmm_dim full-grid run"
echo "components=10 dim=${DIMENSION} method=${METHOD_LABEL} sigma_max=${SIGMA_MAX}"
echo "model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED}"
echo "run_name=${RUN_NAME} run_root=${RUN_ROOT}"
hostname
nvidia-smi

TRAIN_COMMAND=("${PYTHON_COMMAND[@]}" -u -m "${TRAINING_MODULE}" \
  "experiment=gmm_dim" \
  "name=${RUN_NAME}" \
  "seed=${GLOBAL_SEED}" \
  "+potential.seed=${TARGET_SEED}" \
  "potential.num_components=10" \
  "potential.dim=${DIMENSION}" \
  "base_sde.sigma_max=${SIGMA_MAX}" \
  "max_outer_loops=150" \
  "visualize=true" \
  "monitor_initial_state=false" \
  "divergence_early_stop.enabled=true" \
  "fabric.loggers.tags=${TAGS}" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}" \
  "progress_bar=false")
if ((${#METHOD_OVERRIDES[@]})); then
  TRAIN_COMMAND+=("${METHOD_OVERRIDES[@]}")
fi
TRAIN_COMMAND+=("$@")
"${TRAIN_COMMAND[@]}"
