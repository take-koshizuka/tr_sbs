#!/bin/bash

# Run one PIS/DIS member of the GMM mode/dimension grids. Submit the complete
# 54-job extension with submit_gmm_variational_grid.sh.

#SBATCH --job-name=gmm-var-grid
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
SUITE="${GMM_SUITE:-}"
GRID_VALUE="${GMM_GRID_VALUE:-}"
METHOD="${GMM_METHOD:-}"
GLOBAL_SEED="${GMM_SEED:-}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_variational_full_grid_v1}"

case "${GLOBAL_SEED}" in
  0|1|2) ;;
  *) echo "GMM_SEED must be one of: 0, 1, 2." >&2; exit 2 ;;
esac
if [[ "${TARGET_SEED}" != "0" ]]; then
  echo "GMM_TARGET_SEED must remain 0 for a common target across model seeds." >&2
  exit 2
fi
case "${METHOD}" in
  pis|dis) ;;
  *) echo "GMM_METHOD must be pis or dis." >&2; exit 2 ;;
esac

case "${SUITE}" in
  mode)
    case "${GRID_VALUE}" in
      5|10|20|30|40) ;;
      *) echo "For suite=mode, GMM_GRID_VALUE must be 5, 10, 20, 30, or 40." >&2; exit 2 ;;
    esac
    EXPERIMENT="gmm_mode"
    DIMENSION="10"
    NUM_COMPONENTS="${GRID_VALUE}"
    GRID_LABEL="k${NUM_COMPONENTS}"
    ;;
  dim)
    case "${GRID_VALUE}" in
      10|25|50|100) ;;
      *) echo "For suite=dim, GMM_GRID_VALUE must be 10, 25, 50, or 100." >&2; exit 2 ;;
    esac
    EXPERIMENT="gmm_dim"
    DIMENSION="${GRID_VALUE}"
    NUM_COMPONENTS="10"
    GRID_LABEL="d${DIMENSION}"
    ;;
  *)
    echo "GMM_SUITE must be mode or dim." >&2
    exit 2
    ;;
esac

RUN_NAME="gmm_${SUITE}_${GRID_LABEL}_${METHOD}_seed${GLOBAL_SEED}"
TAGS="[gmm,gmm_${SUITE},full_grid,variational,${METHOD},${GRID_LABEL},seed${GLOBAL_SEED},target_seed0]"

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

TRAIN_COMMAND=("${PYTHON_COMMAND[@]}" -u -m tr_asbs.experiment.train_variational
  "experiment=${EXPERIMENT}"
  "algorithm=${METHOD}"
  "name=${RUN_NAME}"
  "run_name=${RUN_NAME}"
  "seed=${GLOBAL_SEED}"
  "+potential.seed=${TARGET_SEED}"
  "potential.num_components=${NUM_COMPONENTS}"
  "potential.dim=${DIMENSION}"
  "visualize=true"
  "fabric.loggers.tags=${TAGS}"
  "root=${RUN_ROOT}"
  "hydra.run.dir=${RUN_ROOT}/${RUN_NAME}"
  "progress_bar=false"
)
TRAIN_COMMAND+=("$@")

echo "Starting GMM variational grid run"
echo "suite=${SUITE} components=${NUM_COMPONENTS} dim=${DIMENSION} method=${METHOD}"
echo "model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED}"
echo "run_name=${RUN_NAME} output=${RUN_ROOT}/${RUN_NAME}"
printf 'command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
hostname
nvidia-smi

"${TRAIN_COMMAND[@]}"
