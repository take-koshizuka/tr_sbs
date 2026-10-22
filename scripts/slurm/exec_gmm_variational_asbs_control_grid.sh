#!/bin/bash

# One score-free PIS/DIS run using the exact GMM-ASBS FourierMLP controller.

#SBATCH --job-name=gmm-var-asbsctl
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
SUITE="${GMM_SUITE:-}"
GRID_VALUE="${GMM_GRID_VALUE:-}"
METHOD="${GMM_METHOD:-}"
GLOBAL_SEED="${GMM_SEED:-}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
source "${REPO_ROOT}/scripts/gmm_storage.sh"
RUN_ROOT="$(gmm_run_root "${REPO_ROOT}" gmm_variational_asbs_control_scoreless_v1)"

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
      *) echo "Mode grid value must be one of: 5, 10, 20, 30, 40." >&2; exit 2 ;;
    esac
    EXPERIMENT="gmm_mode"
    DIFFICULTY_LABEL="k${GRID_VALUE}"
    DIMENSION="10"
    NUM_COMPONENTS="${GRID_VALUE}"
    if [[ "${METHOD}" == "dis" ]]; then SIGMA_MAX="2"; else SIGMA_MAX="128"; fi
    ;;
  dim)
    case "${GRID_VALUE}" in
      10|25|50|100) ;;
      *) echo "Dimension grid value must be one of: 10, 25, 50, 100." >&2; exit 2 ;;
    esac
    EXPERIMENT="gmm_dim"
    DIFFICULTY_LABEL="d${GRID_VALUE}"
    DIMENSION="${GRID_VALUE}"
    NUM_COMPONENTS="10"
    if [[ "${METHOD}" == "dis" ]]; then SIGMA_MAX="4"; else SIGMA_MAX="32"; fi
    ;;
  *) echo "GMM_SUITE must be mode or dim." >&2; exit 2 ;;
esac

RUN_NAME="gmm_${SUITE}_${DIFFICULTY_LABEL}_${METHOD}_asbs_control_no_score_seed${GLOBAL_SEED}"
TAGS="[gmm,gmm_${SUITE},full_grid,fair_asbs_control,score_hint_off,${METHOD},${DIFFICULTY_LABEL},sigma${SIGMA_MAX},seed${GLOBAL_SEED},target_seed0]"

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
  echo "Project Python not found at ${REPO_ROOT}/.venv/bin/python." >&2
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
  "base_sde.sigma_max=${SIGMA_MAX}"
  "algorithm.iters=40000"
  "algorithm.model._target_=tr_asbs.variational.ASBSFourierControl"
  "algorithm.model.num_steps=128"
  "algorithm.model.num_layers=4"
  "algorithm.model.channels=256"
  "algorithm.model.clip_norm=100.0"
  "algorithm.model.score_hint_mode=none"
  "evaluation.num_evaluations=150"
  "evaluation.num_samples=2000"
  "evaluation.final_num_samples=4000"
  "evaluation.report_seed=12345"
  "evaluation.report_checkpoints=true"
  "evaluation.evaluate_initial_state=true"
  "visualize=true"
  "monitor_initial_state=false"
  "divergence_early_stop.enabled=true"
  "divergence_early_stop.primary_metric=loss/SD"
  "divergence_early_stop.secondary_metric=loss/tvd_soft"
  "divergence_early_stop.min_evals=20"
  "divergence_early_stop.primary_factor=2.0"
  "divergence_early_stop.secondary_delta=0.15"
  "divergence_early_stop.patience=8"
  "divergence_early_stop.nonfinite_patience=2"
  "fabric.loggers.tags=${TAGS}"
  "root=${RUN_ROOT}"
  "hydra.run.dir=${RUN_ROOT}/${RUN_NAME}"
  "progress_bar=false"
)
TRAIN_COMMAND+=("$@")

echo "suite=${SUITE} setting=${DIFFICULTY_LABEL} method=${METHOD} sigma_max=${SIGMA_MAX}"
echo "model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED} score_hint=none"
echo "control=ASBSFourierControl/FourierMLP layers=4 channels=256 clip_norm=100"
echo "budget=40000 iterations; path_steps=128; evaluations=150; final_samples=4000"
echo "run_name=${RUN_NAME} output=${RUN_ROOT}/${RUN_NAME}"
printf 'command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
hostname
nvidia-smi

"${TRAIN_COMMAND[@]}"
