#!/bin/bash

# One DIS run from the tuned GMM mode or dimension grid.

#SBATCH --job-name=gmm-dis-grid
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
GLOBAL_SEED="${GMM_SEED:-}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_dis_full_grid_best_v1}"

case "${GLOBAL_SEED}" in
  0|1|2) ;;
  *) echo "GMM_SEED must be one of: 0, 1, 2." >&2; exit 2 ;;
esac
if [[ "${TARGET_SEED}" != "0" ]]; then
  echo "GMM_TARGET_SEED must remain 0 for a common target across model seeds." >&2
  exit 2
fi

case "${SUITE}" in
  mode)
    case "${GRID_VALUE}" in
      5|10|20|30|40) ;;
      *) echo "Mode grid value must be one of: 5, 10, 20, 30, 40." >&2; exit 2 ;;
    esac
    EXPERIMENT="gmm_mode"
    DIFFICULTY_LABEL="k${GRID_VALUE}"
    SIGMA_MAX="2"
    CONFIG_OVERRIDES=("potential.num_components=${GRID_VALUE}" "potential.dim=10")
    ;;
  dim)
    case "${GRID_VALUE}" in
      10|25|50|100) ;;
      *) echo "Dimension grid value must be one of: 10, 25, 50, 100." >&2; exit 2 ;;
    esac
    EXPERIMENT="gmm_dim"
    DIFFICULTY_LABEL="d${GRID_VALUE}"
    SIGMA_MAX="4"
    CONFIG_OVERRIDES=("potential.num_components=10" "potential.dim=${GRID_VALUE}")
    ;;
  *) echo "GMM_SUITE must be mode or dim." >&2; exit 2 ;;
esac

RUN_NAME="${EXPERIMENT}_${DIFFICULTY_LABEL}_dis_seed${GLOBAL_SEED}"
TAGS="[gmm,${EXPERIMENT},full_grid,best_params,${DIFFICULTY_LABEL},dis,sigma${SIGMA_MAX},seed${GLOBAL_SEED},target_seed0]"

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
  echo "Project Python not found." >&2
  exit 2
fi

TRAIN_COMMAND=("${PYTHON_COMMAND[@]}" -u -m tr_asbs.experiment.train_variational
  "experiment=${EXPERIMENT}"
  "algorithm=dis"
  "name=${RUN_NAME}"
  "run_name=${RUN_NAME}"
  "seed=${GLOBAL_SEED}"
  "+potential.seed=${TARGET_SEED}"
  "base_sde.sigma_max=${SIGMA_MAX}"
  "algorithm.iters=40000"
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
TRAIN_COMMAND+=("${CONFIG_OVERRIDES[@]}")
TRAIN_COMMAND+=("$@")

echo "suite=${SUITE} experiment=${EXPERIMENT} setting=${DIFFICULTY_LABEL}"
echo "method=dis sigma_max=${SIGMA_MAX} model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED}"
echo "budget=40000 iterations; 150 training evaluations plus initial evaluation"
echo "final checkpoint report=4000 samples with report_seed=12345"
echo "early_stop=min_evals20, SD>2x best and TVD>best+0.15 for 8 evaluations; nonfinite=2"
echo "run_name=${RUN_NAME} output=${RUN_ROOT}/${RUN_NAME}"
printf 'command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
hostname
nvidia-smi

"${TRAIN_COMMAND[@]}"
