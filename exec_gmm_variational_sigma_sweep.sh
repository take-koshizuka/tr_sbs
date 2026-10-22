#!/bin/bash

# One seed-0 hardest-setting sigma_max tuning run for PIS or DIS.
# The candidate grid and divergence guard match the prior ASBS sigma sweep.

#SBATCH --job-name=gmm-var-sigma
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
GLOBAL_SEED="${GMM_SEED:-0}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_variational_sigma_tuning_seed0_v1}"

case "${SIGMA_MAX}" in
  1|2|4|8) ;;
  *) echo "GMM_SIGMA_MAX must be one of: 1, 2, 4, 8." >&2; exit 2 ;;
esac
case "${METHOD}" in
  pis|dis) ;;
  *) echo "GMM_METHOD must be pis or dis." >&2; exit 2 ;;
esac
if [[ "${GLOBAL_SEED}" != "0" || "${TARGET_SEED}" != "0" ]]; then
  echo "Tuning uses model seed=0 and target seed=0." >&2
  exit 2
fi

case "${EXPERIMENT}" in
  gmm_mode)
    DIFFICULTY_LABEL="k40"
    CONFIG_OVERRIDES=("potential.num_components=40" "potential.dim=10")
    ;;
  gmm_dim)
    DIFFICULTY_LABEL="d100"
    CONFIG_OVERRIDES=("potential.num_components=10" "potential.dim=100")
    ;;
  *) echo "GMM_EXPERIMENT must be gmm_mode or gmm_dim." >&2; exit 2 ;;
esac

RUN_NAME="tune_${EXPERIMENT}_${DIFFICULTY_LABEL}_${METHOD}_smax${SIGMA_MAX}_seed0"
TAGS="[gmm,sigma_max_tuning,variational,${DIFFICULTY_LABEL},${METHOD},sigma${SIGMA_MAX},seed0,target_seed0]"

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
  "algorithm=${METHOD}"
  "name=${RUN_NAME}"
  "run_name=${RUN_NAME}"
  "seed=${GLOBAL_SEED}"
  "+potential.seed=${TARGET_SEED}"
  "base_sde.sigma_max=${SIGMA_MAX}"
  "algorithm.iters=40000"
  "evaluation.num_evaluations=150"
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

echo "experiment=${EXPERIMENT} difficulty=${DIFFICULTY_LABEL} method=${METHOD} sigma_max=${SIGMA_MAX}"
echo "budget=40000 iterations; 150 training evaluations plus initial evaluation"
echo "early_stop=min_evals20, SD>2x best and TVD>best+0.15 for 8 evaluations; nonfinite=2"
echo "run_name=${RUN_NAME} output=${RUN_ROOT}/${RUN_NAME}"
printf 'command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
hostname
nvidia-smi

"${TRAIN_COMMAND[@]}"
