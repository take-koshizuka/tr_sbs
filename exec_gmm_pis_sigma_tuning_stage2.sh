#!/bin/bash

# One seed-0, hardest-setting PIS sigma_max stage-2 tuning run.

#SBATCH --job-name=gmm-pis-s2
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
SIGMA_MAX="${GMM_SIGMA_MAX:-}"
GLOBAL_SEED="${GMM_SEED:-0}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_pis_sigma_tuning_stage2_seed0_v1}"

if [[ "${GLOBAL_SEED}" != "0" || "${TARGET_SEED}" != "0" ]]; then
  echo "PIS stage-2 tuning uses model seed=0 and target seed=0." >&2
  exit 2
fi

case "${EXPERIMENT}" in
  gmm_mode)
    case "${SIGMA_MAX}" in
      32|64|96|128) ;;
      *) echo "gmm_mode sigma_max must be one of: 32, 64, 96, 128." >&2; exit 2 ;;
    esac
    DIFFICULTY_LABEL="k40"
    CONFIG_OVERRIDES=("potential.num_components=40" "potential.dim=10")
    ;;
  gmm_dim)
    case "${SIGMA_MAX}" in
      8|16|24|32) ;;
      *) echo "gmm_dim sigma_max must be one of: 8, 16, 24, 32." >&2; exit 2 ;;
    esac
    DIFFICULTY_LABEL="d100"
    CONFIG_OVERRIDES=("potential.num_components=10" "potential.dim=100")
    ;;
  *) echo "GMM_EXPERIMENT must be gmm_mode or gmm_dim." >&2; exit 2 ;;
esac

RUN_NAME="tune2_${EXPERIMENT}_${DIFFICULTY_LABEL}_pis_smax${SIGMA_MAX}_seed0"
TAGS="[gmm,sigma_max_tuning_stage2,variational,${DIFFICULTY_LABEL},pis,sigma${SIGMA_MAX},seed0,target_seed0]"

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
  "algorithm=pis"
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

echo "experiment=${EXPERIMENT} difficulty=${DIFFICULTY_LABEL} method=pis sigma_max=${SIGMA_MAX}"
echo "model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED}"
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
