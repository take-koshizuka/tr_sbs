#!/bin/bash

# One PIS/DIS score-hint ablation run at the previously selected best sigma.

#SBATCH --job-name=gmm-hint-ablate
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
METHOD="${GMM_METHOD:-}"
HINT_MODE="${GMM_SCORE_HINT_MODE:-}"
GLOBAL_SEED="${GMM_SEED:-}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_score_hint_ablation_v1}"

case "${GLOBAL_SEED}" in 0|1|2) ;; *) echo "GMM_SEED must be 0, 1, or 2." >&2; exit 2 ;; esac
[[ "${TARGET_SEED}" == "0" ]] || { echo "Target seed must remain 0." >&2; exit 2; }
case "${HINT_MODE}" in zero|provided) ;; *) echo "GMM_SCORE_HINT_MODE must be zero or provided." >&2; exit 2 ;; esac
case "${METHOD}" in pis|dis) ;; *) echo "GMM_METHOD must be pis or dis." >&2; exit 2 ;; esac

case "${SUITE}:${METHOD}" in
  mode:dis) EXPERIMENT=gmm_mode; DIFFICULTY=k40; SIGMA_MAX=2; TARGET_OVERRIDES=("potential.num_components=40" "potential.dim=10") ;;
  dim:dis) EXPERIMENT=gmm_dim; DIFFICULTY=d100; SIGMA_MAX=4; TARGET_OVERRIDES=("potential.num_components=10" "potential.dim=100") ;;
  mode:pis) EXPERIMENT=gmm_mode; DIFFICULTY=k40; SIGMA_MAX=128; TARGET_OVERRIDES=("potential.num_components=40" "potential.dim=10") ;;
  dim:pis) EXPERIMENT=gmm_dim; DIFFICULTY=d100; SIGMA_MAX=32; TARGET_OVERRIDES=("potential.num_components=10" "potential.dim=100") ;;
  *) echo "Unsupported suite/method pair: ${SUITE}/${METHOD}" >&2; exit 2 ;;
esac

RUN_NAME="hint_${HINT_MODE}_${EXPERIMENT}_${DIFFICULTY}_${METHOD}_smax${SIGMA_MAX}_seed${GLOBAL_SEED}"
TAGS="[gmm,score_hint_ablation,hint_${HINT_MODE},${DIFFICULTY},${METHOD},sigma${SIGMA_MAX},seed${GLOBAL_SEED},target_seed0]"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_COMMAND="${TR_ASBS_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON_COMMAND}" ]] || { echo "Project Python not found." >&2; exit 2; }

TRAIN_COMMAND=("${PYTHON_COMMAND}" -u -m tr_asbs.experiment.train_variational
  "experiment=${EXPERIMENT}"
  "algorithm=${METHOD}"
  "name=${RUN_NAME}"
  "run_name=${RUN_NAME}"
  "seed=${GLOBAL_SEED}"
  "+potential.seed=${TARGET_SEED}"
  "base_sde.sigma_max=${SIGMA_MAX}"
  "algorithm.model.score_hint_mode=${HINT_MODE}"
  "algorithm.iters=40000"
  "algorithm.num_steps=128"
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
TRAIN_COMMAND+=("${TARGET_OVERRIDES[@]}")
TRAIN_COMMAND+=("$@")

echo "suite=${SUITE} difficulty=${DIFFICULTY} method=${METHOD} sigma_max=${SIGMA_MAX} score_hint_mode=${HINT_MODE}"
echo "model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED} steps=128 iterations=40000 report_seed=12345 final_samples=4000"
echo "run_name=${RUN_NAME} output=${RUN_ROOT}/${RUN_NAME}"
printf 'command:'; printf ' %q' "${TRAIN_COMMAND[@]}"; printf '\n'
hostname
nvidia-smi

"${TRAIN_COMMAND[@]}"
