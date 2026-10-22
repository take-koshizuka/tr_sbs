#!/bin/bash

# One score-conditioned, 128-step TR-IW hardest-setting tuning run.

#SBATCH --job-name=gmm-score-triw
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
SIGMA_MAX="${GMM_SIGMA_MAX:-}"
TRUST_BOUND="${GMM_TRUST_REGION_BOUND:-}"
GLOBAL_SEED="${GMM_SEED:-}"
TARGET_SEED="${GMM_TARGET_SEED:-0}"
TUNING_STAGE="${GMM_TUNING_STAGE:-s1}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_score_triw_tuning_v1}"

case "${TUNING_STAGE}" in
  s1)
    [[ "${GLOBAL_SEED}" == "0" ]] || { echo "Stage 1 requires seed=0." >&2; exit 2; }
    ;;
  s2)
    [[ "${GLOBAL_SEED}" == "1" || "${GLOBAL_SEED}" == "2" ]] || {
      echo "Stage 2 requires seed=1 or seed=2." >&2; exit 2;
    }
    ;;
  *) echo "GMM_TUNING_STAGE must be s1 or s2." >&2; exit 2 ;;
esac
[[ "${TARGET_SEED}" == "0" ]] || { echo "Target seed must remain 0." >&2; exit 2; }

case "${SUITE}" in
  mode)
    EXPERIMENT="gmm_mode_score_triw"
    DIFFICULTY="k40"
    case "${SIGMA_MAX}" in 2|4|8) ;; *) echo "Mode sigma must be 2, 4, or 8." >&2; exit 2;; esac
    case "${TRUST_BOUND}" in 0.1|0.3|1.0) ;; *) echo "Mode bound must be 0.1, 0.3, or 1.0." >&2; exit 2;; esac
    TARGET_OVERRIDES=("potential.num_components=40" "potential.dim=10")
    ;;
  dim)
    EXPERIMENT="gmm_dim_score_triw"
    DIFFICULTY="d100"
    case "${SIGMA_MAX}" in 1|2|4) ;; *) echo "Dim sigma must be 1, 2, or 4." >&2; exit 2;; esac
    case "${TRUST_BOUND}" in 0.03|0.1|0.3) ;; *) echo "Dim bound must be 0.03, 0.1, or 0.3." >&2; exit 2;; esac
    TARGET_OVERRIDES=("potential.num_components=10" "potential.dim=100")
    ;;
  *) echo "GMM_SUITE must be mode or dim." >&2; exit 2 ;;
esac

BOUND_LABEL="${TRUST_BOUND/./p}"
RUN_NAME="score_triw_${TUNING_STAGE}_${EXPERIMENT}_${DIFFICULTY}_smax${SIGMA_MAX}_tr${BOUND_LABEL}_seed${GLOBAL_SEED}"
TAGS="[gmm,score_conditioned_tr_iw,matched_128,${TUNING_STAGE},${DIFFICULTY},sigma${SIGMA_MAX},tr${BOUND_LABEL},seed${GLOBAL_SEED},target_seed0]"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"
# W&B stages images in tempfile before moving them into RUN_ROOT. Avoid a
# cross-filesystem copy, which needs extra quota and failed when /home filled.
export TMPDIR="${RUN_ROOT}/.tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
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

TRAIN_COMMAND=("${PYTHON_COMMAND[@]}" -u -m tr_asbs.experiment.train_gamma_asbs_iw
  "experiment=${EXPERIMENT}"
  "name=${RUN_NAME}"
  "seed=${GLOBAL_SEED}"
  "+potential.seed=${TARGET_SEED}"
  "base_sde.sigma_max=${SIGMA_MAX}"
  "gamma_schedule=trust_region"
  "trust_region_bound=${TRUST_BOUND}"
  "max_outer_loops=150"
  "eval_samples_final=4000"
  "report_seed=12345"
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

echo "stage=${TUNING_STAGE} suite=${SUITE} difficulty=${DIFFICULTY} sigma_max=${SIGMA_MAX} trust_region_bound=${TRUST_BOUND}"
echo "model_seed=${GLOBAL_SEED} target_seed=${TARGET_SEED} path_steps=128 report_seed=12345 final_samples=4000"
echo "run_name=${RUN_NAME} output=${RUN_ROOT}/${RUN_NAME}"
printf 'command:'; printf ' %q' "${TRAIN_COMMAND[@]}"; printf '\n'
hostname
nvidia-smi

"${TRAIN_COMMAND[@]}"
