#!/bin/bash
# Score-conditioned, 128-transition TR-IW with PIS-style cosine noise.
# Submit only on z3 through scripts/monitor_gmm_joint_cosine_triw.py.

#SBATCH --job-name=joint-cos-triw
#SBATCH --mem=64000
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=htc-cmp[101-148,501-532],htc-gpu024

set -euo pipefail
source ~/.profile

REPO_ROOT="${TR_ASBS_REPO_ROOT:?Set TR_ASBS_REPO_ROOT}"
RUN_ROOT="${TR_ASBS_ROOT:?Set TR_ASBS_ROOT}"
PYTHON="${TR_ASBS_PYTHON:?Set TR_ASBS_PYTHON}"
SUITE="${GMM_SUITE:?Set GMM_SUITE}"
SIGMA_MAX="${GMM_SIGMA_MAX:?Set GMM_SIGMA_MAX}"
TRUST_BOUND="${GMM_TRUST_REGION_BOUND:?Set GMM_TRUST_REGION_BOUND}"
SEED="${GMM_SEED:?Set GMM_SEED}"
STAGE="${GMM_STAGE:?Set GMM_STAGE}"
DIFFICULTY="${GMM_DIFFICULTY:-k40_d100}"

case "${DIFFICULTY}:${SUITE}" in
  k40_d100:dim) EXPERIMENT=gmm_joint_dim_score_triw_cosine ;;
  k40_d100:mode) EXPERIMENT=gmm_joint_mode_score_triw_cosine ;;
  k40_d10:mode) EXPERIMENT=gmm_mode_score_triw_cosine ;;
  *) echo "Invalid target: ${DIFFICULTY}:${SUITE}" >&2; exit 2 ;;
esac
case "${STAGE}" in smoke|stage1|stage2) ;; *) echo "Invalid stage: ${STAGE}" >&2; exit 2 ;; esac
[[ "${SEED}" =~ ^[012]$ ]] || { echo "Invalid model seed: ${SEED}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Python not executable: ${PYTHON}" >&2; exit 2; }

BOUND_LABEL="${TRUST_BOUND/./p}"
RUN_NAME="cosine_triw_${STAGE}_${SUITE}_${DIFFICULTY}_smax${SIGMA_MAX}_tr${BOUND_LABEL}_seed${SEED}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}/.tmp"
cd "${REPO_ROOT}"
export TMPDIR="${RUN_DIR}/.tmp"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  "experiment=${EXPERIMENT}"
  "name=${RUN_NAME}"
  "seed=${SEED}"
  "+potential.seed=0"
  "base_sde.sigma_max=${SIGMA_MAX}"
  "trust_region_bound=${TRUST_BOUND}"
  "gamma_schedule=trust_region"
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
  "fabric.loggers.tags=[gmm,score_hint,tr_iw,pis_cosine,matched_128,${DIFFICULTY},${SUITE},${STAGE},sigma${SIGMA_MAX},tr${BOUND_LABEL},seed${SEED}]"
  "root=${RUN_ROOT}"
  "hydra.run.dir=${RUN_DIR}"
  "progress_bar=false"
)

if [[ "${STAGE}" == smoke ]]; then
  ARGS+=(
    "max_outer_loops=1"
    "adjoint_matching_epochs=1"
    "corrector_matching_epochs=1"
    "steps_per_epoch=1"
    "train_batch_size=64"
    "inference_batch_size=512"
    "initial_buffer_samples=512"
    "buffer_samples_per_epoch=512"
    "eval_samples=128"
    "num_samples_for_dual_loss=[64,8]"
    "log_detail_interval_epoch=1"
    "divergence_early_stop.enabled=false"
  )
fi

echo "stage=${STAGE} suite=${SUITE} difficulty=${DIFFICULTY} sigma_max=${SIGMA_MAX} trust_region_bound=${TRUST_BOUND} model_seed=${SEED} target_seed=0"
echo "experiment=${EXPERIMENT} steps=128 solver=exponential output=${RUN_DIR}"
echo "git_commit=$(git rev-parse HEAD)"
printf 'command:'; printf ' %q' "${PYTHON}" -u -m tr_asbs.experiment.train_gamma_asbs_iw "${ARGS[@]}"; printf '\n'
hostname
nvidia-smi
"${PYTHON}" -u -m tr_asbs.experiment.train_gamma_asbs_iw "${ARGS[@]}"
