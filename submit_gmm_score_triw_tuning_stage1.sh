#!/bin/bash

# Submit the 18-run seed-0 Cartesian grid for the two hardest GMM settings.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_score_triw_tuning.sh"
DRY_RUN=false
HYDRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: bash submit_gmm_score_triw_tuning_stage1.sh [--dry-run] [HYDRA_OVERRIDE ...]"
      exit 0
      ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) HYDRA_OVERRIDES+=("$1") ;;
  esac
  shift
done

for override in "${HYDRA_OVERRIDES[@]}"; do
  case "${override}" in
    experiment=*|name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|base_sde.sigma_max=*|trust_region_bound=*|integrator.timesteps.steps=*|controller.*|corrector.*|score_conditioned_model.*|max_outer_loops=*|eval_samples_final=*|report_seed=*|monitor_initial_state=*|monitor_values=*|monitor_modes.*|divergence_early_stop.*|root=*|hydra.run.dir=*)
      echo "The matched TR-IW sweep controls this override: ${override}" >&2
      exit 2
      ;;
  esac
done

[[ -x "${RUNNER}" ]] || { echo "Runner is not executable: ${RUNNER}" >&2; exit 2; }
if [[ "${DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; use --dry-run or submit on the Slurm login node." >&2
  exit 127
fi

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_score_triw_tuning_v1}"
RUN_ROOT="${TR_ASBS_ROOT:-${SCRIPT_DIR}/runs/gmm_score_triw_tuning_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/stage1_submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\tstage\tsuite\tdifficulty\tsigma_max\ttrust_region_bound\tseed\ttarget_seed\trun_directory\tlog_path\n' > "${MANIFEST}"
fi

submit_one() {
  local suite="$1" difficulty="$2" sigma="$3" bound="$4"
  local bound_label="${bound/./p}"
  local experiment="gmm_${suite}_score_triw"
  local job_name="stri1-${difficulty}-s${sigma}-b${bound_label}"
  local run_name="score_triw_s1_${experiment}_${difficulty}_smax${sigma}_tr${bound_label}_seed0"
  local log_template="${LOG_ROOT}/%x-%j.out"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},TR_ASBS_ROOT=${RUN_ROOT},GMM_SUITE=${suite},GMM_SIGMA_MAX=${sigma},GMM_TRUST_REGION_BOUND=${bound},GMM_SEED=0,GMM_TARGET_SEED=0,GMM_TUNING_STAGE=s1"
  local command=(sbatch --parsable "--job-name=${job_name}" "--output=${log_template}" "--export=${exports}" "${RUNNER}")
  ((${#HYDRA_OVERRIDES[@]})) && command+=("${HYDRA_OVERRIDES[@]}")
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    return
  fi
  local job_id
  job_id="$("${command[@]}")"
  printf 'Submitted %s: %s\n' "${job_name}" "${job_id}"
  printf '%s\ts1\t%s\t%s\t%s\t%s\t0\t0\t%s\t%s\n' \
    "${job_id}" "${suite}" "${difficulty}" "${sigma}" "${bound}" \
    "${RUN_ROOT}/${run_name}" "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
}

for sigma in 2 4 8; do
  for bound in 0.1 0.3 1.0; do
    submit_one mode k40 "${sigma}" "${bound}"
  done
done
for sigma in 1 2 4; do
  for bound in 0.03 0.1 0.3; do
    submit_one dim d100 "${sigma}" "${bound}"
  done
done

[[ "${DRY_RUN}" == true ]] || echo "Submission manifest: ${MANIFEST}"

