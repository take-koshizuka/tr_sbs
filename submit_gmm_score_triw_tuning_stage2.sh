#!/bin/bash

# Submit seeds 1 and 2 for the two stage-1 winners in each hardest setting.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_score_triw_tuning.sh"
CANDIDATES=""
DRY_RUN=false

while (($#)); do
  case "$1" in
    --candidates) CANDIDATES="$2"; shift ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: bash submit_gmm_score_triw_tuning_stage2.sh --candidates TOP2.tsv [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

[[ -n "${CANDIDATES}" && -f "${CANDIDATES}" ]] || {
  echo "A stage-1 top2_candidates.tsv file is required." >&2; exit 2;
}
[[ -x "${RUNNER}" ]] || { echo "Runner is not executable: ${RUNNER}" >&2; exit 2; }
if [[ "${DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; use --dry-run or submit on the Slurm login node." >&2
  exit 127
fi

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_score_triw_tuning_v1}"
RUN_ROOT="${TR_ASBS_ROOT:-${SCRIPT_DIR}/runs/gmm_score_triw_tuning_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/stage2_submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\tstage\tsuite\tdifficulty\tsigma_max\ttrust_region_bound\tseed\ttarget_seed\trun_directory\tlog_path\n' > "${MANIFEST}"
fi

candidate_count=0
while IFS=$'\t' read -r suite rank sigma bound; do
  bound="${bound//$'\r'/}"
  [[ "${suite}" == "suite" ]] && continue
  [[ "${rank}" == "1" || "${rank}" == "2" ]] || { echo "Invalid candidate rank: ${rank}" >&2; exit 2; }
  case "${suite}" in mode) difficulty=k40 ;; dim) difficulty=d100 ;; *) echo "Invalid suite: ${suite}" >&2; exit 2 ;; esac
  candidate_count=$((candidate_count + 1))
  for seed in 1 2; do
    bound_label="${bound/./p}"
    experiment="gmm_${suite}_score_triw"
    job_name="stri2-${difficulty}-r${rank}-z${seed}"
    run_name="score_triw_s2_${experiment}_${difficulty}_smax${sigma}_tr${bound_label}_seed${seed}"
    log_template="${LOG_ROOT}/%x-%j.out"
    exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},TR_ASBS_ROOT=${RUN_ROOT},GMM_SUITE=${suite},GMM_SIGMA_MAX=${sigma},GMM_TRUST_REGION_BOUND=${bound},GMM_SEED=${seed},GMM_TARGET_SEED=0,GMM_TUNING_STAGE=s2"
    command=(sbatch --parsable "--job-name=${job_name}" "--output=${log_template}" "--export=${exports}" "${RUNNER}")
    if [[ "${DRY_RUN}" == true ]]; then
      printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
      continue
    fi
    job_id="$("${command[@]}")"
    printf 'Submitted %s: %s\n' "${job_name}" "${job_id}"
    printf '%s\ts2\t%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\n' \
      "${job_id}" "${suite}" "${difficulty}" "${sigma}" "${bound}" "${seed}" \
      "${RUN_ROOT}/${run_name}" "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
  done
done < "${CANDIDATES}"

[[ "${candidate_count}" == "4" ]] || {
  echo "Expected four candidates (two per suite), got ${candidate_count}." >&2; exit 2;
}
[[ "${DRY_RUN}" == true ]] || echo "Submission manifest: ${MANIFEST}"
