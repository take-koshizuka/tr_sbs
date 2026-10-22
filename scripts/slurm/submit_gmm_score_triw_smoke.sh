#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/scripts/slurm/exec_gmm_score_triw_smoke.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true
LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_score_triw_smoke_v1}"
RUN_ROOT="${TR_ASBS_ROOT:-${SCRIPT_DIR}/runs/gmm_score_triw_smoke_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/smoke_submission_${TIMESTAMP}.tsv"
command=(sbatch --parsable --job-name=stri-smoke "--output=${LOG_ROOT}/%x-%j.out" "--export=ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},TR_ASBS_ROOT=${RUN_ROOT}" "${RUNNER}")
if [[ "${DRY_RUN}" == true ]]; then
  printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  exit 0
fi
mkdir -p "${LOG_ROOT}"
job_id="$("${command[@]}")"
printf 'job_id\tstage\trun_directory\tlog_path\n' > "${MANIFEST}"
printf '%s\tsmoke\t%s\t%s\n' "${job_id}" "${RUN_ROOT}/score_triw_smoke_k40_d10" "${LOG_ROOT}/stri-smoke-${job_id}.out" >> "${MANIFEST}"
echo "Submitted score-conditioned TR-IW smoke: ${job_id}"
echo "Submission manifest: ${MANIFEST}"
