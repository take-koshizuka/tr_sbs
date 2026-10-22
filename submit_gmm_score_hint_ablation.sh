#!/bin/bash

# Submit 12 hint-off runs plus four missing PIS hint-on seeds for paired tests.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_score_hint_ablation.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true
[[ -x "${RUNNER}" ]] || { echo "Runner is not executable: ${RUNNER}" >&2; exit 2; }
if [[ "${DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; use --dry-run or submit on the Slurm login node." >&2
  exit 127
fi

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_score_hint_ablation_v1}"
RUN_ROOT="${TR_ASBS_ROOT:-${SCRIPT_DIR}/runs/gmm_score_hint_ablation_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\tsuite\tdifficulty\tmethod\tsigma_max\tscore_hint_mode\tseed\ttarget_seed\trun_directory\tlog_path\n' > "${MANIFEST}"
fi

submit_one() {
  local suite="$1" difficulty="$2" method="$3" sigma="$4" hint_mode="$5" seed="$6"
  local hint_short=off
  [[ "${hint_mode}" == "provided" ]] && hint_short=on
  local job_name="hint-${hint_short}-${difficulty}-${method}-z${seed}"
  local experiment="gmm_${suite}"
  local run_name="hint_${hint_mode}_${experiment}_${difficulty}_${method}_smax${sigma}_seed${seed}"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},TR_ASBS_ROOT=${RUN_ROOT},GMM_SUITE=${suite},GMM_METHOD=${method},GMM_SCORE_HINT_MODE=${hint_mode},GMM_SEED=${seed},GMM_TARGET_SEED=0"
  local command=(sbatch --parsable "--job-name=${job_name}" "--output=${LOG_ROOT}/%x-%j.out" "--export=${exports}" "${RUNNER}")
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    return
  fi
  local job_id
  job_id="$("${command[@]}")"
  printf 'Submitted %s: %s\n' "${job_name}" "${job_id}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\n' \
    "${job_id}" "${suite}" "${difficulty}" "${method}" "${sigma}" \
    "${hint_mode}" "${seed}" "${RUN_ROOT}/${run_name}" \
    "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
}

for seed in 0 1 2; do
  submit_one mode k40 dis 2 zero "${seed}"
  submit_one dim d100 dis 4 zero "${seed}"
  submit_one mode k40 pis 128 zero "${seed}"
  submit_one dim d100 pis 32 zero "${seed}"
done

# PIS's selected high-sigma settings only have seed 0 from tuning. Add the two
# missing score-on seeds so every score-off run has a paired score-on baseline.
for seed in 1 2; do
  submit_one mode k40 pis 128 provided "${seed}"
  submit_one dim d100 pis 32 provided "${seed}"
done

[[ "${DRY_RUN}" == true ]] || echo "Submission manifest: ${MANIFEST}"

