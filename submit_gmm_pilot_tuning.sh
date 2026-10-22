#!/bin/bash

# 14 new jobs. Existing sigma=2/bound=0.3 Gamma-ASBS and sigma=2 IW-fixed
# report3 runs are reused instead of resubmitted.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_pilot_tuning.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

submit_one() {
  local experiment="$1" label="$2" method="$3" sigma="$4" bound="${5:-}"
  local short="gamma-b${bound/./p}"
  [[ "${method}" == "iw_fixed" ]] && short="iwfix"
  local job_name="pilot-${label}-${short}-s${sigma}"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_EXPERIMENT=${experiment},GMM_METHOD=${method},GMM_SIGMA_MAX=${sigma}"
  [[ -z "${bound}" ]] || exports="${exports},GMM_TRUST_REGION_BOUND=${bound}"
  local command=(sbatch --parsable "--job-name=${job_name}" "--export=${exports}" "${RUNNER}")
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    local job_id; job_id="$("${command[@]}")"
    echo "Submitted ${job_name}: ${job_id}"
  fi
}

for spec in "gmm_mode:k40" "gmm_dim:d100"; do
  IFS=: read -r experiment label <<<"${spec}"
  for sigma in 1 2 4; do
    for bound in 0.1 0.3; do
      # Reuse the completed report3 baseline.
      [[ "${sigma}" == 2 && "${bound}" == 0.3 ]] && continue
      submit_one "${experiment}" "${label}" gamma_asbs "${sigma}" "${bound}"
    done
  done
  for sigma in 1 4; do
    submit_one "${experiment}" "${label}" iw_fixed "${sigma}"
  done
done
