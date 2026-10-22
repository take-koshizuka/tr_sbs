#!/bin/bash

# Submit 12 new jobs: 2 settings x 2 methods x sigma_max {1,4,8}.
# sigma_max=2 is already available from the completed report3 comparison.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/scripts/slurm/exec_gmm_sigma_sweep.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

for experiment_spec in "gmm_mode:k40" "gmm_dim:d100"; do
  IFS=: read -r experiment difficulty <<<"${experiment_spec}"
  for method_spec in "asbs:asbs" "iw_tr:iwtr"; do
    IFS=: read -r method short_method <<<"${method_spec}"
    for sigma in 1 4 8; do
      job_name="sig-${difficulty}-${short_method}-s${sigma}"
      exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_EXPERIMENT=${experiment},GMM_METHOD=${method},GMM_SIGMA_MAX=${sigma},GMM_SEED=0,GMM_TARGET_SEED=0,GMM_RUN_SUFFIX=sigma_tune"
      command=(sbatch --parsable "--job-name=${job_name}" "--export=${exports}" "${RUNNER}")
      if [[ "${DRY_RUN}" == true ]]; then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
      else
        job_id="$("${command[@]}")"
        echo "Submitted ${job_name}: ${job_id}"
      fi
    done
  done
done
