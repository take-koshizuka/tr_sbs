#!/bin/bash

# Submit the hardest-setting GMM comparison grid:
#   gmm_mode: num_components=40, dim=10
#   gmm_dim:  dim=100, num_components=10
# crossed with ASBS, Gamma-ASBS, IW fixed, and IW trust-region at 0.1/0.3.
#
# Use --dry-run to inspect all ten sbatch commands without submitting them.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/scripts/slurm/exec_gmm_comparison.sh"
DRY_RUN=false
HYDRA_OVERRIDES=()
RUN_SUFFIX="${GMM_RUN_SUFFIX:-}"

if [[ -n "${RUN_SUFFIX}" && ! "${RUN_SUFFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "GMM_RUN_SUFFIX must contain only letters, digits, underscores, or hyphens." >&2
  exit 2
fi

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    -h|--help)
      echo "Usage: bash scripts/slurm/submit_gmm_comparison.sh [--dry-run] [HYDRA_OVERRIDE ...]"
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      ;;
  esac
  shift
done

for override in "${HYDRA_OVERRIDES[@]}"; do
  case "${override}" in
    experiment=*|name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|gamma_schedule=*|trust_region_bound=*|max_outer_loops=*|visualize=*|monitor_values=*|monitor_modes.*|monitor_initial_state=*)
      echo "The GMM comparison grid controls this override: ${override}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is not executable: ${RUNNER}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; use --dry-run or submit on a Slurm login node." >&2
  exit 127
fi

submit_job() {
  local experiment="$1"
  local difficulty="$2"
  local method="$3"
  local short_method="$4"
  local trust_region_bound="${5:-}"
  local job_name="gmm-${difficulty}-${short_method}-s0"
  local export_values="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_EXPERIMENT=${experiment},GMM_METHOD=${method},GMM_SEED=0,GMM_TARGET_SEED=0"
  if [[ -n "${RUN_SUFFIX}" ]]; then
    job_name="${job_name}-${RUN_SUFFIX}"
    export_values="${export_values},GMM_RUN_SUFFIX=${RUN_SUFFIX}"
  fi
  if [[ -n "${trust_region_bound}" ]]; then
    export_values="${export_values},GMM_TRUST_REGION_BOUND=${trust_region_bound}"
  fi
  local command=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--export=${export_values}"
    "${RUNNER}"
    "${HYDRA_OVERRIDES[@]}"
  )

  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  local job_id
  job_id="$("${command[@]}")"
  echo "Submitted ${job_name}: ${job_id}"
}

for experiment_spec in "gmm_mode:k40" "gmm_dim:d100"; do
  IFS=: read -r experiment difficulty <<<"${experiment_spec}"
  submit_job "${experiment}" "${difficulty}" asbs asbs
  submit_job "${experiment}" "${difficulty}" gamma_asbs gamma
  submit_job "${experiment}" "${difficulty}" gamma_asbs_iw_fixed iw-fix
  submit_job "${experiment}" "${difficulty}" gamma_asbs_iw_trust_region iw-tr01 0.1
  submit_job "${experiment}" "${difficulty}" gamma_asbs_iw_trust_region iw-tr03 0.3
done
