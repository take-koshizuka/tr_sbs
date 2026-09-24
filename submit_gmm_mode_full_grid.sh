#!/bin/bash

# Submit 5 component counts x 3 model seeds x 4 tuned methods = 60 jobs.
# The GMM target is fixed at potential.seed=0 for every run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_mode_full_grid.sh"
DRY_RUN=false
HYDRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    -h|--help)
      echo "Usage: bash submit_gmm_mode_full_grid.sh [--dry-run] [HYDRA_OVERRIDE ...]"
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

if ((${#HYDRA_OVERRIDES[@]})); then
  for override in "${HYDRA_OVERRIDES[@]}"; do
    case "${override}" in
      experiment=*|name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|base_sde.sigma_max=*|gamma_schedule=*|trust_region_bound=*|max_outer_loops=*|visualize=*|monitor_values=*|monitor_modes.*|monitor_initial_state=*|divergence_early_stop.*|root=*|hydra.run.dir=*)
        echo "The tuned gmm_mode grid controls this override: ${override}" >&2
        exit 2
        ;;
    esac
  done
fi

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is not executable: ${RUNNER}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; use --dry-run or submit on a Slurm login node." >&2
  exit 127
fi

submit_one() {
  local components="$1"
  local seed="$2"
  local method="$3"
  local short_method="$4"
  local job_name="gm-k${components}-${short_method}-s${seed}"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_NUM_COMPONENTS=${components},GMM_METHOD=${method},GMM_SEED=${seed},GMM_TARGET_SEED=0"
  local command=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--export=${exports}"
    "${RUNNER}"
  )
  if ((${#HYDRA_OVERRIDES[@]})); then
    command+=("${HYDRA_OVERRIDES[@]}")
  fi

  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  local job_id
  job_id="$("${command[@]}")"
  echo "Submitted k=${components} seed=${seed} method=${method}: ${job_id}"
}

for components in 5 10 20 30 40; do
  for seed in 0 1 2; do
    submit_one "${components}" "${seed}" asbs asbs
    submit_one "${components}" "${seed}" gamma_asbs gamma
    submit_one "${components}" "${seed}" gamma_asbs_iw_fixed iwfix
    submit_one "${components}" "${seed}" gamma_asbs_iw_trust_region iwtr
  done
done
