#!/bin/bash

# Submit the PIS/DIS extension of both GMM grids:
#   mode: 5 K values x 3 seeds x 2 methods = 30 jobs
#   dim:  4 d values x 3 seeds x 2 methods = 24 jobs

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_variational_grid.sh"
SUITE="all"
DRY_RUN=false
HYDRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    --suite)
      if (($# < 2)); then
        echo "--suite requires all, mode, or dim." >&2
        exit 2
      fi
      SUITE="$2"
      shift
      ;;
    -h|--help)
      echo "Usage: bash submit_gmm_variational_grid.sh [--dry-run] [--suite all|mode|dim] [HYDRA_OVERRIDE ...]"
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

case "${SUITE}" in
  all|mode|dim) ;;
  *) echo "--suite must be all, mode, or dim." >&2; exit 2 ;;
esac

for override in "${HYDRA_OVERRIDES[@]}"; do
  case "${override}" in
    experiment=*|algorithm=*|name=*|run_name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|monitor_values=*|monitor_modes.*|root=*|hydra.run.dir=*)
      echo "The variational GMM grid controls this override: ${override}" >&2
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

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_variational_full_grid_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\tsuite\tgrid_value\tseed\tmethod\tlog_path\n' > "${MANIFEST}"
fi

submit_one() {
  local suite="$1"
  local value="$2"
  local seed="$3"
  local method="$4"
  local label
  if [[ "${suite}" == "mode" ]]; then
    label="k${value}"
  else
    label="d${value}"
  fi
  local job_name="gv-${label}-${method}-s${seed}"
  local log_path="${LOG_ROOT}/%x-%j.out"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_SUITE=${suite},GMM_GRID_VALUE=${value},GMM_METHOD=${method},GMM_SEED=${seed},GMM_TARGET_SEED=0"
  local command=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--output=${log_path}"
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
  printf 'Submitted suite=%s value=%s seed=%s method=%s: %s\n' \
    "${suite}" "${value}" "${seed}" "${method}" "${job_id}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${suite}" "${value}" "${seed}" "${method}" \
    "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
}

if [[ "${SUITE}" == "all" || "${SUITE}" == "mode" ]]; then
  for value in 5 10 20 30 40; do
    for seed in 0 1 2; do
      submit_one mode "${value}" "${seed}" pis
      submit_one mode "${value}" "${seed}" dis
    done
  done
fi

if [[ "${SUITE}" == "all" || "${SUITE}" == "dim" ]]; then
  for value in 10 25 50 100; do
    for seed in 0 1 2; do
      submit_one dim "${value}" "${seed}" pis
      submit_one dim "${value}" "${seed}" dis
    done
  done
fi

if [[ "${DRY_RUN}" == false ]]; then
  echo "Submission manifest: ${MANIFEST}"
fi
