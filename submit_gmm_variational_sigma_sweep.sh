#!/bin/bash

# Submit 2 hardest settings x 2 methods x sigma_max {1,2,4,8} = 16 jobs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/exec_gmm_variational_sigma_sweep.sh"
DRY_RUN=false
HYDRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    -h|--help)
      echo "Usage: bash submit_gmm_variational_sigma_sweep.sh [--dry-run] [HYDRA_OVERRIDE ...]"
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
    experiment=*|algorithm=*|name=*|run_name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|base_sde.sigma_max=*|algorithm.iters=*|evaluation.num_evaluations=*|evaluation.evaluate_initial_state=*|monitor_initial_state=*|monitor_values=*|monitor_modes.*|divergence_early_stop.*|root=*|hydra.run.dir=*)
      echo "The fair sigma sweep controls this override: ${override}" >&2
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

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_variational_sigma_tuning_seed0_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\texperiment\tdifficulty\tmethod\tsigma_max\tseed\tlog_path\n' > "${MANIFEST}"
fi

submit_one() {
  local experiment="$1"
  local difficulty="$2"
  local method="$3"
  local sigma="$4"
  local job_name="vsg-${difficulty}-${method}-s${sigma}"
  local log_template="${LOG_ROOT}/%x-%j.out"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_EXPERIMENT=${experiment},GMM_METHOD=${method},GMM_SIGMA_MAX=${sigma},GMM_SEED=0,GMM_TARGET_SEED=0"
  local command=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--output=${log_template}"
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
  printf 'Submitted experiment=%s method=%s sigma=%s: %s\n' \
    "${experiment}" "${method}" "${sigma}" "${job_id}"
  printf '%s\t%s\t%s\t%s\t%s\t0\t%s\n' \
    "${job_id}" "${experiment}" "${difficulty}" "${method}" "${sigma}" \
    "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
}

for experiment_spec in "gmm_mode:k40" "gmm_dim:d100"; do
  IFS=: read -r experiment difficulty <<<"${experiment_spec}"
  for method in pis dis; do
    for sigma in 1 2 4 8; do
      submit_one "${experiment}" "${difficulty}" "${method}" "${sigma}"
    done
  done
done

if [[ "${DRY_RUN}" == false ]]; then
  echo "Submission manifest: ${MANIFEST}"
fi
