#!/bin/bash

# Submit the fair score-free GMM comparison:
#   5 mode settings x 3 seeds x 2 methods = 30 jobs
#   4 dimension settings x 3 seeds x 2 methods = 24 jobs

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/gmm_storage.sh"
RUNNER="${SCRIPT_DIR}/exec_gmm_variational_asbs_control_grid.sh"
METHOD="all"
DRY_RUN=false
HYDRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --method)
      if (($# < 2)); then echo "--method requires all, dis, or pis." >&2; exit 2; fi
      METHOD="$2"
      shift
      ;;
    -h|--help)
      echo "Usage: bash submit_gmm_variational_asbs_control_grid.sh [--dry-run] [--method all|dis|pis] [HYDRA_OVERRIDE ...]"
      exit 0
      ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) HYDRA_OVERRIDES+=("$1") ;;
  esac
  shift
done

case "${METHOD}" in all|dis|pis) ;; *) echo "--method must be all, dis, or pis." >&2; exit 2 ;; esac

for override in "${HYDRA_OVERRIDES[@]}"; do
  case "${override}" in
    experiment=*|algorithm=*|algorithm.model.*|name=*|run_name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|base_sde.sigma_max=*|algorithm.iters=*|evaluation.*|monitor_initial_state=*|monitor_values=*|monitor_modes.*|divergence_early_stop.*|root=*|hydra.run.dir=*)
      echo "The fair variational grid controls this override: ${override}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${RUNNER}" ]]; then echo "Runner is not executable: ${RUNNER}" >&2; exit 2; fi
if [[ "${DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; use --dry-run or submit on a Slurm login node." >&2
  exit 127
fi

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_variational_asbs_control_scoreless_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\tsuite\tgrid_value\tmethod\tsigma_max\tseed\ttarget_seed\tcontrol\tscore_hint\trun_directory\tlog_path\n' > "${MANIFEST}"
fi

submit_one() {
  local suite="$1" value="$2" seed="$3" method="$4"
  local label sigma job_name experiment run_name run_root log_template exports
  if [[ "${suite}" == "mode" ]]; then
    label="k${value}"; experiment="gmm_mode"
    if [[ "${method}" == "dis" ]]; then sigma="2"; else sigma="128"; fi
  else
    label="d${value}"; experiment="gmm_dim"
    if [[ "${method}" == "dis" ]]; then sigma="4"; else sigma="32"; fi
  fi
  job_name="vaf-${label}-${method}-s${seed}"
  run_name="gmm_${suite}_${label}_${method}_asbs_control_no_score_seed${seed}"
  run_root="$(gmm_run_root "${SCRIPT_DIR}" gmm_variational_asbs_control_scoreless_v1)"
  log_template="${LOG_ROOT}/%x-%j.out"
  exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_SUITE=${suite},GMM_GRID_VALUE=${value},GMM_METHOD=${method},GMM_SEED=${seed},GMM_TARGET_SEED=0"
  local command=(sbatch --parsable "--job-name=${job_name}" "--output=${log_template}" "--export=${exports}" "${RUNNER}")
  if ((${#HYDRA_OVERRIDES[@]})); then command+=("${HYDRA_OVERRIDES[@]}"); fi

  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'; return
  fi
  local job_id
  job_id="$("${command[@]}")"
  printf 'Submitted suite=%s value=%s method=%s sigma=%s seed=%s: %s\n' "${suite}" "${value}" "${method}" "${sigma}" "${seed}" "${job_id}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t0\tASBSFourierControl\tnone\t%s\t%s\n' \
    "${job_id}" "${suite}" "${value}" "${method}" "${sigma}" "${seed}" \
    "${run_root}/${run_name}" "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
}

submit_grid() {
  local method="$1"
  for value in 5 10 20 30 40; do for seed in 0 1 2; do submit_one mode "${value}" "${seed}" "${method}"; done; done
  for value in 10 25 50 100; do for seed in 0 1 2; do submit_one dim "${value}" "${seed}" "${method}"; done; done
}

if [[ "${METHOD}" == "all" || "${METHOD}" == "dis" ]]; then submit_grid dis; fi
if [[ "${METHOD}" == "all" || "${METHOD}" == "pis" ]]; then submit_grid pis; fi

if [[ "${DRY_RUN}" == false ]]; then echo "Submission manifest: ${MANIFEST}"; fi
