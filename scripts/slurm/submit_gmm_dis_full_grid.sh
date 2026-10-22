#!/bin/bash

# Submit 5 mode settings x 3 seeds + 4 dimensions x 3 seeds = 27 DIS jobs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/scripts/slurm/exec_gmm_dis_full_grid.sh"
DRY_RUN=false
HYDRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    -h|--help)
      echo "Usage: bash scripts/slurm/submit_gmm_dis_full_grid.sh [--dry-run] [HYDRA_OVERRIDE ...]"
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
    experiment=*|algorithm=*|name=*|run_name=*|seed=*|potential.seed=*|+potential.seed=*|potential.dim=*|potential.num_components=*|base_sde.sigma_max=*|algorithm.iters=*|evaluation.num_evaluations=*|evaluation.num_samples=*|evaluation.final_num_samples=*|evaluation.report_seed=*|evaluation.report_checkpoints=*|evaluation.evaluate_initial_state=*|monitor_initial_state=*|monitor_values=*|monitor_modes.*|divergence_early_stop.*|root=*|hydra.run.dir=*)
      echo "The tuned DIS full grid controls this override: ${override}" >&2
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

LOG_ROOT="${TR_ASBS_LOG_ROOT:-${SCRIPT_DIR}/slurm_logs/gmm_dis_full_grid_best_v1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${LOG_ROOT}/submissions_${TIMESTAMP}.tsv"
if [[ "${DRY_RUN}" == false ]]; then
  mkdir -p "${LOG_ROOT}"
  printf 'job_id\tsuite\tgrid_value\tmethod\tsigma_max\tseed\ttarget_seed\trun_directory\tlog_path\n' > "${MANIFEST}"
fi

submit_one() {
  local suite="$1"
  local value="$2"
  local seed="$3"
  local sigma="$4"
  local prefix label experiment
  if [[ "${suite}" == "mode" ]]; then
    prefix="dm-k"
    label="k${value}"
    experiment="gmm_mode"
  else
    prefix="dd-d"
    label="d${value}"
    experiment="gmm_dim"
  fi
  local job_name="${prefix}${value}-s${seed}"
  local run_name="${experiment}_${label}_dis_seed${seed}"
  local run_root="${SCRIPT_DIR}/runs/gmm_dis_full_grid_best_v1"
  local log_template="${LOG_ROOT}/%x-%j.out"
  local exports="ALL,TR_ASBS_REPO_ROOT=${SCRIPT_DIR},GMM_SUITE=${suite},GMM_GRID_VALUE=${value},GMM_SEED=${seed},GMM_TARGET_SEED=0"
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
  printf 'Submitted suite=%s value=%s seed=%s method=dis sigma=%s: %s\n' \
    "${suite}" "${value}" "${seed}" "${sigma}" "${job_id}"
  printf '%s\t%s\t%s\tdis\t%s\t%s\t0\t%s\t%s\n' \
    "${job_id}" "${suite}" "${value}" "${sigma}" "${seed}" \
    "${run_root}/${run_name}" "${LOG_ROOT}/${job_name}-${job_id}.out" >> "${MANIFEST}"
}

for value in 5 10 20 30 40; do
  for seed in 0 1 2; do
    submit_one mode "${value}" "${seed}" 2
  done
done
for value in 10 25 50 100; do
  for seed in 0 1 2; do
    submit_one dim "${value}" "${seed}" 4
  done
done

if [[ "${DRY_RUN}" == false ]]; then
  echo "Submission manifest: ${MANIFEST}"
fi
