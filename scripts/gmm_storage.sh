#!/usr/bin/env bash
# Shared default for GMM result directories. TR_ASBS_ROOT always wins.

gmm_run_root() {
  local repo_root="$1" campaign="$2" scratch_parent
  if [[ -n "${TR_ASBS_ROOT:-}" ]]; then
    printf '%s\n' "${TR_ASBS_ROOT}"
    return
  fi
  scratch_parent="${TR_ASBS_SCRATCH_ROOT:-/scratch/htc/${USER:-$(id -un)}}"
  if [[ -d "${scratch_parent}" && -w "${scratch_parent}" ]]; then
    printf '%s\n' "${scratch_parent}/tr_asbs_gmm_runs/${campaign}"
  else
    printf '%s\n' "${repo_root}/runs/${campaign}"
  fi
}
