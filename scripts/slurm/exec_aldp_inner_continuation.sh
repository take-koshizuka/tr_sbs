#!/bin/bash
set -euo pipefail
REPO_ROOT="${1:?repo}"
PROJECT_PYTHON="${2:?python}"
CAMPAIGN_ROOT="${3:?campaign}"
CANDIDATE="${4:?candidate}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
# Discard CPU-only selection inherited from bookkeeping or sbatch's environment.
# The neural backend still requires a GPU and refuses CPU training fallback.
unset JAX_PLATFORMS
printf 'jax_platform_selection=auto\n'
hostname
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"${PROJECT_PYTHON}" -u scripts/run_aldp_inner_continuation.py \
  --campaign "${CAMPAIGN_ROOT}" --candidate "${CANDIDATE}"
