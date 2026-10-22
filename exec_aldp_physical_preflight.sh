#!/bin/bash
set -euo pipefail
REPO_ROOT="${1:?repo}"; PROJECT_PYTHON="${2:?python}"; CAMPAIGN_ROOT="${3:?campaign}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
hostname
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"${PROJECT_PYTHON}" -u scripts/run_aldp_physical_preflight.py --campaign "${CAMPAIGN_ROOT}"
