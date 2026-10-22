#!/bin/bash
set -euo pipefail
repo="${1:?repo}"; python="${2:?python}"; campaign="${3:?campaign}"
cd "$repo"
unset JAX_PLATFORMS
export PYTHONPATH="$repo/scripts:$repo/src" PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"$python" -u scripts/run_aldp_noise_solver_numerical_diagnostic.py --campaign "$campaign"
