#!/bin/bash
set -euo pipefail
cd "${1:?code}"; unset JAX_PLATFORMS
export PYTHONPATH="$1/scripts:$1/src" PYTHONUNBUFFERED=1 WANDB_MODE=disabled MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname; nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"${2:?python}" -u scripts/run_aldp_scaled_backward_gate.py --campaign "${3:?campaign}"
