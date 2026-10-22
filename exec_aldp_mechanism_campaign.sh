#!/bin/bash
set -euo pipefail
REPO_ROOT="${1:?repo}"; PROJECT_PYTHON="${2:?python}"; CAMPAIGN_ROOT="${3:?campaign}"
STAGE="${4:?stage}"; CANDIDATE="${5:-source_half}"; SEED="${6:-0}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
hostname
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"${PROJECT_PYTHON}" -u scripts/run_aldp_mechanism_campaign.py \
  --campaign "${CAMPAIGN_ROOT}" --stage "${STAGE}" --candidate "${CANDIDATE}" --seed "${SEED}"
