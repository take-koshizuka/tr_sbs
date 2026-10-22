#!/bin/bash
# Slurm resource requests and ledger are supplied by the campaign monitor.
set -euo pipefail
REPO_ROOT="${1:?repo}"; PROJECT_PYTHON="${2:?python}"; CAMPAIGN_ROOT="${3:?campaign}"
STAGE="${4:?stage}"; CHART="${5:?chart}"; SEED="${6:?seed}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled
hostname
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"${PROJECT_PYTHON}" -u scripts/run_aldp_chart_campaign.py \
  --campaign "${CAMPAIGN_ROOT}" --stage "${STAGE}" --chart "${CHART}" --seed "${SEED}"
