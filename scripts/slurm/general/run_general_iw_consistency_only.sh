#!/bin/bash

#SBATCH --job-name=gen-iw-cons
#SBATCH --mem=64000
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-cmp[101-148,501-532]

set -euo pipefail

source ~/.profile
REPO_ROOT="${TR_ASBS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "${REPO_ROOT}"

RUN_ROOT="${PWD}/runs/asbs_general"
mkdir -p "${RUN_ROOT}"

EXPERIMENT="${EXPERIMENT:-ring_gmm}"
NAME="${NAME:-general_iw_${EXPERIMENT}_consistency_only}"

uv run python -u -m tr_asbs.experiment.train_general_asbs_iw \
  "experiment=${EXPERIMENT}" \
  "name=${NAME}" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}" \
  "backward_loss_mode=consistency_loss" \
  "backward_loss_terms=[consistency_loss]" \
  "score_matching_weight=0.0" \
  "consistency_loss_weight=1.0" \
  "consistency_loss_full_path_weight=0.0" \
  "consistency_var_full_path_weight=0.0" \
  "$@"
