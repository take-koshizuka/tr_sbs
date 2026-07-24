#!/bin/bash

#SBATCH --job-name=gen-iw-sc+co
#SBATCH --mem=64000
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-cmp[101-148,501-532]

set -euo pipefail

source ~/.profile
cd /home/htc/tkoshizuka/tr_asbs

RUN_ROOT="${PWD}/runs/asbs_general"
mkdir -p "${RUN_ROOT}"

EXPERIMENT="${EXPERIMENT:-ring_gmm}"
NAME="${NAME:-general_iw_${EXPERIMENT}_score_plus_consistency}"

uv run python -u -m tr_asbs.experiment.train_general_asbs_iw \
  "experiment=${EXPERIMENT}" \
  "name=${NAME}" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}" \
  "backward_loss_mode=score_matching" \
  "backward_loss_terms=[score_matching,consistency_loss]" \
  "score_matching_weight=1.0" \
  "consistency_loss_weight=1.0" \
  "consistency_loss_full_path_weight=0.0" \
  "consistency_var_full_path_weight=0.0" \
  "$@"

