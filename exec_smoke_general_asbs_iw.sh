#!/bin/bash

#SBATCH --job-name=asbs-smoke-general-iw
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-cmp[101-148,501-532]

set -euo pipefail

source ~/.profile
cd /home/htc/tkoshizuka/tr_asbs

RUN_ROOT="/home/htc/lrichter/SCRATCH/tr_asbs_data/smoke_runs"
mkdir -p "${RUN_ROOT}"

COMMON_OVERRIDES=(
  "max_outer_loops=2"
  "adjoint_matching_epochs=1"
  "corrector_matching_epochs=1"
  "steps_per_epoch=1"
  "train_batch_size=128"
  "inference_batch_size=128"
  "initial_buffer_samples=128"
  "buffer_samples_per_epoch=128"
  "eval_samples=128"
  "eval_samples_final=128"
  "num_samples_for_dual_loss=[64,16]"
  "log_detail_interval_epoch=1"
  "integrator.timesteps.steps=20"
  "progress_bar=false"
  "visualize=false"
  "save_samples=false"
  "experiment=general_asbs_iw_smoke"
  "root=${RUN_ROOT}"
  "hydra.run.dir=${RUN_ROOT}/\${name}"
  "fabric.loggers._target_=lightning.pytorch.loggers.CSVLogger"
)

run_combo() {
  local name="$1"
  shift 1

  echo "================================================================"
  echo "Running train_general_asbs_iw: ${name}"
  echo "================================================================"

  uv run python -u -m tr_asbs.experiment.train_general_asbs_iw \
    "name=${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

run_combo "smoke_general_iw_score_only" \
  "backward_loss_mode=score_matching" \
  "backward_loss_terms=[score_matching]" \
  "score_matching_weight=1.0" \
  "consistency_loss_weight=0.0" \
  "consistency_loss_full_path_weight=0.0" \
  "consistency_var_full_path_weight=0.0"

run_combo "smoke_general_iw_consistency_only" \
  "backward_loss_mode=consistency_loss" \
  "backward_loss_terms=[consistency_loss]" \
  "score_matching_weight=0.0" \
  "consistency_loss_weight=1.0" \
  "consistency_loss_full_path_weight=0.0" \
  "consistency_var_full_path_weight=0.0"

run_combo "smoke_general_iw_score_plus_consistency" \
  "backward_loss_mode=score_matching" \
  "backward_loss_terms=[score_matching,consistency_loss]" \
  "score_matching_weight=1.0" \
  "consistency_loss_weight=1.0" \
  "consistency_loss_full_path_weight=0.0" \
  "consistency_var_full_path_weight=0.0"

run_combo "smoke_general_iw_consistency_full_plus_var_full" \
  "backward_loss_mode=consistency_loss_full_path" \
  "backward_loss_terms=[consistency_loss_full_path,consistency_var_full_path]" \
  "score_matching_weight=0.0" \
  "consistency_loss_weight=0.0" \
  "consistency_loss_full_path_weight=1.0" \
  "consistency_var_full_path_weight=1.0"

run_combo "smoke_general_iw_all_three" \
  "backward_loss_mode=score_matching" \
  "backward_loss_terms=[score_matching,consistency_loss_full_path,consistency_var_full_path]" \
  "score_matching_weight=1.0" \
  "consistency_loss_weight=0.0" \
  "consistency_loss_full_path_weight=1.0" \
  "consistency_var_full_path_weight=1.0"
