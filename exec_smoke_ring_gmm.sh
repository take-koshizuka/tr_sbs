#!/bin/bash

#SBATCH --job-name=asbs-smoke-ring-gmm
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
  "root=${RUN_ROOT}"
  "hydra.run.dir=${RUN_ROOT}/\${name}"
)

run_smoke() {
  local module="$1"
  local experiment="$2"
  local name="$3"
  shift 3

  echo "================================================================"
  echo "Running ${module} experiment=${experiment} name=${name}"
  echo "================================================================"

  uv run python -u -m "tr_asbs.experiment.${module}" \
    "experiment=${experiment}" \
    "name=${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

run_smoke "train_asbs" "ring_gmm" "smoke_train_asbs_ring_gmm"
