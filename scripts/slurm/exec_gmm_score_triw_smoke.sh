#!/bin/bash

# Remote-only end-to-end smoke for the matched score-conditioned TR-IW path.

#SBATCH --job-name=gmm-score-triw-smoke
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=htc-cmp[101-148,501-532]
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source ~/.profile

REPO_ROOT="${TR_ASBS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs/gmm_score_triw_smoke_v1}"
cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_COMMAND="${TR_ASBS_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON_COMMAND}" ]] || { echo "Project Python not found." >&2; exit 2; }

"${PYTHON_COMMAND}" -u -m tr_asbs.experiment.train_gamma_asbs_iw \
  experiment=gmm_mode_score_triw \
  name=score_triw_smoke_k40_d10 \
  seed=0 \
  +potential.seed=0 \
  max_outer_loops=1 \
  adjoint_matching_epochs=1 \
  corrector_matching_epochs=1 \
  steps_per_epoch=1 \
  train_batch_size=64 \
  inference_batch_size=512 \
  initial_buffer_samples=512 \
  buffer_samples_per_epoch=512 \
  eval_samples=128 \
  eval_samples_final=4000 \
  num_samples_for_dual_loss='[64,8]' \
  log_detail_interval_epoch=1 \
  save_checkpoint_interval_epoch=100 \
  report_seed=12345 \
  visualize=true \
  progress_bar=false \
  monitor_initial_state=false \
  divergence_early_stop.enabled=false \
  fabric.loggers.tags='[gmm,score_conditioned_tr_iw,matched_128,smoke]' \
  "root=${RUN_ROOT}" \
  'hydra.run.dir=${root}/${name}'
