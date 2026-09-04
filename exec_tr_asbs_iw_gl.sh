#!/bin/bash

#SBATCH --job-name=tr-iw-gl-d6
#SBATCH --mem=64000
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-cmp[101-148,501-532]

set -euo pipefail

source ~/.profile
cd /home/htc/tkoshizuka/tr_asbs
RUN_ROOT="/home/htc/lrichter/SCRATCH/tr_asbs_data/runs"
mkdir -p "${RUN_ROOT}"

uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw \
  "experiment=mw5_delta_gl" \
  "name=tr_asbs_iw_mw5_delta_gl_delta6" \
  "potential.delta=6.0" \
  "gamma_schedule=trust_region" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}"

