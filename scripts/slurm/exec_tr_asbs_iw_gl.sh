#!/bin/bash

#SBATCH --job-name=tr-iw-gl-d6
#SBATCH --mem=64000
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=htc-cmp[101-148,501-532]

set -euo pipefail

source ~/.profile
REPO_ROOT="${TR_ASBS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${REPO_ROOT}"
RUN_ROOT="${TR_ASBS_ROOT:-${REPO_ROOT}/runs}"
mkdir -p "${RUN_ROOT}"

uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw \
  "experiment=mw5_delta_gl" \
  "name=tr_asbs_iw_mw5_delta_gl_delta6" \
  "potential.delta=6.0" \
  "gamma_schedule=trust_region" \
  "root=${RUN_ROOT}" \
  "hydra.run.dir=${RUN_ROOT}/\${name}"
