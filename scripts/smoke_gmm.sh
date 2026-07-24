#!/bin/bash
#SBATCH --job-name=asbs-iw-smoke
#SBATCH --partition=gpu
#SBATCH --output=/home/htc/tkoshizuka/SCRATCH/tr_asbs/logs/smoke_gmm/run.out
#SBATCH --error=/home/htc/tkoshizuka/SCRATCH/tr_asbs/logs/smoke_gmm/run.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

export PYTHONFAULTHANDLER=1

LOG_DIR=/home/htc/tkoshizuka/SCRATCH/tr_asbs/logs/smoke_gmm
RUN_ROOT=/home/htc/tkoshizuka/SCRATCH/tr_asbs/runs/smoke_gmm

mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_ROOT}"

# Tiny CPU smoke run for GMM path
uv run python -u -m tr_asbs.experiment.train_asbs experiment=gmm_smoke
