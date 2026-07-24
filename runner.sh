#!/bin/bash
# Usage: runner.sh <git_repo_path> <sweep_id>

git_repo_path=$1
sweep_id=$2
run_root="$git_repo_path/sweep_runs"
wandb_dir="$run_root/wandb"
tmp_dir="$run_root/tmp"

# Acquire node information
echo 'Getting node information'
date;hostname;id;pwd

# Switch to the cwd
bash
cd $git_repo_path
echo 'Current working directory:'
pwd

which python
nvidia-smi

# Setup internet access and set environment variables
echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Set output directories'
mkdir -p "$run_root" "$wandb_dir" "$tmp_dir"
export TR_ASBS_ROOT="$run_root"
export WANDB_DIR="$wandb_dir"
export TMPDIR="$tmp_dir"

echo "TR_ASBS_ROOT=$TR_ASBS_ROOT"
echo "WANDB_DIR=$WANDB_DIR"
echo "TMPDIR=$TMPDIR"

# Execute the job
srun uv run wandb agent --count 1 $sweep_id

