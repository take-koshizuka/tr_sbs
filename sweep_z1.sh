#!/bin/bash

### EXPERIMENT configurations
# wandb agent ais2t_guests/measure_transport/bysg8iaw
sweep_ids=(
	'wandb agent ml-takeshi-work/tr_asbs/7asljq2g'
	'wandb agent ml-takeshi-work/tr_asbs/uohvb3ai'
) # Sweep IDs
job_name="MT" # Job name for tracking

### SLURM configurations
max_time="0-72:00:00"
partitions=("gpu")
exclude_nodes="htc-gpu[020-023,100]"
node_subset=""
# ... rest of config
n_gpus=1 # Number of GPUs per job
n_cpus=12 # Number of CPUs per job
submit_all_at_once=false # Submit all jobs at once
num_experiments=4 # Number of experiments per sweep, adjust as needed
max_concurrent_runs=100 # Maximum number of concurrent running jobs
sleep_between_runs=1

### CODE
partition_list=$(IFS=,; echo "${partitions[*]}")
git_repo_path="/home/htc/$USER/tr_asbs" # Path to the git repository
sweep_run_root="$git_repo_path/sweep_runs"
mkdir -p "$sweep_run_root/slurm_output"

# Iterate over the array and remove the "wandb agent " prefix if present
for i in "${!sweep_ids[@]}"; do
    sweep_ids[$i]=${sweep_ids[$i]#wandb agent }
done

# Function to check current running jobs
check_running_jobs() {
    current_running=$(squeue -u $USER -t RUNNING,PENDING -p $partition_list -o "\"%j\""  | grep $job_name | wc -l)
    if [[ $current_running -lt $max_concurrent_runs ]]; then
        return 0 # Okay to submit
    else
        return 1 # Wait needed
    fi
}

# Submit jobs
for sweep_id in "${sweep_ids[@]}"; do
    echo "Starting sweep $sweep_id."
    for (( i=1; i<=$num_experiments; i++ )); do
        if [[ $submit_all_at_once == false ]]; then
            while ! check_running_jobs; do
                sleep 5
            done
        fi
        
        # Add --exclude=$exclude_nodes here:
        sbatch -p $partition_list \
                --gres=gpu:$n_gpus \
                --mem=3000 \
                --cpus-per-task=$n_cpus \
                --time=$max_time \
                --job-name=$job_name \
                --exclude="$exclude_nodes" \
                --output="$sweep_run_root/slurm_output/%j.out" \
                runner.sh $git_repo_path $sweep_id
        
        sleep $sleep_between_runs
    done
done

