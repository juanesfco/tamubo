#!/bin/bash
#SBATCH --job-name=exactbo-e4-memory
#SBATCH --output=experiments/exactbo/experiment4/logs/memory-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/memory-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00

# Step 3 of the Vision stress test: push the uncapped AutoBound + EI 10-d run
# past partition 8 (the GB10 limit) and record where it stops. Boxes live on
# the main GPU, so its 141 GB is the limit with any GPU count (use e.g.
# sbatch --gpus-per-node=1 for one GPU). nvidia-smi samples device memory
# (CuPy and JAX together) every 5 s.
# Extra arguments are passed to stress_partitioning.py, e.g.
#   sbatch experiments/exactbo/experiment4/vision_memory.sbatch --autobound-batch-size 4096
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_memory.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E4="${REPO_ROOT}/experiments/exactbo/experiment4"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
cd "${E4}"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -l 5 \
  > "logs/memory-${SLURM_JOB_ID}-nvidia-smi.csv" &
SMI=$!

python stress_partitioning.py --acquisition ei --max-partitions 14 --progress \
  --label "ei/autobound/${SLURM_GPUS_ON_NODE:-8}gpu/push/job${SLURM_JOB_ID}" \
  --out results/memory_partition_counts.csv --summary results/memory_runs.jsonl "$@"
status=$?
kill "${SMI}" 2>/dev/null
exit "${status}"
