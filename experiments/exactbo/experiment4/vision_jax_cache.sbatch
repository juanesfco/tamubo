#!/bin/bash
#SBATCH --job-name=exactbo-e4-jaxcache
#SBATCH --output=experiments/exactbo/experiment4/logs/jaxcache-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/jaxcache-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:40:00

# What JAX's persistent compilation cache saves: the same AutoBound + EI 10-d
# run (LHS, to partition 8) twice on all GPUs of the job, against a fresh cache
# directory: the first run compiles and fills it, the second reuses it.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_jax_cache.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E4="${REPO_ROOT}/experiments/exactbo/experiment4"
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${E4}/logs/jax_cache_${SLURM_JOB_ID}"

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
cd "${E4}"

for run in cold warm; do
  python stress_partitioning.py --acquisition ei --max-partitions 9 \
    --label "ei/autobound/${SLURM_GPUS_ON_NODE}gpu/cache-${run}/job${SLURM_JOB_ID}" \
    --out results/jax_cache_partition_counts.csv --summary results/jax_cache_runs.jsonl
  echo "cache entries after ${run} run: $(find "${JAX_COMPILATION_CACHE_DIR}" -type f | wc -l)"
done
