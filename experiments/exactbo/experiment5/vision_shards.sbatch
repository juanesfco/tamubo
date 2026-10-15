#!/bin/bash
#SBATCH --job-name=exactbo-e5-shards
#SBATCH --output=experiments/exactbo/experiment5/logs/shards-%j.out
#SBATCH --error=experiments/exactbo/experiment5/logs/shards-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00

# Frontier sharded over the GPUs (one H200 node, uncapped): each GPU holds a
# contiguous range of the parents and its children's EI bounds. Tests (including
# multi-GPU = one GPU), then the largest regression runs on 1 and 8 GPUs; the
# 1-GPU runs must reproduce results/split_runs.jsonl.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment5/vision_shards.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E5="${REPO_ROOT}/experiments/exactbo/experiment5"
DRIVER="${REPO_ROOT}/experiments/exactbo/experiment4/stress_partitioning.py"
OUT="${E5}/results/shard_partition_counts.csv"
SUMMARY="${E5}/results/shard_runs.jsonl"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME}/.cache/jax_compilation_cache}"
cd "${E5}"

run() {  # run <timeout_s> <label> <args...>
  local limit="$1" label="$2"; shift 2
  echo "##### ${label} #####"
  timeout "${limit}" python "${DRIVER}" --progress --label "${label}" --out "${OUT}" --summary "${SUMMARY}" "$@" \
    || echo "RUN ${label}: FAILED or timed out (exit $?)"
}

nvidia-smi topo -m
echo "##### tests #####"
python -m pytest -q test_partition.py ../experiment3/test_bounds.py

for gpus in 1 8; do
  run 900 "regression/problem10d/ei/lhs/ratio=0/${gpus}gpu" --acquisition ei --box-sampling lhs --max-partitions 9 \
    --n-gpus "${gpus}"
  run 1800 "problem10d/ei/lhs/ratio=0.333/${gpus}gpu" --acquisition ei --box-sampling lhs --split-ratio 0.3333333333 \
    --max-partitions 1000 --n-gpus "${gpus}"
  run 1800 "problem5d/n0=32/logei/center/ratio=0.3333333333/${gpus}gpu" --problem problem5d --n0 32 \
    --box-sampling center --split-ratio 0.3333333333 --max-partitions 1000 --n-gpus "${gpus}"
done
