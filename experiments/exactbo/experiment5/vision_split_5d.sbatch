#!/bin/bash
#SBATCH --job-name=exactbo-e5-5d
#SBATCH --output=experiments/exactbo/experiment5/logs/split5d-%j.out
#SBATCH --error=experiments/exactbo/experiment5/logs/split5d-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00

# Length-scale-aware splitting, part 1 (one H200, uncapped):
#   - partition/bound tests on GPU
#   - regression: split_ratio=0 must reproduce the experiment-3/4 10-d counts
#   - problem5d, X0=32 (out of memory at split 10 with every dimension split):
#     split_ratio 1 and 1/3, LHS and center sampling
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment5/vision_split_5d.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E5="${REPO_ROOT}/experiments/exactbo/experiment5"
DRIVER="${REPO_ROOT}/experiments/exactbo/experiment4/stress_partitioning.py"
OUT="${E5}/results/split_partition_counts.csv"
SUMMARY="${E5}/results/split_runs.jsonl"
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

echo "##### tests #####"
python -m pytest -q test_partition.py ../experiment3/test_bounds.py

run 900 "regression/problem10d/ei/lhs/ratio=0" --acquisition ei --box-sampling lhs --max-partitions 9

for ratio in 1 0.3333333333; do
  for sampling in lhs center; do
    run 1800 "problem5d/n0=32/logei/${sampling}/ratio=${ratio}" --problem problem5d --n0 32 \
      --box-sampling "${sampling}" --split-ratio "${ratio}" --max-partitions 1000
  done
done
