#!/bin/bash
#SBATCH --job-name=exactbo-e5-frontier
#SBATCH --output=experiments/exactbo/experiment5/logs/frontier-%j.out
#SBATCH --error=experiments/exactbo/experiment5/logs/frontier-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

# Bounds-only frontier storage (one H200, uncapped): the partition loop keeps
# only parents and one EI bound per child and rebuilds children in chunks.
# Reruns the runs of vision_split_{5d,10d}.sbatch, which must reproduce
# results/split_runs.jsonl (targets per partition, incumbent, point).
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment5/vision_frontier.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E5="${REPO_ROOT}/experiments/exactbo/experiment5"
DRIVER="${REPO_ROOT}/experiments/exactbo/experiment4/stress_partitioning.py"
OUT="${E5}/results/frontier_partition_counts.csv"
SUMMARY="${E5}/results/frontier_runs.jsonl"
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
run 1800 "problem10d/ei/lhs/ratio=1" --acquisition ei --box-sampling lhs --split-ratio 1 --max-partitions 1000
run 1800 "problem10d/ei/lhs/ratio=0.333" --acquisition ei --box-sampling lhs --split-ratio 0.3333333333 --max-partitions 1000
run 1800 "problem10d/logei/center/ratio=1" --split-ratio 1 --max-partitions 1000
for ratio in 1 0.3333333333; do
  for sampling in lhs center; do
    run 1800 "problem5d/n0=32/logei/${sampling}/ratio=${ratio}" --problem problem5d --n0 32 \
      --box-sampling "${sampling}" --split-ratio "${ratio}" --max-partitions 1000
  done
done
