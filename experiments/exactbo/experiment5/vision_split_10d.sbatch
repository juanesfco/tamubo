#!/bin/bash
#SBATCH --job-name=exactbo-e5-10d
#SBATCH --output=experiments/exactbo/experiment5/logs/split10d-%j.out
#SBATCH --error=experiments/exactbo/experiment5/logs/split10d-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00

# Length-scale-aware splitting, part 2: the uncapped problem10d search (X0=32,
# seed 0) on one H200. With every dimension split it runs out of memory at
# partition 11 (experiment 4, step 3). Runs to convergence, out of memory or the
# time limit, with --progress at every split.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment5/vision_split_10d.sbatch

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

# Same setting as the experiment-3/4 memory push (AutoBound + EI, LHS).
run 7200 "problem10d/ei/lhs/ratio=1" --acquisition ei --box-sampling lhs --split-ratio 1 --max-partitions 1000
run 7200 "problem10d/ei/lhs/ratio=0.333" --acquisition ei --box-sampling lhs --split-ratio 0.3333333333 --max-partitions 1000
# Library defaults (logEI, center sampling).
run 7200 "problem10d/logei/center/ratio=1" --split-ratio 1 --max-partitions 1000
