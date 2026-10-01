#!/bin/bash
#SBATCH --job-name=exactbo-multigpu-bench
#SBATCH --output=development/cupy_multigpu_benchmark/logs/bench-%j.out
#SBATCH --error=development/cupy_multigpu_benchmark/logs/bench-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00

# ExactBO CuPy backend: scaling over 1, 2, 4, 8 GPUs of one Vision node.
# Submit from the repo root:
#   sbatch development/cupy_multigpu_benchmark/bench_multigpu.sbatch
# Results are appended to logs/results_<jobid>.csv; see README.md.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
BENCH="${REPO_ROOT}/development/cupy_multigpu_benchmark"
OUT="${BENCH}/logs/results_${SLURM_JOB_ID}.csv"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
# Reuse JAX's AutoBound compilations across jobs (~2-4 s per program and GPU).
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME}/.cache/jax_compilation_cache}"
cd "${BENCH}"

python bench_multigpu.py --dims 2 5 10 --preset moderate --iters 3 --out "${OUT}"
python bench_multigpu.py --dims 5 --preset stress --iters 1 --out "${OUT}"
python bench_multigpu.py --dims 10 --preset stress --box-sampling center --iters 1 --out "${OUT}"
