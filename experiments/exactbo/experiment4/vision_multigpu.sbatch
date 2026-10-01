#!/bin/bash
#SBATCH --job-name=exactbo-e4-multigpu
#SBATCH --output=experiments/exactbo/experiment4/logs/multigpu-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/multigpu-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00

# Step 2 of the Vision stress test: AutoBound on 1/2/4/8 GPUs of one node.
# First checks that CuPy and JAX map every GPU to the same device, then runs
# the two AutoBound configurations of step 1 with n_gpus = 1, 2, 4, 8.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_multigpu.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E4="${REPO_ROOT}/experiments/exactbo/experiment4"
OUT="${E4}/results/multigpu_partition_counts.csv"
SUMMARY="${E4}/results/multigpu_runs.jsonl"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
# Reuse JAX's AutoBound compilations across jobs (~2-4 s per program and GPU).
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME}/.cache/jax_compilation_cache}"
cd "${E4}"

status=0
echo "##### device check #####"
python check_devices.py || status=1

# Use as many of 1, 2, 4, 8 GPUs as the job was given (e.g. sbatch --gpus-per-node=2 ...).
for g in 1 2 4 8; do
  [ "${g}" -le "${SLURM_GPUS_ON_NODE:-8}" ] || continue
  echo "##### ${g} GPU(s) #####"
  python stress_partitioning.py --acquisition ei --max-partitions 9 --n-gpus "${g}" \
    --label "ei/autobound/${g}gpu/job${SLURM_JOB_ID}" --out "${OUT}" --summary "${SUMMARY}" || status=1
  python stress_partitioning.py --acquisition logei --max-partitions 8 --n-gpus "${g}" \
    --label "logei/autobound/${g}gpu/job${SLURM_JOB_ID}" --out "${OUT}" --summary "${SUMMARY}" || status=1
done
exit "${status}"
