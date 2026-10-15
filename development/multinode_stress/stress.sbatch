#!/bin/bash
#SBATCH --job-name=vision-stress
#SBATCH --output=development/multinode_stress/logs/stress-%j.out
#SBATCH --error=development/multinode_stress/logs/stress-%j.err
#SBATCH --partition=def
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=01:00:00

# Multi-node GPU stress test (GEMM, HBM, memory, NCCL, ring matmul, burn), one
# task per GPU. Submit from the repo root; extra arguments go to stress.py:
#   sbatch development/multinode_stress/stress.sbatch
#   sbatch --nodes=4 development/multinode_stress/stress.sbatch
#   sbatch --nodes=1 --time=00:15:00 development/multinode_stress/stress.sbatch --quick
# Output: logs/stress-<jobid>.out and logs/<jobid>/ (see README.md).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
HERE="${REPO_ROOT}/development/multinode_stress"
OUT="${HERE}/logs/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"

# NCCL init/network/P2P details, one file per process (summarize.py reads them).
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,P2P
export NCCL_DEBUG_FILE="${OUT}/nccl.%h.%p.log"

srun --kill-on-bad-exit=1 --cpu-bind=cores python "${HERE}/stress.py" --out "${OUT}" "$@"
python "${HERE}/summarize.py" "${OUT}" || echo "summarize.py failed"
