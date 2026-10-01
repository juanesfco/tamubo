#!/bin/bash
#SBATCH --job-name=exactbo-e4-profile
#SBATCH --output=experiments/exactbo/experiment4/logs/profile-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/profile-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00

# Why AutoBound partitioning barely speeds up on more GPUs: time breakdown of
# the uncapped AutoBound + EI 10-d run (to partition 8) on 1 GPU and on all
# GPUs of the job (default 2; use sbatch --gpus-per-node=N for more).
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_profile.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
cd "${REPO_ROOT}/experiments/exactbo/experiment4"

python profile_sharding.py --n-gpus 1
python profile_sharding.py --n-gpus "${SLURM_GPUS_ON_NODE:-2}"
