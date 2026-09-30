#!/bin/bash
#SBATCH --job-name=exactbo-exp2
#SBATCH --output=experiments/exactbo/experiment2/logs/exp2-%j.out
#SBATCH --error=experiments/exactbo/experiment2/logs/exp2-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

# Runs experiment 2 on Vision. ExactBO uses cupy (backend "auto" or "cupy")
# and splits its per-box work across every GPU allocated here; request more
# with e.g. `sbatch --gpus-per-node=8 ...`.
#
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment2/submit_vision.sbatch [path/to/config.json]

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
EXP_DIR="${REPO_ROOT}/experiments/exactbo/experiment2"
CONFIG_PATH="${1:-${EXP_DIR}/experiment_config.json}"

# cupy-cuda12x loads the CUDA runtime/NVRTC libraries from the system toolkit.
module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"

python "${EXP_DIR}/run_experiment.py" --config "${CONFIG_PATH}"
