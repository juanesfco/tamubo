#!/bin/bash
#SBATCH --job-name=exactbo-e4-correctness
#SBATCH --output=experiments/exactbo/experiment4/logs/correctness-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/correctness-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00

# Step 1 of the Vision stress test: the experiment-3 soundness tests and the
# four uncapped 10-d partitioning runs on one H200, to compare with the GB10
# counts in experiments/exactbo/experiment3/README.md.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_correctness.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
EXP3="${REPO_ROOT}/experiments/exactbo/experiment3"
OUT="${REPO_ROOT}/experiments/exactbo/experiment4/results/partition_counts_vision_1gpu.csv"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
cd "${EXP3}"

status=0
echo "##### environment #####"
python - <<'EOF' || status=1
import cupy, jax
print("cupy", cupy.__version__, "GPUs:", cupy.cuda.runtime.getDeviceCount(),
      cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("jax", jax.__version__, "devices:", jax.devices())
EOF

echo "##### pytest test_bounds.py #####"
python -m pytest -q test_bounds.py || status=1

echo "##### partitioning (GB10 README configurations) #####"
python run_partitioning.py --acquisition ei --bound-method interval --max-partitions 6 --out "${OUT}" || status=1
python run_partitioning.py --acquisition logei --bound-method interval --max-partitions 6 --out "${OUT}" || status=1
python run_partitioning.py --acquisition logei --bound-method autobound --max-partitions 8 --out "${OUT}" || status=1
python run_partitioning.py --acquisition ei --bound-method autobound --max-partitions 9 --out "${OUT}" || status=1
exit "${status}"
