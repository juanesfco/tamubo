#!/bin/bash
#SBATCH --job-name=cupy-bench
#SBATCH --output=development/cupy_benchmark/logs/bench-%j.out
#SBATCH --partition=def
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00

# 16 matrix multiplications on 1 GPU, 8 GPUs (1 node) and 16 GPUs (2 nodes).
# Submit from the repo root:
#   sbatch development/cupy_benchmark/bench.sbatch

module load CUDA/12.9.1
source "${SLURM_SUBMIT_DIR}/envs/tamubo_exactbo/bin/activate"
HERE="${SLURM_SUBMIT_DIR}/development/cupy_benchmark"
RESULTS="${HERE}/logs/results-${SLURM_JOB_ID}.txt"

srun --nodes=1 --ntasks=1  python "${HERE}/bench.py" | tee -a "${RESULTS}"
srun --nodes=1 --ntasks=8  python "${HERE}/bench.py" | tee -a "${RESULTS}"
srun --nodes=2 --ntasks=16 python "${HERE}/bench.py" | tee -a "${RESULTS}"

# Time for each setup = its slowest GPU.
echo
awk '{ split($2, g, "="); split($6, s, "=");
       if (s[2] > t[g[2]]) t[g[2]] = s[2] }
     END { for (k in t) printf "%2d GPUs: %6.3f s   speedup %4.1fx\n", k, t[k], t[1] / t[k] }' \
    "${RESULTS}" | sort -n
