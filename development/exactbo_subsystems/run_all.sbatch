#!/bin/bash
#SBATCH --job-name=exactbo-subsystems
#SBATCH --output=development/exactbo_subsystems/logs/run-%j.out
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00

# Every subsystem script on one GPU (CuPy + JAX), then two larger cases.
# Submit from the repo root:
#   sbatch development/exactbo_subsystems/run_all.sbatch

module load CUDA/12.9.1
source "${SLURM_SUBMIT_DIR}/envs/tamubo_exactbo/bin/activate"
cd "${SLURM_SUBMIT_DIR}/development/exactbo_subsystems"

for s in s1_partition s2_distance_bounds s3_kernel_bounds s4_mu_bounds s5_sigma_bounds s6_logei_bounds s7_exact_search; do
    echo "### ${s}"
    python "${s}.py" --backend cupy
done

echo "### s6_logei_bounds, d=10, N=128"
python s6_logei_bounds.py --backend cupy --d 10 --N 128 --n 8192
echo "### s7_exact_search, d=4, N=32"
python s7_exact_search.py --backend cupy --d 4 --N 32 --eps-x 0.01
