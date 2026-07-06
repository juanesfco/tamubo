#!/bin/bash

#SBATCH --job-name=tamubo-matrix-benchmark
#SBATCH --account=132672089060
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=native/jobs/slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juan.florez@tamu.edu

set -euo pipefail

cd $SCRATCH/tamubo

module purge
module load GCC/14.3.0 OpenMPI/5.0.8 Ninja/1.13.0 CUDA/13.1.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Slurm/HPRC-specific defaults. Generic benchmark defaults live in run_matrix_benchmark.sh.
export MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS:-2}}"
export CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75}"
export DATA_DIR="${DATA_DIR:-${PWD}/native/data/matrix_benchmark_${SLURM_JOB_ID}}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "workdir=${PWD}"
echo "MPI_RANKS=${MPI_RANKS}"
echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

module list
which nvcc
nvcc --version
which mpicxx
mpicxx --version
which cmake
cmake --version
which ninja
ninja --version
nvidia-smi -L

bash ./native/benchmarks/matrix/run_matrix_benchmark.sh
