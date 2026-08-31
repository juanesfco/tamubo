#!/usr/bin/env bash

#SBATCH --job-name=tamubo-matrix-h200
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/vision-slurm-%x-%j.out

set -euo pipefail

###############################################################################
# Fixed platform and benchmark configuration
###############################################################################

VISION_USER="juan.florez_tamu.edu"
SCRATCH_ROOT="/scratch/user/${VISION_USER}"

PROJECT_DIR="${SCRATCH_ROOT}/tamubo"
RESULTS_ROOT="${SCRATCH_ROOT}/tamubo/native/benchmarks/matrix/data"
PLATFORM="vision-h200"

MATRIX_N=8192
MPI_RANKS=1
REPEATS=5
CYCLES=3
HOLD_MS=0

ATOL="1e-8"
RTOL="1e-8"

CUDA_ARCHITECTURES=90
CMAKE_GENERATOR="Unix Makefiles"
CPU_THREADS=14

SMOKE_DIR="${PROJECT_DIR}/native/benchmarks/smoke"
MATRIX_DIR="${PROJECT_DIR}/native/benchmarks/matrix"

SMOKE_BUILD_DIR="${SMOKE_DIR}/executables/build-vision-h200"
SMOKE_EXECUTABLE_DIR="${SMOKE_DIR}/executables"

MATRIX_BUILD_DIR="${MATRIX_DIR}/executables/build-vision-h200"
MATRIX_EXECUTABLE_DIR="${MATRIX_DIR}/executables"

RUN_ID="job-${SLURM_JOB_ID}-n${MATRIX_N}-r${MPI_RANKS}"
DATA_DIR="${RESULTS_ROOT}/${PLATFORM}/${RUN_ID}"

RUN_LOG="${DATA_DIR}/run.out"
METADATA_LOG="${DATA_DIR}/metadata.txt"
SMOKE_LOG="${DATA_DIR}/smoke.out"
COMPARE_LOG="${DATA_DIR}/matrix_compare.out"

export OMP_NUM_THREADS="${CPU_THREADS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CPU_THREADS}"

###############################################################################
# Validate paths and initialize output
###############################################################################

if [[ ! -d "${PROJECT_DIR}/.git" ]]; then
    echo "ERROR: repository not found at ${PROJECT_DIR}" >&2
    echo "Clone it before submitting this job:" >&2
    echo "  cd ${SCRATCH_ROOT}" >&2
    echo "  git clone https://github.com/juanesfco/tamubo.git" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"

# Everything after this point is saved both in the Slurm output and run.out.
exec > >(tee -a "${RUN_LOG}") 2>&1

START_TIME="$(date -u --iso-8601=seconds)"
START_EPOCH="$(date +%s)"

finish_job() {
    local status=$?
    local end_time
    local end_epoch
    local elapsed

    end_time="$(date -u --iso-8601=seconds)"
    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - START_EPOCH))

    echo
    echo "job_finished=${end_time}"
    echo "elapsed_seconds=${elapsed}"
    echo "exit_status=${status}"
    echo "results_directory=${DATA_DIR}"

    trap - EXIT
    exit "${status}"
}

trap finish_job EXIT

###############################################################################
# Load the VISION software environment
###############################################################################

module purge
module load GCC/14.3.0
module load CUDA/12.9.1
module load OpenMPI/5.0.8
module load CMake/3.31.8

###############################################################################
# Record reproducibility and hardware metadata
###############################################################################

cd "${PROJECT_DIR}"

{
    echo "start_time=${START_TIME}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "slurm_job_name=${SLURM_JOB_NAME}"
    echo "hostname=$(hostname)"
    echo "project_dir=${PROJECT_DIR}"
    echo "results_root=${RESULTS_ROOT}"
    echo "data_dir=${DATA_DIR}"
    echo
    echo "platform=${PLATFORM}"
    echo "matrix_n=${MATRIX_N}"
    echo "mpi_ranks=${MPI_RANKS}"
    echo "repeats=${REPEATS}"
    echo "cycles=${CYCLES}"
    echo "hold_ms=${HOLD_MS}"
    echo "atol=${ATOL}"
    echo "rtol=${RTOL}"
    echo "cuda_architectures=${CUDA_ARCHITECTURES}"
    echo "cmake_generator=${CMAKE_GENERATOR}"
    echo "cpu_threads=${CPU_THREADS}"
    echo
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git branch --show-current)"
    echo
    echo "git_status:"
    git status --short
    echo
    echo "loaded_modules:"
    module list 2>&1
    echo
    echo "operating_system:"
    uname -a
    echo
    echo "cpu_information:"
    lscpu
    echo
    echo "memory_information:"
    free -h
    echo
    echo "gpu_inventory:"
    nvidia-smi -L
    echo
    echo "gpu_properties:"
    nvidia-smi \
        --query-gpu=name,uuid,driver_version,pstate,memory.total,power.limit \
        --format=csv
    echo
    echo "gpu_topology:"
    nvidia-smi topo -m
    echo
    echo "gcc:"
    command -v gcc
    gcc --version
    echo
    echo "g++:"
    command -v g++
    g++ --version
    echo
    echo "nvcc:"
    command -v nvcc
    nvcc --version
    echo
    echo "mpicxx:"
    command -v mpicxx
    mpicxx --version
    mpicxx --showme:command
    echo
    echo "mpirun:"
    command -v mpirun
    mpirun --version
    echo
    echo "cmake:"
    command -v cmake
    cmake --version
    echo
    echo "slurm_allocation:"
    scontrol show job "${SLURM_JOB_ID}"
} | tee "${METADATA_LOG}"

###############################################################################
# Smoke tests
###############################################################################

echo
echo "======================================================================"
echo "Running CUDA/MPI smoke tests"
echo "======================================================================"

BUILD_DIR="${SMOKE_BUILD_DIR}" \
EXECUTABLE_DIR="${SMOKE_EXECUTABLE_DIR}" \
MPI_RANKS="${MPI_RANKS}" \
CMAKE_GENERATOR="${CMAKE_GENERATOR}" \
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
bash "${SMOKE_DIR}/run_smoke_tests.sh" \
    | tee "${SMOKE_LOG}"

###############################################################################
# Matrix benchmark
###############################################################################

echo
echo "======================================================================"
echo "Running matrix benchmark"
echo "======================================================================"

BUILD_DIR="${MATRIX_BUILD_DIR}" \
EXECUTABLE_DIR="${MATRIX_EXECUTABLE_DIR}" \
DATA_DIR="${DATA_DIR}" \
MATRIX_N="${MATRIX_N}" \
MPI_RANKS="${MPI_RANKS}" \
REPEATS="${REPEATS}" \
CYCLES="${CYCLES}" \
HOLD_MS="${HOLD_MS}" \
ATOL="${ATOL}" \
RTOL="${RTOL}" \
CMAKE_GENERATOR="${CMAKE_GENERATOR}" \
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
bash "${MATRIX_DIR}/run_matrix_benchmark.sh"

###############################################################################
# Validate benchmark output
###############################################################################

echo
echo "======================================================================"
echo "Validating benchmark results"
echo "======================================================================"

if [[ ! -f "${COMPARE_LOG}" ]]; then
    echo "ERROR: comparison log was not created: ${COMPARE_LOG}" >&2
    exit 1
fi

COMPARISON_PASSES="$(
    grep -c 'failures=0' "${COMPARE_LOG}" || true
)"

if [[ "${COMPARISON_PASSES}" -ne 2 ]]; then
    echo "ERROR: expected two successful matrix comparisons." >&2
    echo "Found ${COMPARISON_PASSES} successful comparisons." >&2
    cat "${COMPARE_LOG}" >&2
    exit 1
fi

if grep -q 'comparison failed' "${COMPARE_LOG}"; then
    echo "ERROR: matrix comparison reported a failure." >&2
    cat "${COMPARE_LOG}" >&2
    exit 1
fi

echo "Both AB and BA comparisons passed."

###############################################################################
# Final summary
###############################################################################

echo
echo "======================================================================"
echo "Performance summary"
echo "======================================================================"

grep 'finished all.*aggregate_approx_gflop_s=' \
    "${DATA_DIR}/matrix_products_tiled.out" \
    "${DATA_DIR}/matrix_products_cublas.out"

echo
echo "Comparison summary:"
cat "${COMPARE_LOG}"

echo
echo "Final GPU status:"
nvidia-smi \
    --query-gpu=name,uuid,pstate,temperature.gpu,power.draw,memory.used,memory.total \
    --format=csv

echo
echo "Result files:"
find "${DATA_DIR}" -maxdepth 1 -type f \
    -printf '%f\t%s bytes\n' | sort

echo
du -sh "${DATA_DIR}"

echo
echo "VISION matrix benchmark completed successfully."
