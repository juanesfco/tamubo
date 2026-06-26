#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
MPI_RANKS="${MPI_RANKS:-2}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja
cmake --build "${BUILD_DIR}"

"${BUILD_DIR}/cuda_smoke"
"${BUILD_DIR}/unified_memory_query"
mpirun -np "${MPI_RANKS}" "${BUILD_DIR}/mpi_smoke"
mpirun -np "${MPI_RANKS}" "${BUILD_DIR}/cuda_mpi_smoke"
