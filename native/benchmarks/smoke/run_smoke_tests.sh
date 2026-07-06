#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${NATIVE_DIR}/build"
MPI_RANKS="${MPI_RANKS:-2}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"

if [ -z "${CMAKE_GENERATOR}" ] && [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  CACHED_GENERATOR="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  if [ -n "${CACHED_GENERATOR}" ]; then
    CMAKE_GENERATOR="${CACHED_GENERATOR}"
  fi
fi

if [ -z "${CMAKE_GENERATOR}" ]; then
  if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
  else
    CMAKE_GENERATOR="Unix Makefiles"
  fi
fi

cmake_args=(-S "${NATIVE_DIR}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}")
if [ -n "${CUDA_ARCHITECTURES}" ]; then
  cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}"

"${BUILD_DIR}/cuda_smoke"
"${BUILD_DIR}/unified_memory_query"
mpirun -np "${MPI_RANKS}" "${BUILD_DIR}/mpi_smoke"
mpirun -np "${MPI_RANKS}" "${BUILD_DIR}/cuda_mpi_smoke"
