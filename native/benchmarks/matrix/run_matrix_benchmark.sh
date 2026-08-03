#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/executables/build}"
EXECUTABLE_DIR="${EXECUTABLE_DIR:-${SCRIPT_DIR}/executables}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"

MATRIX_N="${MATRIX_N:-1024}"
MPI_RANKS="${MPI_RANKS:-1}"
REPEATS="${REPEATS:-5}"
CYCLES="${CYCLES:-3}"
HOLD_MS="${HOLD_MS:-0}"
ATOL="${ATOL:-1e-8}"
RTOL="${RTOL:-1e-8}"
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

mkdir -p "${DATA_DIR}"

cmake_args=(-S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}")
if [ -n "${CUDA_ARCHITECTURES}" ]; then
  cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}"

A_PATH="${DATA_DIR}/A_${MATRIX_N}.bin"
B_PATH="${DATA_DIR}/B_${MATRIX_N}.bin"

TILED_AB="${DATA_DIR}/AB_tiled.bin"
TILED_BA="${DATA_DIR}/BA_tiled.bin"
CUBLAS_AB="${DATA_DIR}/AB_cublas.bin"
CUBLAS_BA="${DATA_DIR}/BA_cublas.bin"

TILED_LOG="${DATA_DIR}/matrix_products_tiled.out"
CUBLAS_LOG="${DATA_DIR}/matrix_products_cublas.out"
COMPARE_LOG="${DATA_DIR}/matrix_compare.out"

"${EXECUTABLE_DIR}/matrix_make_inputs" "${MATRIX_N}" "${A_PATH}" "${B_PATH}"

mpirun -np "${MPI_RANKS}" "${EXECUTABLE_DIR}/matrix_products_mpi" \
  --a "${A_PATH}" \
  --b "${B_PATH}" \
  --out-ab "${TILED_AB}" \
  --out-ba "${TILED_BA}" \
  --repeats "${REPEATS}" \
  --cycles "${CYCLES}" \
  --hold-ms "${HOLD_MS}" \
  | tee "${TILED_LOG}"

mpirun -np "${MPI_RANKS}" "${EXECUTABLE_DIR}/matrix_products_cublas_mpi" \
  --a "${A_PATH}" \
  --b "${B_PATH}" \
  --out-ab "${CUBLAS_AB}" \
  --out-ba "${CUBLAS_BA}" \
  --repeats "${REPEATS}" \
  --cycles "${CYCLES}" \
  --hold-ms "${HOLD_MS}" \
  | tee "${CUBLAS_LOG}"

{
  "${EXECUTABLE_DIR}/matrix_compare" \
    --expected "${TILED_AB}" \
    --actual "${CUBLAS_AB}" \
    --atol "${ATOL}" \
    --rtol "${RTOL}"

  "${EXECUTABLE_DIR}/matrix_compare" \
    --expected "${TILED_BA}" \
    --actual "${CUBLAS_BA}" \
    --atol "${ATOL}" \
    --rtol "${RTOL}"
} | tee "${COMPARE_LOG}"

echo "benchmark outputs:"
echo "  data:      ${DATA_DIR}"
echo "  tiled:     ${TILED_LOG}"
echo "  cuBLAS:    ${CUBLAS_LOG}"
echo "  compare:   ${COMPARE_LOG}"
