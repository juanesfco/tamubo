cmake_minimum_required(VERSION 3.24)
project(tamubo_native LANGUAGES CXX CUDA)

# Keep editor tooling synchronized with the actual CUDA/MPI target flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(benchmarks)
add_subdirectory(exactbo)
