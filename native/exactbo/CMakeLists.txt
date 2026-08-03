cmake_minimum_required(VERSION 3.24)
project(tamubo_exactbo LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/executables")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(tamubo_exactbo_native STATIC src/box_store.cpp)
target_include_directories(tamubo_exactbo_native PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(exactbo_partitioning src/partitioning.cu)
target_link_libraries(exactbo_partitioning PRIVATE MPI::MPI_CXX tamubo_exactbo_native)

# Small teaching/debugging programs. They deliberately avoid MPI, binary
# files, batching, and BoxStore so each algorithm can be inspected by itself.
add_executable(exactbo_bounding src/bounding.cu)
add_executable(exactbo_box_partition src/box_partition.cu)
