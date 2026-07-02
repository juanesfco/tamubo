cmake_minimum_required(VERSION 3.24)
project(tamubo_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cuda_smoke src/cuda_smoke.cu)

add_executable(mpi_smoke src/mpi_smoke.cpp)
target_link_libraries(mpi_smoke PRIVATE MPI::MPI_CXX)

add_executable(cuda_mpi_smoke src/cuda_mpi_smoke.cu)
target_link_libraries(cuda_mpi_smoke PRIVATE MPI::MPI_CXX)

add_executable(unified_memory_query src/unified_memory_query.cu)

add_executable(matrix_make_inputs src/matrix_make_inputs.cpp)

add_executable(matrix_products_mpi src/matrix_products_mpi.cu)
target_link_libraries(matrix_products_mpi PRIVATE MPI::MPI_CXX)

add_executable(matrix_products_cublas_mpi src/matrix_products_cublas_mpi.cu)
target_link_libraries(matrix_products_cublas_mpi PRIVATE MPI::MPI_CXX CUDA::cublas)

add_executable(matrix_compare src/matrix_compare.cpp)
