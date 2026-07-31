cmake_minimum_required(VERSION 3.24)
project(tamubo_matrix_benchmarks LANGUAGES CXX CUDA)

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

add_executable(matrix_make_inputs src/matrix_make_inputs.cpp)

add_executable(matrix_products_mpi src/matrix_products_mpi.cu)
target_link_libraries(matrix_products_mpi PRIVATE MPI::MPI_CXX)

add_executable(matrix_products_cublas_mpi src/matrix_products_cublas_mpi.cu)
target_link_libraries(matrix_products_cublas_mpi PRIVATE MPI::MPI_CXX CUDA::cublas)

add_executable(matrix_compare src/matrix_compare.cpp)
