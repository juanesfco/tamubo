add_executable(matrix_make_inputs matrix_make_inputs.cpp)

add_executable(matrix_products_mpi matrix_products_mpi.cu)
target_link_libraries(matrix_products_mpi PRIVATE MPI::MPI_CXX)

add_executable(matrix_products_cublas_mpi matrix_products_cublas_mpi.cu)
target_link_libraries(matrix_products_cublas_mpi PRIVATE MPI::MPI_CXX CUDA::cublas)

add_executable(matrix_compare matrix_compare.cpp)
