add_executable(cuda_smoke cuda_smoke.cu)

add_executable(mpi_smoke mpi_smoke.cpp)
target_link_libraries(mpi_smoke PRIVATE MPI::MPI_CXX)

add_executable(cuda_mpi_smoke cuda_mpi_smoke.cu)
target_link_libraries(cuda_mpi_smoke PRIVATE MPI::MPI_CXX)

add_executable(unified_memory_query unified_memory_query.cu)
