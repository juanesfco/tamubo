add_library(tamubo_exactbo_native STATIC
  src/box_store.cpp
)
target_include_directories(tamubo_exactbo_native PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(exactbo_rbf_k_bounds executables/rbf_k_bounds.cu)
target_link_libraries(exactbo_rbf_k_bounds PRIVATE MPI::MPI_CXX)

add_executable(exactbo_mu_bounds executables/mu_bounds.cu)
target_link_libraries(exactbo_mu_bounds PRIVATE MPI::MPI_CXX)

add_executable(exactbo_sigma_bounds executables/sigma_bounds.cu)
target_link_libraries(exactbo_sigma_bounds PRIVATE MPI::MPI_CXX)

add_executable(exactbo_ei_bounds executables/ei_bounds.cu)
target_link_libraries(exactbo_ei_bounds PRIVATE MPI::MPI_CXX)

add_executable(exactbo_split_boxes executables/split_boxes.cu)
target_link_libraries(exactbo_split_boxes PRIVATE MPI::MPI_CXX tamubo_exactbo_native)

add_executable(exactbo_partitioning executables/partitioning.cu)
target_link_libraries(exactbo_partitioning PRIVATE MPI::MPI_CXX tamubo_exactbo_native)
