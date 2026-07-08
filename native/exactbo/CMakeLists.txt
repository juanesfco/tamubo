add_library(tamubo_exactbo_native INTERFACE)
target_include_directories(tamubo_exactbo_native INTERFACE
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
