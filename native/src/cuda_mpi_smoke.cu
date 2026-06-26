#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(status));             \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                         \
        }                                                                    \
    } while (0)

__global__ void fill_rank(double* x, int n, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = static_cast<double>(rank);
    }
}

int current_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME] = {};
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "cuda_mpi_smoke: no CUDA devices visible\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int device = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    constexpr int n = 4;
    double h[n] = {};
    double* d = nullptr;

    CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    int launch_cpu = current_cpu();
    fill_rank<<<1, n>>>(d, n, rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    std::printf(
        "cuda_mpi_smoke: rank %d/%d host=%s pid=%d launch_cpu=%d "
        "gpu_device=%d/%d gpu_name=\"%s\" gpu_pci=%04x:%02x:%02x value=%.1f\n",
        rank,
        size,
        processor_name,
        getpid(),
        launch_cpu,
        device,
        device_count,
        prop.name,
        prop.pciDomainID,
        prop.pciBusID,
        prop.pciDeviceID,
        h[0]);

    MPI_Finalize();
    return 0;
}
