#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(status));             \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void add_one(double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += 1.0;
    }
}

int main() {
    constexpr int n = 4;
    double h[n] = {1.0, 2.0, 3.0, 4.0};
    double* d = nullptr;

    CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(double), cudaMemcpyHostToDevice));

    add_one<<<1, n>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    std::printf("cuda_smoke: %.1f %.1f %.1f %.1f\n", h[0], h[1], h[2], h[3]);
    return 0;
}
