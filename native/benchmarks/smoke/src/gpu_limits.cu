#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, device);

    std::cout << "GPU: " << p.name << '\n'
              << "Compute capability: " << p.major << '.' << p.minor << '\n'
              << "SMs: " << p.multiProcessorCount << '\n'
              << "Max threads/block: " << p.maxThreadsPerBlock << '\n'
              << "Max block dimensions: "
              << p.maxThreadsDim[0] << " x "
              << p.maxThreadsDim[1] << " x "
              << p.maxThreadsDim[2] << '\n'
              << "Max grid dimensions (blocks): "
              << p.maxGridSize[0] << " x "
              << p.maxGridSize[1] << " x "
              << p.maxGridSize[2] << '\n'
              << "Max threads/SM: " << p.maxThreadsPerMultiProcessor << '\n';
}