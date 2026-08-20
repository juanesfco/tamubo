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
              << "Max threads/SM: " << p.maxThreadsPerMultiProcessor << '\n'
              << "Max shared memory/SM: " << p.sharedMemPerMultiprocessor << " bytes\n"
              << "Max shared memory/block: " << p.sharedMemPerBlock << " bytes\n"
              << "Max registers/SM: " << p.regsPerMultiprocessor << '\n'
              << "Max registers/block: " << p.regsPerBlock << '\n'
              << "Max opt-in per block: " << p.sharedMemPerBlockOptin << " bytes\n";
}