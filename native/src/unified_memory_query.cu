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

static int get_attr(cudaDeviceAttr attr, int device) {
    int value = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, device));
    return value;
}

static void print_unified_memory_mode(int device) {
    int concurrent_managed = get_attr(cudaDevAttrConcurrentManagedAccess, device);
    int pageable_access = get_attr(cudaDevAttrPageableMemoryAccess, device);
    int host_page_tables = get_attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
    int direct_managed_host = get_attr(cudaDevAttrDirectManagedMemAccessFromHost, device);
    int host_native_atomics = get_attr(cudaDevAttrHostNativeAtomicSupported, device);

    std::printf("  cudaDevAttrConcurrentManagedAccess=%d\n", concurrent_managed);
    std::printf("  cudaDevAttrPageableMemoryAccess=%d\n", pageable_access);
    std::printf("  cudaDevAttrPageableMemoryAccessUsesHostPageTables=%d\n", host_page_tables);
    std::printf("  cudaDevAttrDirectManagedMemAccessFromHost=%d\n", direct_managed_host);
    std::printf("  cudaDevAttrHostNativeAtomicSupported=%d\n", host_native_atomics);

    if (!concurrent_managed) {
        std::printf("  mode: limited unified memory support\n");
    } else if (!pageable_access) {
        std::printf("  mode: full support for cudaMallocManaged allocations\n");
    } else if (host_page_tables) {
        std::printf("  mode: full system-memory access with hardware coherency/ATS\n");
    } else {
        std::printf("  mode: full system-memory access with software coherency/HMM\n");
    }
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("unified_memory_query: device_count=%d\n", device_count);

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("device %d: %s\n", device, prop.name);
        std::printf("  compute capability=%d.%d\n", prop.major, prop.minor);
        std::printf("  totalGlobalMem=%zu bytes\n", static_cast<size_t>(prop.totalGlobalMem));
        print_unified_memory_mode(device);
    }

    return 0;
}
