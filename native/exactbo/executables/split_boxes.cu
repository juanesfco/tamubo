#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

constexpr char kInputMagic[8] = {'T', 'S', 'P', 'L', 'I', 'N', '1', '!'};
constexpr char kOutputMagic[8] = {'T', 'S', 'P', 'L', 'O', 'U', '1', '!'};

int current_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

std::string format_bytes(std::size_t bytes) {
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << (static_cast<double>(bytes) / gib) << " GiB";
    return out.str();
}

void print_device_memory(int rank, const std::string& label) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::size_t used_bytes = total_bytes - free_bytes;
    std::cout << "rank " << rank << " memory [" << label << "] "
              << "used=" << format_bytes(used_bytes)
              << " free=" << format_bytes(free_bytes)
              << " total=" << format_bytes(total_bytes) << "\n";
}

struct Options {
    std::string input_path;
    std::string output_path;
    // Zero selects a conservative batch size from the currently free device
    // memory. A positive value is an upper bound, not a forced allocation.
    std::uint64_t split_batch_parents = 0;
};

struct NodePlacement {
    int local_rank = 0;
    int local_size = 1;
};

NodePlacement node_placement() {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    NodePlacement placement;
    MPI_Comm_rank(local_comm, &placement.local_rank);
    MPI_Comm_size(local_comm, &placement.local_size);
    MPI_Comm_free(&local_comm);
    return placement;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* flag) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--input") == 0) {
            options.input_path = need_value("--input");
        } else if (std::strcmp(argv[i], "--output") == 0) {
            options.output_path = need_value("--output");
        } else if (std::strcmp(argv[i], "--split-batch-parents") == 0) {
            const char* value = need_value("--split-batch-parents");
            if (value[0] == '-') {
                throw std::runtime_error("--split-batch-parents must be a nonnegative integer");
            }
            std::size_t parsed = 0;
            options.split_batch_parents = std::stoull(value, &parsed);
            if (value[parsed] != '\0') {
                throw std::runtime_error("--split-batch-parents must be a nonnegative integer");
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: exactbo_split_boxes --input input.bin --output output.bin\n"
                      << "       [--split-batch-parents N]\n"
                      << "N=0 (the default) chooses a safe batch size automatically.\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    if (options.input_path.empty() || options.output_path.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    return options;
}

template <class T>
void read_value(std::istream& in, T& value, const std::string& label) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read " + label);
    }
}

void read_doubles(std::istream& in, std::vector<double>& values, const std::string& label) {
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("failed to read " + label);
    }
}

void check_no_trailing_bytes(std::istream& in) {
    char trailing = 0;
    in.read(&trailing, 1);
    if (in.gcount() != 0) {
        throw std::runtime_error("input file has trailing bytes");
    }
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const std::string& label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::runtime_error(label + " overflows uint64");
    }
    return a * b;
}

int checked_int_count(std::uint64_t value, const std::string& label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(label + " is too large for MPI_Gatherv int counts");
    }
    return static_cast<int>(value);
}

std::uint64_t local_start_for_rank(std::uint64_t n, int rank, int size) {
    std::uint64_t base = n / static_cast<std::uint64_t>(size);
    std::uint64_t rem = n % static_cast<std::uint64_t>(size);
    return static_cast<std::uint64_t>(rank) * base + std::min<std::uint64_t>(rank, rem);
}

std::uint64_t local_count_for_rank(std::uint64_t n, int rank, int size) {
    std::uint64_t base = n / static_cast<std::uint64_t>(size);
    std::uint64_t rem = n % static_cast<std::uint64_t>(size);
    return base + (static_cast<std::uint64_t>(rank) < rem ? 1 : 0);
}

struct SplitInput {
    std::uint64_t n = 0;
    std::uint64_t d = 0;
    std::uint64_t keep_inactive = 1;
    std::vector<double> domain_width;
    std::vector<double> bounds_L;
    std::vector<double> bounds_U;
    std::vector<unsigned char> active_mask;
};

SplitInput read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }

    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kInputMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid split_boxes input magic");
    }

    SplitInput input;
    read_value(in, input.n, "n");
    read_value(in, input.d, "d");
    read_value(in, input.keep_inactive, "keep_inactive");

    if (input.d == 0) {
        throw std::runtime_error("d must be > 0");
    }
    std::uint64_t elements = checked_mul(input.n, input.d, "n*d");

    input.domain_width.resize(input.d);
    input.bounds_L.resize(elements);
    input.bounds_U.resize(elements);
    input.active_mask.resize(input.n);

    read_doubles(in, input.domain_width, "domain_width");
    read_doubles(in, input.bounds_L, "bounds_L");
    read_doubles(in, input.bounds_U, "bounds_U");
    in.read(reinterpret_cast<char*>(input.active_mask.data()), static_cast<std::streamsize>(input.active_mask.size()));
    if (!in) {
        throw std::runtime_error("failed to read active_mask");
    }
    check_no_trailing_bytes(in);
    return input;
}

__device__ double normalized_width(const double* bounds_L, const double* bounds_U, const double* domain_width, std::uint64_t src, std::uint64_t d, std::uint64_t dim) {
    double width = bounds_U[src * d + dim] - bounds_L[src * d + dim];
    double scale = domain_width[dim];
    return scale != 0.0 ? width / scale : width;
}

__device__ std::uint64_t split_dim_for_rank(const double* bounds_L, const double* bounds_U, const double* domain_width, std::uint64_t src, std::uint64_t d, std::uint64_t rank) {
    for (std::uint64_t dim = 0; dim < d; ++dim) {
        double score = normalized_width(bounds_L, bounds_U, domain_width, src, d, dim);
        std::uint64_t order = 0;
        for (std::uint64_t other = 0; other < d; ++other) {
            double other_score = normalized_width(bounds_L, bounds_U, domain_width, src, d, other);
            if (other_score > score || (other_score == score && other < dim)) {
                ++order;
            }
        }
        if (order == rank) {
            return dim;
        }
    }
    return d - 1;
}

__global__ void split_active_boxes_kernel(
    const double* bounds_L,
    const double* bounds_U,
    const double* domain_width,
    const std::uint64_t* active_indices,
    std::uint64_t active_count,
    std::uint64_t d,
    std::uint64_t stride,
    double* out_L,
    double* out_U) {
    for (std::uint64_t active_pos = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         active_pos < active_count;
         active_pos += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        std::uint64_t src = active_indices == nullptr ? active_pos : active_indices[active_pos];
        std::uint64_t out_box_base = active_pos * stride;

        for (std::uint64_t child = 0; child < stride; ++child) {
            for (std::uint64_t dim = 0; dim < d; ++dim) {
                out_L[(out_box_base + child) * d + dim] = bounds_L[src * d + dim];
                out_U[(out_box_base + child) * d + dim] = bounds_U[src * d + dim];
            }
        }

        for (std::uint64_t rank = 0; rank < d; ++rank) {
            std::uint64_t dim = split_dim_for_rank(bounds_L, bounds_U, domain_width, src, d, rank);
            double low = bounds_L[src * d + dim];
            double high = bounds_U[src * d + dim];
            double third_width = (high - low) / 3.0;
            double lower_third = low + third_width;
            double upper_third = high - third_width;
            std::uint64_t lower_row = 2 * rank;
            std::uint64_t upper_row = lower_row + 1;

            out_U[(out_box_base + lower_row) * d + dim] = lower_third;
            out_L[(out_box_base + upper_row) * d + dim] = upper_third;

            for (std::uint64_t child = upper_row + 1; child < stride; ++child) {
                out_L[(out_box_base + child) * d + dim] = lower_third;
                out_U[(out_box_base + child) * d + dim] = upper_third;
            }
        }
    }
}

void copy_to_device(void** dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        *dst = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(dst, bytes));
    CUDA_CHECK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
}

void allocate_device(void** dst, std::size_t bytes) {
    if (bytes == 0) {
        *dst = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(dst, bytes));
}

void free_device(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void check_mpi(int status, const char* label) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(label) + ": " + std::string(message, static_cast<std::size_t>(length)));
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const std::string& label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::runtime_error(label + " overflows uint64");
    }
    return a + b;
}

MPI_Offset checked_mpi_offset(std::uint64_t value, const std::string& label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<MPI_Offset>::max())) {
        throw std::runtime_error(label + " is too large for MPI_Offset");
    }
    return static_cast<MPI_Offset>(value);
}

struct SplitCounts {
    std::uint64_t local_active = 0;
    std::uint64_t local_inactive = 0;
    std::uint64_t global_active = 0;
    std::uint64_t global_inactive = 0;
    std::uint64_t active_prefix = 0;
    std::uint64_t inactive_prefix = 0;
};

SplitCounts count_split_rows(const SplitInput& input, int rank, int size) {
    SplitCounts counts;
    const std::uint64_t start = local_start_for_rank(input.n, rank, size);
    const std::uint64_t local_n = local_count_for_rank(input.n, rank, size);
    for (std::uint64_t i = 0; i < local_n; ++i) {
        if (input.active_mask[start + i] != 0) {
            ++counts.local_active;
        } else if (input.keep_inactive != 0) {
            ++counts.local_inactive;
        }
    }

    unsigned long long local_active = static_cast<unsigned long long>(counts.local_active);
    unsigned long long local_inactive = static_cast<unsigned long long>(counts.local_inactive);
    unsigned long long global_active = 0;
    unsigned long long global_inactive = 0;
    unsigned long long active_prefix = 0;
    unsigned long long inactive_prefix = 0;
    check_mpi(MPI_Allreduce(&local_active, &global_active, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce(active count)");
    check_mpi(MPI_Allreduce(&local_inactive, &global_inactive, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Allreduce(inactive count)");
    check_mpi(MPI_Exscan(&local_active, &active_prefix, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Exscan(active count)");
    check_mpi(MPI_Exscan(&local_inactive, &inactive_prefix, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
              "MPI_Exscan(inactive count)");
    if (rank == 0) {
        active_prefix = 0;
        inactive_prefix = 0;
    }
    counts.global_active = static_cast<std::uint64_t>(global_active);
    counts.global_inactive = static_cast<std::uint64_t>(global_inactive);
    counts.active_prefix = static_cast<std::uint64_t>(active_prefix);
    counts.inactive_prefix = static_cast<std::uint64_t>(inactive_prefix);
    return counts;
}

std::uint64_t choose_split_batch_parents(
    std::uint64_t requested,
    std::uint64_t local_parent_count,
    std::uint64_t d,
    std::uint64_t stride,
    int ranks_sharing_device) {
    constexpr std::uint64_t kAutoHardCap = 1ULL << 20;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    // Dense parent L/U plus all child L/U arrays. Use at most one quarter of
    // the reported free memory per rank, divided again when ranks share a GPU.
    // The other quarters cover the mirrored host buffers and CUDA/MPI overhead;
    // this is also conservative on unified-memory systems.
    const std::uint64_t parent_and_children = checked_add(stride, 1, "split stride+parent");
    const std::uint64_t values_per_parent = checked_mul(
        checked_mul(2, d, "split values per parent"), parent_and_children,
        "split values per parent");
    const std::uint64_t bytes_per_parent = checked_mul(
        values_per_parent, sizeof(double), "split bytes per parent");
    const std::uint64_t sharing = static_cast<std::uint64_t>(std::max(1, ranks_sharing_device));
    const std::uint64_t divisor = checked_mul(4, sharing, "split memory safety divisor");
    const std::uint64_t budget = static_cast<std::uint64_t>(free_bytes) / divisor;
    std::uint64_t automatic = bytes_per_parent == 0 ? 1 : budget / bytes_per_parent;
    automatic = std::max<std::uint64_t>(1, std::min(automatic, kAutoHardCap));

    std::uint64_t chosen = requested == 0 ? automatic : std::min(requested, automatic);
    chosen = std::max<std::uint64_t>(1, chosen);
    if (local_parent_count != 0) {
        chosen = std::min(chosen, local_parent_count);
    }
    return chosen;
}

constexpr std::uint64_t kOutputHeaderBytes = sizeof(kOutputMagic) + 4 * sizeof(std::uint64_t);

MPI_File open_streamed_output(
    const std::string& path,
    std::uint64_t n_out,
    std::uint64_t d,
    std::uint64_t n_active,
    std::uint64_t stride,
    int rank) {
    const std::uint64_t elements = checked_mul(n_out, d, "split output elements");
    const std::uint64_t plane_bytes = checked_mul(elements, sizeof(double), "split output plane bytes");
    const std::uint64_t total_bytes = checked_add(
        kOutputHeaderBytes, checked_mul(2, plane_bytes, "split output data bytes"),
        "split output file bytes");

    MPI_File file = MPI_FILE_NULL;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR,
                            MPI_INFO_NULL, &file),
              "MPI_File_open(output)");
    const MPI_Offset output_size =
        checked_mpi_offset(total_bytes, "split output file bytes");
    // Some ROMIO/filesystem combinations extend the visible file while
    // preallocating. Reserve first, then enforce the format's exact size.
    check_mpi(MPI_File_preallocate(file, output_size),
              "MPI_File_preallocate(output)");
    check_mpi(MPI_File_set_size(file, output_size),
              "MPI_File_set_size(output)");

    if (rank == 0) {
        std::array<char, static_cast<std::size_t>(kOutputHeaderBytes)> header{};
        std::size_t offset = 0;
        auto append = [&](const void* source, std::size_t bytes) {
            std::memcpy(header.data() + offset, source, bytes);
            offset += bytes;
        };
        append(kOutputMagic, sizeof(kOutputMagic));
        append(&n_out, sizeof(n_out));
        append(&d, sizeof(d));
        append(&n_active, sizeof(n_active));
        append(&stride, sizeof(stride));
        check_mpi(MPI_File_write_at(file, 0, header.data(), static_cast<int>(header.size()),
                                    MPI_BYTE, MPI_STATUS_IGNORE),
                  "MPI_File_write_at(output header)");
    }
    check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(output header)");
    return file;
}

void write_output_rows(
    MPI_File file,
    std::uint64_t output_row,
    std::uint64_t rows,
    std::uint64_t n_out,
    std::uint64_t d,
    const double* bounds_L,
    const double* bounds_U) {
    if (rows == 0) {
        return;
    }
    const std::uint64_t elements = checked_mul(rows, d, "streamed output batch elements");
    const int mpi_count = checked_int_count(elements, "streamed output batch elements");
    const std::uint64_t plane_elements = checked_mul(n_out, d, "streamed output plane elements");
    const std::uint64_t element_offset = checked_mul(output_row, d, "streamed output element offset");
    const std::uint64_t lower_bytes = checked_add(
        kOutputHeaderBytes, checked_mul(element_offset, sizeof(double), "streamed lower byte offset"),
        "streamed lower file offset");
    const std::uint64_t upper_element_offset = checked_add(
        plane_elements, element_offset, "streamed upper element offset");
    const std::uint64_t upper_bytes = checked_add(
        kOutputHeaderBytes, checked_mul(upper_element_offset, sizeof(double), "streamed upper byte offset"),
        "streamed upper file offset");

    check_mpi(MPI_File_write_at(file, checked_mpi_offset(lower_bytes, "streamed lower file offset"),
                                bounds_L, mpi_count, MPI_DOUBLE, MPI_STATUS_IGNORE),
              "MPI_File_write_at(lower bounds)");
    check_mpi(MPI_File_write_at(file, checked_mpi_offset(upper_bytes, "streamed upper file offset"),
                                bounds_U, mpi_count, MPI_DOUBLE, MPI_STATUS_IGNORE),
              "MPI_File_write_at(upper bounds)");
}

void stream_local_split(
    const SplitInput& input,
    const Options& options,
    const SplitCounts& counts,
    MPI_File output,
    int rank,
    int size,
    int device,
    int ranks_sharing_device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::uint64_t stride = 2 * input.d + 1;
    const std::uint64_t start = local_start_for_rank(input.n, rank, size);
    const std::uint64_t local_n = local_count_for_rank(input.n, rank, size);
    const std::uint64_t local_parent_count = checked_add(
        counts.local_active, counts.local_inactive, "local split parent count");
    const std::uint64_t batch_parents = choose_split_batch_parents(
        options.split_batch_parents, local_parent_count, input.d, stride, ranks_sharing_device);
    const std::uint64_t n_out = checked_add(
        checked_mul(counts.global_active, stride, "global active output rows"),
        counts.global_inactive, "global output rows");

    std::cout << "rank " << rank << " streaming split_boxes pid=" << getpid()
              << " launch_cpu=" << current_cpu()
              << " gpu_device=" << device << " gpu_name=\"" << prop.name << "\""
              << " rows=[" << start << "," << (start + local_n) << ")"
              << " local_active=" << counts.local_active
              << " local_inactive=" << counts.local_inactive
              << " split_batch_parents=" << batch_parents << "\n";

    if (counts.local_active != 0) {
        const std::uint64_t active_capacity = std::min(batch_parents, counts.local_active);
        const std::uint64_t parent_elements = checked_mul(active_capacity, input.d, "split parent batch elements");
        const std::uint64_t child_rows = checked_mul(active_capacity, stride, "split child batch rows");
        const std::uint64_t child_elements = checked_mul(child_rows, input.d, "split child batch elements");
        std::vector<double> parent_L(parent_elements);
        std::vector<double> parent_U(parent_elements);
        std::vector<double> child_L(child_elements);
        std::vector<double> child_U(child_elements);

        double* d_parent_L = nullptr;
        double* d_parent_U = nullptr;
        double* d_domain_width = nullptr;
        double* d_child_L = nullptr;
        double* d_child_U = nullptr;
        const std::size_t parent_bytes = static_cast<std::size_t>(parent_elements) * sizeof(double);
        const std::size_t child_bytes = static_cast<std::size_t>(child_elements) * sizeof(double);
        const std::size_t domain_bytes = input.domain_width.size() * sizeof(double);

        print_device_memory(rank, "before split batch cudaMalloc");
        allocate_device(reinterpret_cast<void**>(&d_parent_L), parent_bytes);
        allocate_device(reinterpret_cast<void**>(&d_parent_U), parent_bytes);
        copy_to_device(reinterpret_cast<void**>(&d_domain_width), input.domain_width.data(), domain_bytes);
        allocate_device(reinterpret_cast<void**>(&d_child_L), child_bytes);
        allocate_device(reinterpret_cast<void**>(&d_child_U), child_bytes);
        print_device_memory(rank, "after split batch cudaMalloc");

        std::uint64_t batch_count = 0;
        std::uint64_t output_row = checked_mul(counts.active_prefix, stride, "rank active output offset");
        auto flush_active = [&]() {
            if (batch_count == 0) {
                return;
            }
            const std::uint64_t batch_parent_elements = checked_mul(batch_count, input.d, "active batch parent elements");
            const std::uint64_t batch_rows = checked_mul(batch_count, stride, "active batch output rows");
            const std::uint64_t batch_child_elements = checked_mul(batch_rows, input.d, "active batch child elements");
            const std::size_t batch_parent_bytes = static_cast<std::size_t>(batch_parent_elements) * sizeof(double);
            const std::size_t batch_child_bytes = static_cast<std::size_t>(batch_child_elements) * sizeof(double);
            CUDA_CHECK(cudaMemcpy(d_parent_L, parent_L.data(), batch_parent_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_parent_U, parent_U.data(), batch_parent_bytes, cudaMemcpyHostToDevice));
            const int block = 128;
            const int grid = std::max(1, std::min(
                static_cast<int>((batch_count + block - 1) / block), prop.maxGridSize[0]));
            split_active_boxes_kernel<<<grid, block>>>(
                d_parent_L, d_parent_U, d_domain_width, nullptr, batch_count,
                input.d, stride, d_child_L, d_child_U);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(child_L.data(), d_child_L, batch_child_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(child_U.data(), d_child_U, batch_child_bytes, cudaMemcpyDeviceToHost));
            write_output_rows(output, output_row, batch_rows, n_out, input.d,
                              child_L.data(), child_U.data());
            output_row += batch_rows;
            batch_count = 0;
        };

        for (std::uint64_t i = 0; i < local_n; ++i) {
            const std::uint64_t global_row = start + i;
            if (input.active_mask[global_row] == 0) {
                continue;
            }
            std::copy_n(input.bounds_L.data() + global_row * input.d, input.d,
                        parent_L.data() + batch_count * input.d);
            std::copy_n(input.bounds_U.data() + global_row * input.d, input.d,
                        parent_U.data() + batch_count * input.d);
            ++batch_count;
            if (batch_count == active_capacity) {
                flush_active();
            }
        }
        flush_active();

        free_device(d_child_U);
        free_device(d_child_L);
        free_device(d_domain_width);
        free_device(d_parent_U);
        free_device(d_parent_L);
        CUDA_CHECK(cudaDeviceSynchronize());
        print_device_memory(rank, "after split batch cudaFree");
    }

    if (counts.local_inactive != 0) {
        const std::uint64_t inactive_capacity = std::min(batch_parents, counts.local_inactive);
        const std::uint64_t inactive_elements = checked_mul(
            inactive_capacity, input.d, "inactive output batch elements");
        std::vector<double> inactive_L(inactive_elements);
        std::vector<double> inactive_U(inactive_elements);
        std::uint64_t batch_count = 0;
        std::uint64_t output_row = checked_add(
            checked_mul(counts.global_active, stride, "inactive output base"),
            counts.inactive_prefix, "rank inactive output offset");
        auto flush_inactive = [&]() {
            if (batch_count == 0) {
                return;
            }
            write_output_rows(output, output_row, batch_count, n_out, input.d,
                              inactive_L.data(), inactive_U.data());
            output_row += batch_count;
            batch_count = 0;
        };

        for (std::uint64_t i = 0; i < local_n; ++i) {
            const std::uint64_t global_row = start + i;
            if (input.active_mask[global_row] != 0) {
                continue;
            }
            std::copy_n(input.bounds_L.data() + global_row * input.d, input.d,
                        inactive_L.data() + batch_count * input.d);
            std::copy_n(input.bounds_U.data() + global_row * input.d, input.d,
                        inactive_U.data() + batch_count * input.d);
            ++batch_count;
            if (batch_count == inactive_capacity) {
                flush_inactive();
            }
        }
        flush_inactive();
    }
}
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        Options options = parse_args(argc, argv);
        SplitInput input = read_input(options.input_path);
        const std::uint64_t stride = 2 * input.d + 1;

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA devices visible");
        }
        const NodePlacement placement = node_placement();
        const int device = placement.local_rank % device_count;
        int ranks_sharing_device = 0;
        for (int local = 0; local < placement.local_size; ++local) {
            ranks_sharing_device += (local % device_count) == device ? 1 : 0;
        }

        const SplitCounts counts = count_split_rows(input, rank, size);
        const std::uint64_t active_rows = checked_mul(counts.global_active, stride, "active output rows");
        const std::uint64_t n_out = checked_add(active_rows, counts.global_inactive, "split output rows");
        if (rank == 0) {
            std::cout << "exactbo_split_boxes: mpi_size=" << size
                      << " device_count=" << device_count
                      << " input=\"" << options.input_path << "\""
                      << " output=\"" << options.output_path << "\""
                      << " n=" << input.n << " d=" << input.d
                      << " active=" << counts.global_active << " stride=" << stride
                      << " keep_inactive=" << input.keep_inactive
                      << " n_out=" << n_out
                      << " requested_split_batch_parents=" << options.split_batch_parents << "\n";
        }
        std::cout << "rank " << rank
                  << " local_rank=" << placement.local_rank
                  << " local_size=" << placement.local_size
                  << " using gpu_device=" << device
                  << " ranks_sharing_device=" << ranks_sharing_device << "\n";

        MPI_File output = open_streamed_output(
            options.output_path, n_out, input.d, counts.global_active, stride, rank);
        stream_local_split(
            input, options, counts, output, rank, size, device, ranks_sharing_device);
        check_mpi(MPI_File_sync(output), "MPI_File_sync(output)");
        check_mpi(MPI_File_close(&output), "MPI_File_close(output)");
        if (rank == 0) {
            std::cout << "rank 0 wrote streamed split_boxes output to " << options.output_path << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " exactbo_split_boxes: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    cudaDeviceReset();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
