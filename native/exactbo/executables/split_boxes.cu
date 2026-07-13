#include <algorithm>
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
};

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
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: exactbo_split_boxes --input input.bin --output output.bin\n";
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

std::vector<double> slice_rows(const std::vector<double>& values, std::uint64_t start, std::uint64_t count, std::uint64_t stride) {
    auto begin = values.begin() + static_cast<std::ptrdiff_t>(start * stride);
    auto end = begin + static_cast<std::ptrdiff_t>(count * stride);
    return std::vector<double>(begin, end);
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

void write_output(
    const std::string& path,
    std::uint64_t n_out,
    std::uint64_t d,
    std::uint64_t n_active,
    std::uint64_t stride,
    const std::vector<double>& bounds_L,
    const std::vector<double>& bounds_U) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out.write(kOutputMagic, sizeof(kOutputMagic));
    out.write(reinterpret_cast<const char*>(&n_out), sizeof(n_out));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(&n_active), sizeof(n_active));
    out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    out.write(reinterpret_cast<const char*>(bounds_L.data()), static_cast<std::streamsize>(bounds_L.size() * sizeof(double)));
    out.write(reinterpret_cast<const char*>(bounds_U.data()), static_cast<std::streamsize>(bounds_U.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path);
    }
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
        std::uint64_t src = active_indices[active_pos];
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

__global__ void copy_inactive_boxes_kernel(
    const double* bounds_L,
    const double* bounds_U,
    const std::uint64_t* inactive_indices,
    std::uint64_t inactive_count,
    std::uint64_t d,
    double* out_L,
    double* out_U) {
    for (std::uint64_t out_row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_row < inactive_count;
         out_row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        std::uint64_t src = inactive_indices[out_row];
        for (std::uint64_t dim = 0; dim < d; ++dim) {
            out_L[out_row * d + dim] = bounds_L[src * d + dim];
            out_U[out_row * d + dim] = bounds_U[src * d + dim];
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

struct LocalOutput {
    std::uint64_t active_rows = 0;
    std::uint64_t inactive_rows = 0;
    std::vector<double> active_L;
    std::vector<double> active_U;
    std::vector<double> inactive_L;
    std::vector<double> inactive_U;
};

LocalOutput compute_local(const SplitInput& input, int rank, int size, int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::uint64_t stride = 2 * input.d + 1;
    std::uint64_t start = local_start_for_rank(input.n, rank, size);
    std::uint64_t local_n = local_count_for_rank(input.n, rank, size);

    std::vector<double> local_L = slice_rows(input.bounds_L, start, local_n, input.d);
    std::vector<double> local_U = slice_rows(input.bounds_U, start, local_n, input.d);
    std::vector<std::uint64_t> active_indices;
    std::vector<std::uint64_t> inactive_indices;
    active_indices.reserve(local_n);
    inactive_indices.reserve(local_n);
    for (std::uint64_t local_row = 0; local_row < local_n; ++local_row) {
        if (input.active_mask[start + local_row] != 0) {
            active_indices.push_back(local_row);
        } else if (input.keep_inactive != 0) {
            inactive_indices.push_back(local_row);
        }
    }

    LocalOutput output;
    output.active_rows = checked_mul(static_cast<std::uint64_t>(active_indices.size()), stride, "local active output rows");
    output.inactive_rows = static_cast<std::uint64_t>(inactive_indices.size());
    output.active_L.resize(checked_mul(output.active_rows, input.d, "local active output elements"));
    output.active_U.resize(output.active_L.size());
    output.inactive_L.resize(checked_mul(output.inactive_rows, input.d, "local inactive output elements"));
    output.inactive_U.resize(output.inactive_L.size());

    std::cout << "rank " << rank << " starting split_boxes pid=" << getpid()
              << " launch_cpu=" << current_cpu()
              << " gpu_device=" << device << " gpu_name=\"" << prop.name << "\""
              << " rows=[" << start << "," << (start + local_n) << ")"
              << " local_active=" << active_indices.size()
              << " local_inactive=" << inactive_indices.size()
              << " d=" << input.d << " stride=" << stride << "\n";

    if (local_n == 0) {
        return output;
    }

    double* d_bounds_L = nullptr;
    double* d_bounds_U = nullptr;
    double* d_domain_width = nullptr;
    std::uint64_t* d_active_indices = nullptr;
    std::uint64_t* d_inactive_indices = nullptr;
    double* d_active_L = nullptr;
    double* d_active_U = nullptr;
    double* d_inactive_L = nullptr;
    double* d_inactive_U = nullptr;

    std::size_t local_bytes = local_L.size() * sizeof(double);
    std::size_t domain_bytes = input.domain_width.size() * sizeof(double);
    std::size_t active_index_bytes = active_indices.size() * sizeof(std::uint64_t);
    std::size_t inactive_index_bytes = inactive_indices.size() * sizeof(std::uint64_t);
    std::size_t active_bytes = output.active_L.size() * sizeof(double);
    std::size_t inactive_bytes = output.inactive_L.size() * sizeof(double);

    print_device_memory(rank, "before cudaMalloc");
    copy_to_device(reinterpret_cast<void**>(&d_bounds_L), local_L.data(), local_bytes);
    copy_to_device(reinterpret_cast<void**>(&d_bounds_U), local_U.data(), local_bytes);
    copy_to_device(reinterpret_cast<void**>(&d_domain_width), input.domain_width.data(), domain_bytes);
    copy_to_device(reinterpret_cast<void**>(&d_active_indices), active_indices.data(), active_index_bytes);
    copy_to_device(reinterpret_cast<void**>(&d_inactive_indices), inactive_indices.data(), inactive_index_bytes);
    allocate_device(reinterpret_cast<void**>(&d_active_L), active_bytes);
    allocate_device(reinterpret_cast<void**>(&d_active_U), active_bytes);
    allocate_device(reinterpret_cast<void**>(&d_inactive_L), inactive_bytes);
    allocate_device(reinterpret_cast<void**>(&d_inactive_U), inactive_bytes);
    print_device_memory(rank, "after cudaMalloc/copies");

    int block = 128;
    if (!active_indices.empty()) {
        int active_grid = std::max(1, std::min(static_cast<int>((active_indices.size() + block - 1) / block), prop.maxGridSize[0]));
        split_active_boxes_kernel<<<active_grid, block>>>(
            d_bounds_L, d_bounds_U, d_domain_width, d_active_indices,
            static_cast<std::uint64_t>(active_indices.size()), input.d, stride,
            d_active_L, d_active_U);
        CUDA_CHECK(cudaGetLastError());
    }
    if (!inactive_indices.empty()) {
        int inactive_grid = std::max(1, std::min(static_cast<int>((inactive_indices.size() + block - 1) / block), prop.maxGridSize[0]));
        copy_inactive_boxes_kernel<<<inactive_grid, block>>>(
            d_bounds_L, d_bounds_U, d_inactive_indices,
            static_cast<std::uint64_t>(inactive_indices.size()), input.d,
            d_inactive_L, d_inactive_U);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after kernel");

    if (active_bytes != 0) {
        CUDA_CHECK(cudaMemcpy(output.active_L.data(), d_active_L, active_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output.active_U.data(), d_active_U, active_bytes, cudaMemcpyDeviceToHost));
    }
    if (inactive_bytes != 0) {
        CUDA_CHECK(cudaMemcpy(output.inactive_L.data(), d_inactive_L, inactive_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output.inactive_U.data(), d_inactive_U, inactive_bytes, cudaMemcpyDeviceToHost));
    }

    free_device(d_inactive_U);
    free_device(d_inactive_L);
    free_device(d_active_U);
    free_device(d_active_L);
    free_device(d_inactive_indices);
    free_device(d_active_indices);
    free_device(d_domain_width);
    free_device(d_bounds_U);
    free_device(d_bounds_L);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after cudaFree");
    return output;
}

void gather_rows(
    const std::vector<double>& local,
    std::uint64_t local_rows,
    std::uint64_t d,
    int rank,
    int size,
    std::vector<double>& all,
    std::uint64_t& total_rows) {
    unsigned long long local_rows_ull = static_cast<unsigned long long>(local_rows);
    std::vector<unsigned long long> rows_per_rank;
    if (rank == 0) {
        rows_per_rank.resize(size);
    }
    MPI_Gather(&local_rows_ull, 1, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? rows_per_rank.data() : nullptr, 1, MPI_UNSIGNED_LONG_LONG,
               0, MPI_COMM_WORLD);

    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        total_rows = 0;
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            std::uint64_t rows = static_cast<std::uint64_t>(rows_per_rank[r]);
            std::uint64_t elems = checked_mul(rows, d, "MPI gather elements");
            counts[r] = checked_int_count(elems, "MPI gather elements");
            displs[r] = offset;
            offset += counts[r];
            total_rows += rows;
        }
        all.resize(checked_mul(total_rows, d, "gathered output elements"));
    }

    int send_count = checked_int_count(checked_mul(local_rows, d, "local gather elements"), "local gather elements");
    MPI_Gatherv(local.empty() ? nullptr : local.data(), send_count, MPI_DOUBLE,
                rank == 0 && !all.empty() ? all.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        std::uint64_t stride = 2 * input.d + 1;

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA devices visible");
        }
        int device = rank % device_count;

        if (rank == 0) {
            std::uint64_t n_active = 0;
            for (unsigned char value : input.active_mask) {
                n_active += value != 0 ? 1 : 0;
            }
            std::uint64_t n_out = checked_mul(n_active, stride, "active output rows");
            if (input.keep_inactive != 0) {
                n_out += input.n - n_active;
            }
            std::cout << "exactbo_split_boxes: mpi_size=" << size
                      << " device_count=" << device_count
                      << " input=\"" << options.input_path << "\""
                      << " output=\"" << options.output_path << "\""
                      << " n=" << input.n << " d=" << input.d
                      << " active=" << n_active << " stride=" << stride
                      << " keep_inactive=" << input.keep_inactive
                      << " n_out=" << n_out << "\n";
        }

        LocalOutput local = compute_local(input, rank, size, device);

        std::vector<double> all_active_L;
        std::vector<double> all_active_U;
        std::vector<double> all_inactive_L;
        std::vector<double> all_inactive_U;
        std::uint64_t active_rows_total = 0;
        std::uint64_t active_rows_total_u = 0;
        std::uint64_t inactive_rows_total = 0;
        std::uint64_t inactive_rows_total_u = 0;

        gather_rows(local.active_L, local.active_rows, input.d, rank, size, all_active_L, active_rows_total);
        gather_rows(local.active_U, local.active_rows, input.d, rank, size, all_active_U, active_rows_total_u);
        gather_rows(local.inactive_L, local.inactive_rows, input.d, rank, size, all_inactive_L, inactive_rows_total);
        gather_rows(local.inactive_U, local.inactive_rows, input.d, rank, size, all_inactive_U, inactive_rows_total_u);

        if (rank == 0) {
            if (active_rows_total != active_rows_total_u || inactive_rows_total != inactive_rows_total_u) {
                throw std::runtime_error("gathered lower/upper row counts do not match");
            }
            std::uint64_t n_active = active_rows_total / stride;
            std::uint64_t n_out = active_rows_total + inactive_rows_total;
            std::vector<double> out_L(checked_mul(n_out, input.d, "final output elements"));
            std::vector<double> out_U(out_L.size());

            std::copy(all_active_L.begin(), all_active_L.end(), out_L.begin());
            std::copy(all_active_U.begin(), all_active_U.end(), out_U.begin());
            std::copy(all_inactive_L.begin(), all_inactive_L.end(), out_L.begin() + static_cast<std::ptrdiff_t>(all_active_L.size()));
            std::copy(all_inactive_U.begin(), all_inactive_U.end(), out_U.begin() + static_cast<std::ptrdiff_t>(all_active_U.size()));

            write_output(options.output_path, n_out, input.d, n_active, stride, out_L, out_U);
            std::cout << "rank 0 wrote split_boxes output to " << options.output_path << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " exactbo_split_boxes: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    cudaDeviceReset();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
