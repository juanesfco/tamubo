#include <algorithm>
#include <cmath>
#include <cstdint>
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

constexpr char kInputMagic[8] = {'T', 'R', 'B', 'F', 'K', 'I', 'N', '1'};
constexpr char kOutputMagic[8] = {'T', 'R', 'B', 'F', 'K', 'O', 'U', '1'};

struct Options {
    std::string input_path;
    std::string output_path;
};

struct RbfInput {
    std::uint64_t n = 0;
    std::uint64_t d = 0;
    double sigma_f_2 = 1.0;
    std::vector<double> bounds_l;
    std::vector<double> bounds_u;
    std::vector<double> xi;
    std::vector<double> length_scale;
};

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
            std::cout << "Usage: exactbo_rbf_k_bounds --input input.bin --output output.bin\n";
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

RbfInput read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }

    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kInputMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid rbf_k_bounds input magic");
    }

    RbfInput input;
    read_value(in, input.n, "n");
    read_value(in, input.d, "d");
    read_value(in, input.sigma_f_2, "sigma_f_2");

    if (input.d == 0) {
        throw std::runtime_error("d must be > 0");
    }
    if (input.n > 0 && input.d > (std::numeric_limits<std::uint64_t>::max() / input.n)) {
        throw std::runtime_error("n*d overflows uint64");
    }

    std::uint64_t elements = input.n * input.d;
    input.bounds_l.resize(elements);
    input.bounds_u.resize(elements);
    input.xi.resize(input.d);
    input.length_scale.resize(input.d);

    read_doubles(in, input.bounds_l, "bounds_l");
    read_doubles(in, input.bounds_u, "bounds_u");
    read_doubles(in, input.xi, "xi");
    read_doubles(in, input.length_scale, "length_scale");

    char trailing = 0;
    in.read(&trailing, 1);
    if (in.gcount() != 0) {
        throw std::runtime_error("input file has trailing bytes");
    }

    return input;
}

void write_output(
    const std::string& path,
    std::uint64_t n,
    std::uint64_t d,
    const std::vector<double>& k_lo,
    const std::vector<double>& k_hi) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    out.write(kOutputMagic, sizeof(kOutputMagic));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(k_lo.data()),
              static_cast<std::streamsize>(k_lo.size() * sizeof(double)));
    out.write(reinterpret_cast<const char*>(k_hi.data()),
              static_cast<std::streamsize>(k_hi.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path);
    }
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

__global__ void rbf_k_bounds_kernel(
    const double* bounds_l,
    const double* bounds_u,
    const double* xi,
    const double* length_scale,
    double sigma_f_2,
    std::uint64_t local_n,
    std::uint64_t d,
    double* k_lo,
    double* k_hi) {
    for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        double sum_min_sq = 0.0;
        double sum_max_sq = 0.0;

        for (std::uint64_t col = 0; col < d; ++col) {
            std::uint64_t idx = row * d + col;
            double diff_lo = (bounds_l[idx] - xi[col]) / length_scale[col];
            double diff_hi = (xi[col] - bounds_u[idx]) / length_scale[col];
            double d_min = fmax(fmax(diff_lo, diff_hi), 0.0);
            double d_max = fmax(fabs(diff_lo), fabs(diff_hi));
            sum_min_sq += d_min * d_min;
            sum_max_sq += d_max * d_max;
        }

        k_lo[row] = sigma_f_2 * exp(-0.5 * sum_max_sq);
        k_hi[row] = sigma_f_2 * exp(-0.5 * sum_min_sq);
    }
}

std::vector<double> slice_rows(
    const std::vector<double>& values,
    std::uint64_t start,
    std::uint64_t count,
    std::uint64_t d) {
    auto begin = values.begin() + static_cast<std::ptrdiff_t>(start * d);
    auto end = begin + static_cast<std::ptrdiff_t>(count * d);
    return std::vector<double>(begin, end);
}

void compute_local(
    const RbfInput& input,
    int rank,
    int size,
    int device,
    std::vector<double>& local_lo,
    std::vector<double>& local_hi) {
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::uint64_t start = local_start_for_rank(input.n, rank, size);
    std::uint64_t local_n = local_count_for_rank(input.n, rank, size);
    local_lo.resize(local_n);
    local_hi.resize(local_n);

    std::cout << "rank " << rank
              << " starting rbf_k_bounds"
              << " pid=" << getpid()
              << " launch_cpu=" << current_cpu()
              << " gpu_device=" << device
              << " gpu_name=\"" << prop.name << "\""
              << " rows=[" << start << "," << (start + local_n) << ")"
              << " n=" << input.n
              << " d=" << input.d << "\n";

    if (local_n == 0) {
        return;
    }

    std::vector<double> local_l = slice_rows(input.bounds_l, start, local_n, input.d);
    std::vector<double> local_u = slice_rows(input.bounds_u, start, local_n, input.d);

    std::size_t box_bytes = local_l.size() * sizeof(double);
    std::size_t vector_bytes = input.d * sizeof(double);
    std::size_t out_bytes = local_n * sizeof(double);

    double* d_bounds_l = nullptr;
    double* d_bounds_u = nullptr;
    double* d_xi = nullptr;
    double* d_length_scale = nullptr;
    double* d_k_lo = nullptr;
    double* d_k_hi = nullptr;

    print_device_memory(rank, "before cudaMalloc");
    CUDA_CHECK(cudaMalloc(&d_bounds_l, box_bytes));
    CUDA_CHECK(cudaMalloc(&d_bounds_u, box_bytes));
    CUDA_CHECK(cudaMalloc(&d_xi, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_length_scale, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_k_lo, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_k_hi, out_bytes));
    print_device_memory(rank, "after cudaMalloc");

    CUDA_CHECK(cudaMemcpy(d_bounds_l, local_l.data(), box_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds_u, local_u.data(), box_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xi, input.xi.data(), vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length_scale, input.length_scale.data(), vector_bytes, cudaMemcpyHostToDevice));
    print_device_memory(rank, "after host-to-device copies");

    int block = 256;
    int grid = static_cast<int>((local_n + block - 1) / block);
    grid = std::max(1, std::min(grid, prop.maxGridSize[0]));
    rbf_k_bounds_kernel<<<grid, block>>>(
        d_bounds_l,
        d_bounds_u,
        d_xi,
        d_length_scale,
        input.sigma_f_2,
        local_n,
        input.d,
        d_k_lo,
        d_k_hi);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after kernel");

    CUDA_CHECK(cudaMemcpy(local_lo.data(), d_k_lo, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_hi.data(), d_k_hi, out_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_k_hi));
    CUDA_CHECK(cudaFree(d_k_lo));
    CUDA_CHECK(cudaFree(d_length_scale));
    CUDA_CHECK(cudaFree(d_xi));
    CUDA_CHECK(cudaFree(d_bounds_u));
    CUDA_CHECK(cudaFree(d_bounds_l));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after cudaFree");
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
        RbfInput input = read_input(options.input_path);

        if (input.n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("n is too large for MPI_Gatherv int counts");
        }

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA devices visible");
        }
        int device = rank % device_count;

        if (rank == 0) {
            std::cout << "exactbo_rbf_k_bounds: mpi_size=" << size
                      << " device_count=" << device_count
                      << " input=\"" << options.input_path << "\""
                      << " output=\"" << options.output_path << "\""
                      << " n=" << input.n
                      << " d=" << input.d
                      << " sigma_f_2=" << input.sigma_f_2 << "\n";
        }

        std::vector<double> local_lo;
        std::vector<double> local_hi;
        compute_local(input, rank, size, device, local_lo, local_hi);

        int local_count = static_cast<int>(local_lo.size());
        std::vector<int> counts(size);
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(size);
        std::vector<double> all_lo;
        std::vector<double> all_hi;
        if (rank == 0) {
            int offset = 0;
            for (int i = 0; i < size; ++i) {
                displs[i] = offset;
                offset += counts[i];
            }
            all_lo.resize(input.n);
            all_hi.resize(input.n);
        }

        MPI_Gatherv(local_lo.data(), local_count, MPI_DOUBLE,
                    all_lo.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_hi.data(), local_count, MPI_DOUBLE,
                    all_hi.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            write_output(options.output_path, input.n, input.d, all_lo, all_hi);
            std::cout << "rank 0 wrote rbf_k_bounds output to " << options.output_path << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " exactbo_rbf_k_bounds: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
