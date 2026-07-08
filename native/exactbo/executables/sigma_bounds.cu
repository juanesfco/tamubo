#include <algorithm>
#include <cmath>
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

Options parse_args(int argc, char** argv, const char* usage) {
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
            std::cout << usage << "\n";
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

void make_counts_displs(std::uint64_t n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        std::uint64_t count = local_count_for_rank(n, r, size);
        if (count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("local row count is too large for MPI_Gatherv int counts");
        }
        counts[r] = static_cast<int>(count);
        displs[r] = offset;
        offset += counts[r];
    }
}

std::vector<double> slice_rows(const std::vector<double>& values, std::uint64_t start, std::uint64_t count, std::uint64_t stride) {
    auto begin = values.begin() + static_cast<std::ptrdiff_t>(start * stride);
    auto end = begin + static_cast<std::ptrdiff_t>(count * stride);
    return std::vector<double>(begin, end);
}

constexpr char kInputMagic[8] = {'T', 'S', 'I', 'G', 'I', 'N', '1', '!'};
constexpr char kOutputMagic[8] = {'T', 'S', 'I', 'G', 'O', 'U', '1', '!'};

struct SigmaInput {
    std::uint64_t n = 0;
    std::uint64_t N = 0;
    double sigma_f_2 = 1.0;
    double y_train_std = 1.0;
    std::uint64_t scaled_output = 1;
    std::vector<double> L;
    std::vector<double> K_lo;
    std::vector<double> K_hi;
};

SigmaInput read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open input file: " + path);
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kInputMagic, sizeof(magic)) != 0) throw std::runtime_error("invalid sigma_bounds input magic");
    SigmaInput input;
    read_value(in, input.n, "n");
    read_value(in, input.N, "N");
    read_value(in, input.sigma_f_2, "sigma_f_2");
    read_value(in, input.y_train_std, "y_train_std");
    read_value(in, input.scaled_output, "scaled_output");
    if (input.N == 0) throw std::runtime_error("N must be > 0");
    if (input.n > 0 && input.N > (std::numeric_limits<std::uint64_t>::max() / input.n)) throw std::runtime_error("n*N overflows uint64");
    input.L.resize(input.N * input.N);
    input.K_lo.resize(input.n * input.N);
    input.K_hi.resize(input.n * input.N);
    read_doubles(in, input.L, "L");
    read_doubles(in, input.K_lo, "K_lo");
    read_doubles(in, input.K_hi, "K_hi");
    check_no_trailing_bytes(in);
    return input;
}

void write_output(const std::string& path, std::uint64_t n, const std::vector<double>& lo, const std::vector<double>& hi) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open output file: " + path);
    out.write(kOutputMagic, sizeof(kOutputMagic));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(lo.data()), static_cast<std::streamsize>(lo.size() * sizeof(double)));
    out.write(reinterpret_cast<const char*>(hi.data()), static_cast<std::streamsize>(hi.size() * sizeof(double)));
    if (!out) throw std::runtime_error("failed to write output file: " + path);
}

__global__ void sigma_bounds_kernel(const double* K_lo, const double* K_hi, const double* L, std::uint64_t local_n, std::uint64_t N, double sigma_f_2, double y_train_std, std::uint64_t scaled, double* v_lo, double* v_hi, double* sig_lo, double* sig_hi) {
    for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        double q_hi = 0.0;
        double q_lo = 0.0;
        for (std::uint64_t j = 0; j < N; ++j) {
            double sum_lo = 0.0;
            double sum_hi = 0.0;
            for (std::uint64_t i = 0; i < j; ++i) {
                double Lji = L[j * N + i];
                double prev_lo = v_lo[row * N + i];
                double prev_hi = v_hi[row * N + i];
                if (Lji >= 0.0) {
                    sum_lo += Lji * prev_lo;
                    sum_hi += Lji * prev_hi;
                } else {
                    sum_lo += Lji * prev_hi;
                    sum_hi += Lji * prev_lo;
                }
            }
            double diag = L[j * N + j];
            double cur_lo = (K_lo[row * N + j] - sum_hi) / diag;
            double cur_hi = (K_hi[row * N + j] - sum_lo) / diag;
            v_lo[row * N + j] = cur_lo;
            v_hi[row * N + j] = cur_hi;
            double sq_lo = cur_lo * cur_lo;
            double sq_hi = cur_hi * cur_hi;
            q_hi += fmax(sq_lo, sq_hi);
            double low_sq = fmin(sq_lo, sq_hi);
            if (cur_lo < 0.0 && cur_hi > 0.0) low_sq = 0.0;
            q_lo += low_sq;
        }
        double var_lo = fmax(sigma_f_2 - q_hi, 1.0e-12);
        double var_hi = fmax(sigma_f_2 - q_lo, 1.0e-12);
        double lo = sqrt(var_lo);
        double hi = sqrt(var_hi);
        if (!scaled) {
            lo *= y_train_std;
            hi *= y_train_std;
        }
        sig_lo[row] = lo;
        sig_hi[row] = hi;
    }
}

void compute_local(const SigmaInput& input, int rank, int size, int device, std::vector<double>& local_lo, std::vector<double>& local_hi) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::uint64_t start = local_start_for_rank(input.n, rank, size);
    std::uint64_t local_n = local_count_for_rank(input.n, rank, size);
    local_lo.resize(local_n);
    local_hi.resize(local_n);
    std::cout << "rank " << rank << " starting sigma_bounds pid=" << getpid() << " launch_cpu=" << current_cpu()
              << " gpu_device=" << device << " gpu_name=\"" << prop.name << "\" rows=[" << start << "," << (start + local_n) << ")"
              << " n=" << input.n << " N=" << input.N << "\n";
    if (local_n == 0) return;
    std::vector<double> local_K_lo = slice_rows(input.K_lo, start, local_n, input.N);
    std::vector<double> local_K_hi = slice_rows(input.K_hi, start, local_n, input.N);
    std::size_t L_bytes = input.L.size() * sizeof(double);
    std::size_t K_bytes = local_K_lo.size() * sizeof(double);
    std::size_t V_bytes = local_n * input.N * sizeof(double);
    std::size_t out_bytes = local_n * sizeof(double);
    double *d_L=nullptr, *d_K_lo=nullptr, *d_K_hi=nullptr, *d_v_lo=nullptr, *d_v_hi=nullptr, *d_lo=nullptr, *d_hi=nullptr;
    print_device_memory(rank, "before cudaMalloc");
    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    CUDA_CHECK(cudaMalloc(&d_K_lo, K_bytes));
    CUDA_CHECK(cudaMalloc(&d_K_hi, K_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_lo, V_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_hi, V_bytes));
    CUDA_CHECK(cudaMalloc(&d_lo, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_hi, out_bytes));
    print_device_memory(rank, "after cudaMalloc");
    CUDA_CHECK(cudaMemcpy(d_L, input.L.data(), L_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_lo, local_K_lo.data(), K_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_hi, local_K_hi.data(), K_bytes, cudaMemcpyHostToDevice));
    print_device_memory(rank, "after host-to-device copies");
    int block = 256;
    int grid = std::max(1, std::min(static_cast<int>((local_n + block - 1) / block), prop.maxGridSize[0]));
    sigma_bounds_kernel<<<grid, block>>>(d_K_lo, d_K_hi, d_L, local_n, input.N, input.sigma_f_2, input.y_train_std, input.scaled_output, d_v_lo, d_v_hi, d_lo, d_hi);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after kernel");
    CUDA_CHECK(cudaMemcpy(local_lo.data(), d_lo, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_hi.data(), d_hi, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hi)); CUDA_CHECK(cudaFree(d_lo)); CUDA_CHECK(cudaFree(d_v_hi)); CUDA_CHECK(cudaFree(d_v_lo)); CUDA_CHECK(cudaFree(d_K_hi)); CUDA_CHECK(cudaFree(d_K_lo)); CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after cudaFree");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        Options options = parse_args(argc, argv, "Usage: exactbo_sigma_bounds --input input.bin --output output.bin");
        SigmaInput input = read_input(options.input_path);
        if (input.n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) throw std::runtime_error("n is too large for MPI_Gatherv int counts");
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) throw std::runtime_error("no CUDA devices visible");
        int device = rank % device_count;
        if (rank == 0) std::cout << "exactbo_sigma_bounds: mpi_size=" << size << " device_count=" << device_count << " input=\"" << options.input_path << "\" output=\"" << options.output_path << "\" n=" << input.n << " N=" << input.N << "\n";
        std::vector<double> local_lo, local_hi;
        compute_local(input, rank, size, device, local_lo, local_hi);
        int local_count = static_cast<int>(local_lo.size());
        std::vector<int> counts(size), displs(size);
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<double> all_lo, all_hi;
        if (rank == 0) { make_counts_displs(input.n, size, counts, displs); all_lo.resize(input.n); all_hi.resize(input.n); }
        MPI_Gatherv(local_lo.data(), local_count, MPI_DOUBLE, all_lo.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_hi.data(), local_count, MPI_DOUBLE, all_hi.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) { write_output(options.output_path, input.n, all_lo, all_hi); std::cout << "rank 0 wrote sigma_bounds output to " << options.output_path << "\n"; }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " exactbo_sigma_bounds: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
