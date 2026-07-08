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

constexpr char kInputMagic[8] = {'T', 'E', 'I', 'B', 'I', 'N', '1', '!'};
constexpr char kOutputMagic[8] = {'T', 'E', 'I', 'B', 'O', 'U', '1', '!'};
constexpr double kSqrt2 = 1.41421356237309504880;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;

struct EiInput {
    std::uint64_t n = 0;
    double y_min = 0.0;
    double pad = 1.0e-12;
    std::vector<double> mu_lo;
    std::vector<double> mu_hi;
    std::vector<double> sig_lo;
    std::vector<double> sig_hi;
};

EiInput read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open input file: " + path);
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kInputMagic, sizeof(magic)) != 0) throw std::runtime_error("invalid ei_bounds input magic");
    EiInput input;
    read_value(in, input.n, "n");
    read_value(in, input.y_min, "y_min");
    read_value(in, input.pad, "pad");
    input.mu_lo.resize(input.n);
    input.mu_hi.resize(input.n);
    input.sig_lo.resize(input.n);
    input.sig_hi.resize(input.n);
    read_doubles(in, input.mu_lo, "mu_lo");
    read_doubles(in, input.mu_hi, "mu_hi");
    read_doubles(in, input.sig_lo, "sig_lo");
    read_doubles(in, input.sig_hi, "sig_hi");
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

__device__ void interval_product_bounds(double a_lo, double a_hi, double b_lo, double b_hi, double& out_lo, double& out_hi) {
    double p0 = a_lo * b_lo;
    double p1 = a_lo * b_hi;
    double p2 = a_hi * b_lo;
    double p3 = a_hi * b_hi;
    out_lo = fmin(fmin(p0, p1), fmin(p2, p3));
    out_hi = fmax(fmax(p0, p1), fmax(p2, p3));
}

__device__ double erf_approx(double x) {
    double p = 0.3275911;
    double a1 = 0.254829592;
    double a2 = -0.284496736;
    double a3 = 1.421413741;
    double a4 = -1.453152027;
    double a5 = 1.061405429;
    double sign = (x > 0.0) - (x < 0.0);
    double ax = fabs(x);
    double t = 1.0 / (1.0 + p * ax);
    double poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t);
    return sign * (1.0 - poly * exp(-(ax * ax)));
}

__device__ double norm_cdf(double z) {
    return 0.5 * (1.0 + erf_approx(z / kSqrt2));
}

__device__ double norm_pdf(double z) {
    return kInvSqrt2Pi * exp(-0.5 * z * z);
}

__global__ void ei_bounds_kernel(const double* mu_lo, const double* mu_hi, const double* sig_lo, const double* sig_hi, std::uint64_t local_n, double y_min, double pad, double* ei_lo, double* ei_hi) {
    for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        double N_lo = y_min - mu_hi[row];
        double N_hi = y_min - mu_lo[row];
        double inv_pad = 1.0 / pad;
        double J_lo = (sig_hi[row] != 0.0) ? (1.0 / sig_hi[row]) : inv_pad;
        double J_hi = (sig_lo[row] != 0.0) ? (1.0 / sig_lo[row]) : inv_pad;

        double Z_lo = 0.0;
        double Z_hi = 0.0;
        interval_product_bounds(N_lo, N_hi, J_lo, J_hi, Z_lo, Z_hi);

        double Phi_lo = norm_cdf(Z_lo);
        double Phi_hi = norm_cdf(Z_hi);
        double pdf_lo = norm_pdf(Z_lo);
        double pdf_hi = norm_pdf(Z_hi);
        double phi_lo = fmin(pdf_lo, pdf_hi);
        double phi_hi = fmax(pdf_lo, pdf_hi);
        if (Z_lo <= 0.0 && Z_hi >= 0.0) phi_hi = kInvSqrt2Pi;

        double U_lo = 0.0, U_hi = 0.0, V_lo = 0.0, V_hi = 0.0;
        interval_product_bounds(N_lo, N_hi, Phi_lo, Phi_hi, U_lo, U_hi);
        interval_product_bounds(sig_lo[row], sig_hi[row], phi_lo, phi_hi, V_lo, V_hi);

        double lo = fmax(U_lo + V_lo, 0.0);
        double hi = fmax(U_hi + V_hi, 0.0);
        if (sig_hi[row] == 0.0) hi = 0.0;
        if (sig_hi[row] == 0.0 || sig_lo[row] == 0.0) lo = 0.0;
        ei_lo[row] = lo;
        ei_hi[row] = hi;
    }
}

void compute_local(const EiInput& input, int rank, int size, int device, std::vector<double>& local_lo, std::vector<double>& local_hi) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::uint64_t start = local_start_for_rank(input.n, rank, size);
    std::uint64_t local_n = local_count_for_rank(input.n, rank, size);
    local_lo.resize(local_n);
    local_hi.resize(local_n);
    std::cout << "rank " << rank << " starting ei_bounds pid=" << getpid() << " launch_cpu=" << current_cpu()
              << " gpu_device=" << device << " gpu_name=\"" << prop.name << "\" rows=[" << start << "," << (start + local_n) << ")"
              << " n=" << input.n << "\n";
    if (local_n == 0) return;
    std::vector<double> local_mu_lo = slice_rows(input.mu_lo, start, local_n, 1);
    std::vector<double> local_mu_hi = slice_rows(input.mu_hi, start, local_n, 1);
    std::vector<double> local_sig_lo = slice_rows(input.sig_lo, start, local_n, 1);
    std::vector<double> local_sig_hi = slice_rows(input.sig_hi, start, local_n, 1);
    std::size_t bytes = local_n * sizeof(double);
    double *d_mu_lo=nullptr, *d_mu_hi=nullptr, *d_sig_lo=nullptr, *d_sig_hi=nullptr, *d_lo=nullptr, *d_hi=nullptr;
    print_device_memory(rank, "before cudaMalloc");
    CUDA_CHECK(cudaMalloc(&d_mu_lo, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu_hi, bytes));
    CUDA_CHECK(cudaMalloc(&d_sig_lo, bytes));
    CUDA_CHECK(cudaMalloc(&d_sig_hi, bytes));
    CUDA_CHECK(cudaMalloc(&d_lo, bytes));
    CUDA_CHECK(cudaMalloc(&d_hi, bytes));
    print_device_memory(rank, "after cudaMalloc");
    CUDA_CHECK(cudaMemcpy(d_mu_lo, local_mu_lo.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu_hi, local_mu_hi.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig_lo, local_sig_lo.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig_hi, local_sig_hi.data(), bytes, cudaMemcpyHostToDevice));
    print_device_memory(rank, "after host-to-device copies");
    int block = 256;
    int grid = std::max(1, std::min(static_cast<int>((local_n + block - 1) / block), prop.maxGridSize[0]));
    ei_bounds_kernel<<<grid, block>>>(d_mu_lo, d_mu_hi, d_sig_lo, d_sig_hi, local_n, input.y_min, input.pad, d_lo, d_hi);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_device_memory(rank, "after kernel");
    CUDA_CHECK(cudaMemcpy(local_lo.data(), d_lo, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_hi.data(), d_hi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hi)); CUDA_CHECK(cudaFree(d_lo)); CUDA_CHECK(cudaFree(d_sig_hi)); CUDA_CHECK(cudaFree(d_sig_lo)); CUDA_CHECK(cudaFree(d_mu_hi)); CUDA_CHECK(cudaFree(d_mu_lo));
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
        Options options = parse_args(argc, argv, "Usage: exactbo_ei_bounds --input input.bin --output output.bin");
        EiInput input = read_input(options.input_path);
        if (input.n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) throw std::runtime_error("n is too large for MPI_Gatherv int counts");
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) throw std::runtime_error("no CUDA devices visible");
        int device = rank % device_count;
        if (rank == 0) std::cout << "exactbo_ei_bounds: mpi_size=" << size << " device_count=" << device_count << " input=\"" << options.input_path << "\" output=\"" << options.output_path << "\" n=" << input.n << "\n";
        std::vector<double> local_lo, local_hi;
        compute_local(input, rank, size, device, local_lo, local_hi);
        int local_count = static_cast<int>(local_lo.size());
        std::vector<int> counts(size), displs(size);
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<double> all_lo, all_hi;
        if (rank == 0) { make_counts_displs(input.n, size, counts, displs); all_lo.resize(input.n); all_hi.resize(input.n); }
        MPI_Gatherv(local_lo.data(), local_count, MPI_DOUBLE, all_lo.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_hi.data(), local_count, MPI_DOUBLE, all_hi.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) { write_output(options.output_path, input.n, all_lo, all_hi); std::cout << "rank 0 wrote ei_bounds output to " << options.output_path << "\n"; }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " exactbo_ei_bounds: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
