#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tamubo/exactbo/box_store.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif

// Frequently used standard-library names are imported once here.  This keeps
// the algorithm below readable without hiding the origin of less common names.
using std::cerr;
using std::copy_n;
using std::cout;
using std::error_code;
using std::exception;
using std::exit;
using std::fill;
using std::fprintf;
using std::fixed;
using std::gcd;
using std::ifstream;
using std::ios;
using std::isfinite;
using std::make_unique;
using std::memcmp;
using std::max;
using std::min;
using std::move;
using std::numeric_limits;
using std::ofstream;
using std::ostringstream;
using std::runtime_error;
using std::setprecision;
using std::size_t;
using std::string;
using std::strcmp;
using std::strerror;
using std::stoi;
using std::stoull;
using std::to_string;
using std::uint64_t;
using std::unique_ptr;
using std::vector;

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(status));                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
        }                                                                       \
    } while (0)

namespace {

// -----------------------------------------------------------------------------
// File format constants and small data containers
// -----------------------------------------------------------------------------

constexpr char kInputMagic[8] = {'T', 'P', 'A', 'R', 'I', 'N', '1', '!'};
constexpr char kOutputMagic[8] = {'T', 'P', 'A', 'R', 'O', 'U', '1', '!'};
constexpr double kInvSqrt2Pi = 0.39894228040143267794;
constexpr double kSqrt2 = 1.41421356237309504880;

struct Options {
    string input_path;
    string output_path;
    int verbose = 0;
    int device_batch_rows = 4096;
    // Zero asks the memory planner to choose a safe split batch automatically.
    int split_batch_parents = 0;
    string box_storage = "auto";
    // Zero derives a conservative limit from MemAvailable/cgroup state.
    uint64_t host_box_limit_bytes = 0;
    string spill_dir;
    bool keep_spill_files = false;
};

using tamubo::exactbo::BoxStore;
using tamubo::exactbo::FileBoxStore;
using tamubo::exactbo::FileBoxWriter;
using tamubo::exactbo::HostBoxStore;

// Training size, dimension, and partition limit are small ordinary ints in the
// algorithm.  read_input() converts them from the Python file format.
struct PartitionInput {
    int n_train = 0;
    int d = 0;
    int max_partitions = 0;
    double epsilon_ei = 0.0;
    double sigma_f_2 = 1.0;
    double sigma_n_2 = 0.0;
    double y_train_mean = 0.0;
    double y_train_std = 1.0;
    double y_min_scaled = 0.0;
    vector<double> epsilon_x;
    vector<double> domain_L;
    vector<double> domain_U;
    vector<double> X_train;
    vector<double> alpha;
    vector<double> L;
    vector<double> length_scale;
};

struct BestSample {
    double ei = -numeric_limits<double>::infinity();
    long long box_idx = -1;
    vector<double> x;
};

// -----------------------------------------------------------------------------
// MPI rank and device-information helpers
// -----------------------------------------------------------------------------

int current_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}

int local_rank_on_node() {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    return local_rank;
}

int local_size_on_node() {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    int local_size = 1;
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);
    return local_size;
}

string format_bytes(size_t bytes) {
    constexpr double gib_divisor = 1024.0 * 1024.0 * 1024.0;
    constexpr double gb_divisor  = 1000.0 * 1000.0 * 1000.0;
    ostringstream out;
    out << fixed << setprecision(3)
        << bytes / gib_divisor << " GiB"
        << " (" << bytes / gb_divisor << " GB)";
    return out.str();
}

void print_device_memory(int rank, const string& label) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t used_bytes = total_bytes - free_bytes;
    cout << "rank " << rank << " memory [" << label << "] "
              << "used=" << format_bytes(used_bytes)
              << " free=" << format_bytes(free_bytes)
              << " total=" << format_bytes(total_bytes) << "\n";
}

// -----------------------------------------------------------------------------
// Command-line and binary file input/output
// -----------------------------------------------------------------------------

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* flag) -> char* {
            if (i + 1 >= argc) {
                throw runtime_error(string("missing value for ") + flag);
            }
            return argv[++i];
        };
        if (strcmp(argv[i], "--input") == 0) {
            options.input_path = need_value("--input");
        } else if (strcmp(argv[i], "--output") == 0) {
            options.output_path = need_value("--output");
        } else if (strcmp(argv[i], "--verbose") == 0) {
            options.verbose = 1;
        } else if (strcmp(argv[i], "--device-batch-rows") == 0) {
            const char* value = need_value("--device-batch-rows");
            size_t parsed = 0;
            options.device_batch_rows = stoi(value, &parsed);
            if (value[0] == '-' || value[parsed] != '\0' ||
                options.device_batch_rows == 0) {
                throw runtime_error("--device-batch-rows must be a positive integer");
            }
        } else if (strcmp(argv[i], "--split-batch-parents") == 0) {
            const char* value = need_value("--split-batch-parents");
            size_t parsed = 0;
            options.split_batch_parents = stoi(value, &parsed);
            if (value[0] == '-' || value[parsed] != '\0') {
                throw runtime_error("--split-batch-parents must be a nonnegative integer");
            }
        } else if (strcmp(argv[i], "--box-storage") == 0) {
            options.box_storage = need_value("--box-storage");
            if (options.box_storage != "auto" && options.box_storage != "host" &&
                options.box_storage != "file") {
                throw runtime_error("--box-storage must be auto, host, or file");
            }
        } else if (strcmp(argv[i], "--host-box-limit-bytes") == 0) {
            const char* value = need_value("--host-box-limit-bytes");
            size_t parsed = 0;
            options.host_box_limit_bytes = stoull(value, &parsed);
            if (value[0] == '-' || value[parsed] != '\0') {
                throw runtime_error("--host-box-limit-bytes must be a nonnegative integer");
            }
        } else if (strcmp(argv[i], "--spill-dir") == 0) {
            options.spill_dir = need_value("--spill-dir");
        } else if (strcmp(argv[i], "--keep-spill-files") == 0) {
            options.keep_spill_files = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            cout << "Usage: exactbo_partitioning --input input.bin --output output.bin\n"
                      << "       [--device-batch-rows N] [--split-batch-parents N]\n"
                      << "       [--box-storage auto|host|file] [--host-box-limit-bytes N]\n"
                      << "       [--spill-dir PATH] [--keep-spill-files] [--verbose]\n";
            exit(EXIT_SUCCESS);
        } else {
            throw runtime_error(string("unknown argument: ") + argv[i]);
        }
    }
    if (options.input_path.empty() || options.output_path.empty()) {
        throw runtime_error("--input and --output are required");
    }
    return options;
}


uint64_t local_start_for_rank(uint64_t n, int rank, int size) {
    uint64_t base = n / size;
    uint64_t rem = n % size;
    return rank * base + min<uint64_t>(rank, rem);
}

uint64_t local_count_for_rank(uint64_t n, int rank, int size) {
    uint64_t base = n / size;
    uint64_t rem = n % size;
    return base + (rank < rem ? 1 : 0);
}

PartitionInput read_input(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) {
        throw runtime_error("failed to open input file: " + path);
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || memcmp(magic, kInputMagic, sizeof(magic)) != 0) {
        throw runtime_error("invalid exactbo_partitioning input magic");
    }

    PartitionInput input;
    uint64_t n_train_on_disk = 0;
    uint64_t dimensions_on_disk = 0;
    uint64_t max_partitions_on_disk = 0;
    // Binary streams read bytes through char pointers.  These casts are required
    // by the C++ stream API; they do not change the stored values.
    in.read(reinterpret_cast<char*>(&n_train_on_disk), sizeof(n_train_on_disk));
    in.read(reinterpret_cast<char*>(&dimensions_on_disk), sizeof(dimensions_on_disk));
    in.read(reinterpret_cast<char*>(&max_partitions_on_disk), sizeof(max_partitions_on_disk));
    in.read(reinterpret_cast<char*>(&input.epsilon_ei), sizeof(input.epsilon_ei));
    in.read(reinterpret_cast<char*>(&input.sigma_f_2), sizeof(input.sigma_f_2));
    in.read(reinterpret_cast<char*>(&input.sigma_n_2), sizeof(input.sigma_n_2));
    in.read(reinterpret_cast<char*>(&input.y_train_mean), sizeof(input.y_train_mean));
    in.read(reinterpret_cast<char*>(&input.y_train_std), sizeof(input.y_train_std));
    in.read(reinterpret_cast<char*>(&input.y_min_scaled), sizeof(input.y_min_scaled));
    if (!in) {
        throw runtime_error("input file ended while reading GP settings");
    }
    if (n_train_on_disk > numeric_limits<int>::max() ||
        dimensions_on_disk > numeric_limits<int>::max() ||
        max_partitions_on_disk > numeric_limits<int>::max()) {
        throw runtime_error("training size, dimensions, or partitions exceed int range");
    }
    // These casts deliberately narrow validated on-disk 64-bit values.
    input.n_train = static_cast<int>(n_train_on_disk);
    input.d = static_cast<int>(dimensions_on_disk);
    input.max_partitions = static_cast<int>(max_partitions_on_disk);

    if (input.n_train == 0 || input.d == 0 || input.max_partitions == 0) {
        throw runtime_error("n_train, d, and max_partitions must be > 0");
    }
    if (input.d >= 63) {
        throw runtime_error("d is too large for m=2^d samples");
    }

    input.epsilon_x.resize(input.d);
    input.domain_L.resize(input.d);
    input.domain_U.resize(input.d);
    input.X_train.resize(input.n_train * input.d);
    input.alpha.resize(input.n_train);
    input.L.resize(input.n_train * input.n_train);
    input.length_scale.resize(input.d);

    in.read(reinterpret_cast<char*>(input.epsilon_x.data()), input.epsilon_x.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(input.domain_L.data()), input.domain_L.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(input.domain_U.data()), input.domain_U.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(input.X_train.data()), input.X_train.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(input.alpha.data()), input.alpha.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(input.L.data()), input.L.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(input.length_scale.data()), input.length_scale.size() * sizeof(double));
    if (!in) {
        throw runtime_error("input file ended while reading GP arrays");
    }

    for (int dim = 0; dim < input.d; ++dim) {
        if (!isfinite(input.domain_L[dim]) || !isfinite(input.domain_U[dim]) ||
            !(input.domain_L[dim] < input.domain_U[dim])) {
            throw runtime_error("each domain lower bound must be finite and smaller than its upper bound");
        }
        if (!isfinite(input.epsilon_x[dim]) || input.epsilon_x[dim] < 0.0) {
            throw runtime_error("epsilon_x values must be finite and nonnegative");
        }
        if (!isfinite(input.length_scale[dim]) || !(input.length_scale[dim] > 0.0)) {
            throw runtime_error("length scales must be finite and positive");
        }
    }
    if (!isfinite(input.epsilon_ei) || input.epsilon_ei < 0.0 ||
        !isfinite(input.sigma_f_2) || !(input.sigma_f_2 > 0.0) ||
        !isfinite(input.y_min_scaled)) {
        throw runtime_error("invalid EI tolerance or GP scalar parameters");
    }
    for (int j = 0; j < input.n_train; ++j) {
        double diagonal = input.L[j * input.n_train + j];
        if (!isfinite(diagonal) || !(diagonal > 0.0)) {
            throw runtime_error("the Cholesky factor must have a finite positive diagonal");
        }
    }
    return input;
}

void write_output(
    const string& path,
    const BestSample& best,
    int d,
    uint64_t partitions_done,
    uint64_t n_boxes_final,
    int converged) {
    ofstream out(path, ios::binary);
    if (!out) {
        throw runtime_error("failed to open output file: " + path);
    }
    out.write(kOutputMagic, sizeof(kOutputMagic));
    uint64_t dimensions_on_disk = d;
    out.write(reinterpret_cast<const char*>(&dimensions_on_disk),
              sizeof(dimensions_on_disk));
    out.write(reinterpret_cast<const char*>(&partitions_done), sizeof(partitions_done));
    out.write(reinterpret_cast<const char*>(&n_boxes_final), sizeof(n_boxes_final));
    out.write(reinterpret_cast<const char*>(&converged), sizeof(converged));
    out.write(reinterpret_cast<const char*>(&best.ei), sizeof(best.ei));
    out.write(reinterpret_cast<const char*>(best.x.data()), d * sizeof(double));
    if (!out) {
        throw runtime_error("failed to write output file: " + path);
    }
}

// -----------------------------------------------------------------------------
// Deterministic sample points and CUDA memory management
// -----------------------------------------------------------------------------

vector<double> centered_latin_hypercube_unit(uint64_t n_points, int d) {
    vector<double> lhs(n_points * d);
    vector<double> centers(n_points);
    for (uint64_t i = 0; i < n_points; ++i) {
        centers[i] = (i + 0.5) / n_points;
    }
    for (int j = 0; j < d; ++j) {
        uint64_t step = 2 * j + 1;
        while (gcd(step, n_points) != 1) {
            step += 2;
        }
        for (uint64_t i = 0; i < n_points; ++i) {
            lhs[i * d + j] = centers[(i * step + j) % n_points];
        }
    }
    return lhs;
}

void check_cuda_alloc(cudaError_t status, const char* label, size_t bytes, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    fprintf(stderr,
                 "CUDA allocation error at %s:%d for %s: requested=%zu bytes free=%zu bytes total=%zu bytes error=%s\n",
                 file, line, label, bytes, free_bytes, total_bytes, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

template <class T>
void allocate_device(T** pointer, size_t bytes, const char* label) {
    if (bytes == 0) {
        *pointer = nullptr;
        return;
    }

    // cudaMalloc is a C API and therefore requires void**.  Keeping the cast
    // here lets every caller use its natural type, such as double**.
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(pointer), bytes);
    check_cuda_alloc(status, label, bytes, __FILE__, __LINE__);
}

template <class T>
void copy_to_device(T** device_pointer, const T* host_pointer,
                    size_t bytes, const char* label) {
    allocate_device(device_pointer, bytes, label);
    if (bytes != 0) {
        CUDA_CHECK(cudaMemcpy(*device_pointer, host_pointer, bytes,
                              cudaMemcpyHostToDevice));
    }
}

void free_device(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// -----------------------------------------------------------------------------
// GPU mathematics: normal distribution, interval EI, GP evaluation, splitting
// -----------------------------------------------------------------------------

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

__device__ void interval_product_bounds(double a_lo, double a_hi, double b_lo, double b_hi, double& out_lo, double& out_hi) {
    double p0 = a_lo * b_lo;
    double p1 = a_lo * b_hi;
    double p2 = a_hi * b_lo;
    double p3 = a_hi * b_hi;
    out_lo = fmin(fmin(p0, p1), fmin(p2, p3));
    out_hi = fmax(fmax(p0, p1), fmax(p2, p3));
}

__device__ double ei_value(double mu, double sigma, double y_min) {
    sigma = fmax(sigma, 1.0e-12);
    double improvement = y_min - mu;
    double z = improvement / sigma;
    return improvement * norm_cdf(z) + sigma * norm_pdf(z);
}

__device__ double ei_upper_from_intervals(double mu_lo, double mu_hi, double sig_lo, double sig_hi, double y_min) {
    double N_lo = y_min - mu_hi;
    double N_hi = y_min - mu_lo;
    double inv_pad = 1.0e12;
    double J_lo = (sig_hi != 0.0) ? (1.0 / sig_hi) : inv_pad;
    double J_hi = (sig_lo != 0.0) ? (1.0 / sig_lo) : inv_pad;

    double Z_lo = 0.0;
    double Z_hi = 0.0;
    interval_product_bounds(N_lo, N_hi, J_lo, J_hi, Z_lo, Z_hi);

    double Phi_lo = norm_cdf(Z_lo);
    double Phi_hi = norm_cdf(Z_hi);
    double pdf_lo = norm_pdf(Z_lo);
    double pdf_hi = norm_pdf(Z_hi);
    double phi_lo = fmin(pdf_lo, pdf_hi);
    double phi_hi = fmax(pdf_lo, pdf_hi);
    if (Z_lo <= 0.0 && Z_hi >= 0.0) {
        phi_hi = kInvSqrt2Pi;
    }

    double U_lo = 0.0;
    double U_hi = 0.0;
    double V_lo = 0.0;
    double V_hi = 0.0;
    interval_product_bounds(N_lo, N_hi, Phi_lo, Phi_hi, U_lo, U_hi);
    interval_product_bounds(sig_lo, sig_hi, phi_lo, phi_hi, V_lo, V_hi);

    double hi = fmax(U_hi + V_hi, 0.0);
    if (sig_hi == 0.0) {
        hi = 0.0;
    }
    return hi;
}

__global__ void ei_hi_bounds_kernel(
    const double* boxes_L,
    const double* boxes_U,
    const double* X_train,
    const double* alpha,
    const double* L,
    const double* length_scale,
    uint64_t local_n,
    int n_train,
    int d,
    double sigma_f_2,
    double y_min_scaled,
    double* K_lo,
    double* K_hi,
    double* v_lo,
    double* v_hi,
    double* ei_hi) {
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        double mu_lo = 0.0;
        double mu_hi = 0.0;
        for (int j = 0; j < n_train; ++j) {
            double dmin_sq = 0.0;
            double dmax_sq = 0.0;
            for (int dim = 0; dim < d; ++dim) {
                double xi = X_train[j * d + dim];
                double ell = length_scale[dim];
                double diff_lo = (boxes_L[row * d + dim] - xi) / ell;
                double diff_hi = (xi - boxes_U[row * d + dim]) / ell;
                double dmin = fmax(fmax(diff_lo, diff_hi), 0.0);
                double dmax = fmax(fabs(diff_lo), fabs(diff_hi));
                dmin_sq += dmin * dmin;
                dmax_sq += dmax * dmax;
            }
            double klo = sigma_f_2 * exp(-0.5 * dmax_sq);
            double khi = sigma_f_2 * exp(-0.5 * dmin_sq);
            K_lo[row * n_train + j] = klo;
            K_hi[row * n_train + j] = khi;
            double a = alpha[j];
            double a_pos = fmax(a, 0.0);
            double a_neg = fmin(a, 0.0);
            mu_lo += klo * a_pos + khi * a_neg;
            mu_hi += khi * a_pos + klo * a_neg;
        }

        double q_hi = 0.0;
        double q_lo = 0.0;
        for (int j = 0; j < n_train; ++j) {
            double sum_lo = 0.0;
            double sum_hi = 0.0;
            for (int i = 0; i < j; ++i) {
                double Lji = L[j * n_train + i];
                double prev_lo = v_lo[row * n_train + i];
                double prev_hi = v_hi[row * n_train + i];
                if (Lji >= 0.0) {
                    sum_lo += Lji * prev_lo;
                    sum_hi += Lji * prev_hi;
                } else {
                    sum_lo += Lji * prev_hi;
                    sum_hi += Lji * prev_lo;
                }
            }
            double diag = L[j * n_train + j];
            double cur_lo = (K_lo[row * n_train + j] - sum_hi) / diag;
            double cur_hi = (K_hi[row * n_train + j] - sum_lo) / diag;
            v_lo[row * n_train + j] = cur_lo;
            v_hi[row * n_train + j] = cur_hi;
            double sq_lo = cur_lo * cur_lo;
            double sq_hi = cur_hi * cur_hi;
            q_hi += fmax(sq_lo, sq_hi);
            double low_sq = fmin(sq_lo, sq_hi);
            if (cur_lo < 0.0 && cur_hi > 0.0) {
                low_sq = 0.0;
            }
            q_lo += low_sq;
        }
        double sig_lo = sqrt(fmax(sigma_f_2 - q_hi, 1.0e-12));
        double sig_hi = sqrt(fmax(sigma_f_2 - q_lo, 1.0e-12));
        ei_hi[row] = ei_upper_from_intervals(mu_lo, mu_hi, sig_lo, sig_hi, y_min_scaled);
    }
}

__global__ void sample_best_kernel(
    const double* boxes_L,
    const double* boxes_U,
    const unsigned char* mask,
    const double* X_train,
    const double* alpha,
    const double* L,
    const double* length_scale,
    const double* lhs,
    uint64_t local_n,
    int n_train,
    int d,
    uint64_t n_samples,
    double sigma_f_2,
    double y_min_scaled,
    double* v,
    double* best_ei,
    double* best_points) {
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        if (mask[row] == 0) {
            best_ei[row] = -INFINITY;
            for (int dim = 0; dim < d; ++dim) {
                best_points[row * d + dim] = 0.0;
            }
            continue;
        }

        double row_best_ei = -INFINITY;
        uint64_t row_best_sample = 0;
        for (uint64_t sample = 0; sample < n_samples; ++sample) {
            double mu = 0.0;
            double q = 0.0;
            for (int j = 0; j < n_train; ++j) {
                double sqdist = 0.0;
                for (int dim = 0; dim < d; ++dim) {
                    double lo = boxes_L[row * d + dim];
                    double hi = boxes_U[row * d + dim];
                    double x = lo + (hi - lo) * lhs[sample * d + dim];
                    double diff = (x - X_train[j * d + dim]) / length_scale[dim];
                    sqdist += diff * diff;
                }
                double k = sigma_f_2 * exp(-0.5 * sqdist);
                mu += k * alpha[j];
                double sum = 0.0;
                for (int i = 0; i < j; ++i) {
                    sum += L[j * n_train + i] * v[row * n_train + i];
                }
                double vj = (k - sum) / L[j * n_train + j];
                v[row * n_train + j] = vj;
                q += vj * vj;
            }
            double sigma = sqrt(fmax(sigma_f_2 - q, 1.0e-12));
            double ei = ei_value(mu, sigma, y_min_scaled);
            if (ei > row_best_ei) {
                row_best_ei = ei;
                row_best_sample = sample;
            }
        }

        best_ei[row] = row_best_ei;
        for (int dim = 0; dim < d; ++dim) {
            double lo = boxes_L[row * d + dim];
            double hi = boxes_U[row * d + dim];
            best_points[row * d + dim] = lo + (hi - lo) * lhs[row_best_sample * d + dim];
        }
    }
}

__device__ double normalized_width(const double* boxes_L, const double* boxes_U, const double* domain_width, int row, int d, int dim) {
    double width = boxes_U[row * d + dim] - boxes_L[row * d + dim];
    double scale = domain_width[dim];
    return scale != 0.0 ? width / scale : width;
}

__device__ int split_dim_for_rank(const double* boxes_L, const double* boxes_U, const double* domain_width, int row, int d, int rank) {
    for (int dim = 0; dim < d; ++dim) {
        double score = normalized_width(boxes_L, boxes_U, domain_width, row, d, dim);
        int order = 0;
        for (int other = 0; other < d; ++other) {
            double other_score = normalized_width(boxes_L, boxes_U, domain_width, row, d, other);
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

__global__ void split_dense_boxes_kernel(
    const double* boxes_L,
    const double* boxes_U,
    const double* domain_width,
    int parent_count,
    int d,
    int stride,
    double* out_L,
    double* out_U) {
    for (int pos = blockIdx.x * blockDim.x + threadIdx.x;
         pos < parent_count;
         pos += blockDim.x * gridDim.x) {
        const int src = pos;
        size_t out_base = pos;
        out_base *= stride;
        for (int child = 0; child < stride; ++child) {
            for (int dim = 0; dim < d; ++dim) {
                out_L[(out_base + child) * d + dim] = boxes_L[src * d + dim];
                out_U[(out_base + child) * d + dim] = boxes_U[src * d + dim];
            }
        }
        for (int order = 0; order < d; ++order) {
            const int dim =
                split_dim_for_rank(boxes_L, boxes_U, domain_width, src, d, order);
            const double low = boxes_L[src * d + dim];
            const double high = boxes_U[src * d + dim];
            const double third = (high - low) / 3.0;
            const double lower_third = low + third;
            const double upper_third = high - third;
            const int lower_row = 2 * order;
            const int upper_row = lower_row + 1;
            out_U[(out_base + lower_row) * d + dim] = lower_third;
            out_L[(out_base + upper_row) * d + dim] = upper_third;
            for (int child = upper_row + 1; child < stride; ++child) {
                out_L[(out_base + child) * d + dim] = lower_third;
                out_U[(out_base + child) * d + dim] = upper_third;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// MPI communication helpers
// -----------------------------------------------------------------------------

void gather_double_rows(
    const vector<double>& local,
    uint64_t local_rows,
    int cols,
    int rank,
    int size,
    vector<double>& all,
    uint64_t& total_rows) {
    unsigned long long local_rows_ull = local_rows;
    vector<unsigned long long> rows_per_rank;
    if (rank == 0) {
        rows_per_rank.resize(size);
    }
    MPI_Gather(&local_rows_ull, 1, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? rows_per_rank.data() : nullptr, 1,
               MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // This function is used only for host-backed generations. The storage
    // planner already guarantees that their total number of doubles fits in
    // the 32-bit count and displacement arguments required by MPI_Gatherv.
    vector<int> counts;
    vector<int> displacements;
    if (rank == 0) {
        counts.resize(size);
        displacements.resize(size);
        total_rows = 0;
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            uint64_t rows = rows_per_rank[r];
            uint64_t elements = rows * cols;
            counts[r] = static_cast<int>(elements);
            displacements[r] = offset;
            offset += counts[r];
            total_rows += rows;
        }
        all.resize(total_rows * cols);
    }

    int send_count = static_cast<int>(local_rows * cols);
    MPI_Gatherv(local.empty() ? nullptr : local.data(), send_count, MPI_DOUBLE,
                rank == 0 && !all.empty() ? all.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void broadcast_double_vector(vector<double>& values, int rank) {
    int count = rank == 0 ? static_cast<int>(values.size()) : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        values.resize(count);
    }
    if (count > 0) {
        MPI_Bcast(values.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void broadcast_string(string& value, int rank) {
    int count = rank == 0 ? static_cast<int>(value.size()) : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        value.resize(count);
    }
    if (count > 0) {
        MPI_Bcast(value.data(), count, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
}

uint64_t allreduce_sum_u64(uint64_t local) {
    unsigned long long send = local;
    unsigned long long receive = 0;
    MPI_Allreduce(&send, &receive, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return receive;
}

uint64_t current_host_box_limit(const Options& options,
                                     int ranks_on_node) {
    uint64_t local_budget = options.host_box_limit_bytes;
    if (local_budget == 0) {
        // Both generations coexist while splitting. Compute the per-rank
        // budget on each node first, then use one world-wide minimum so every
        // rank makes the same storage decision even with uneven node sizes.
        local_budget = tamubo::exactbo::available_host_memory_bytes() / 3;
        local_budget /= max(ranks_on_node, 1);
    }
    unsigned long long send = local_budget;
    unsigned long long receive = 0;
    MPI_Allreduce(&send, &receive, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                  MPI_COMM_WORLD);
    return receive;
}

uint64_t exscan_sum_u64(uint64_t local, int rank) {
    unsigned long long send = local;
    unsigned long long prefix = 0;
    MPI_Exscan(&send, &prefix, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return rank == 0 ? 0 : prefix;
}

uint64_t count_mask(const vector<unsigned char>& mask) {
    uint64_t count = 0;
    for (unsigned char value : mask) {
        count += value != 0;
    }
    return count;
}

// -----------------------------------------------------------------------------
// Batched host-to-GPU workflows
// -----------------------------------------------------------------------------

vector<double> compute_ei_hi_streamed(
    const BoxStore& boxes,
    const PartitionInput& input,
    int device_batch_rows,
    int rank,
    int size,
    int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const uint64_t start = local_start_for_rank(boxes.rows(), rank, size);
    const uint64_t local_n = local_count_for_rank(boxes.rows(), rank, size);
    vector<double> local_ei(local_n);

    double *d_X = nullptr, *d_alpha = nullptr, *d_L = nullptr, *d_length = nullptr;
    if (local_n != 0) {
        copy_to_device(&d_X, input.X_train.data(),
                       input.X_train.size() * sizeof(double), "d_X_train");
        copy_to_device(&d_alpha, input.alpha.data(),
                       input.alpha.size() * sizeof(double), "d_alpha");
        copy_to_device(&d_L, input.L.data(),
                       input.L.size() * sizeof(double), "d_cholesky_L");
        copy_to_device(&d_length, input.length_scale.data(),
                       input.length_scale.size() * sizeof(double), "d_length_scale");
    }

    for (uint64_t offset = 0; offset < local_n; offset += device_batch_rows) {
        const int rows = static_cast<int>(min<uint64_t>(device_batch_rows, local_n - offset));
        size_t box_elements = rows;
        box_elements *= input.d;
        size_t work_elements = rows;
        work_elements *= input.n_train;
        vector<double> host_L(box_elements);
        vector<double> host_U(box_elements);
        boxes.read_rows(start + offset, rows, host_L.data(), host_U.data());

        double *d_boxes_L = nullptr, *d_boxes_U = nullptr;
        double *d_Klo = nullptr, *d_Khi = nullptr, *d_vlo = nullptr, *d_vhi = nullptr;
        double* d_ei = nullptr;
        copy_to_device(&d_boxes_L, host_L.data(),
                       box_elements * sizeof(double), "d_boxes_L");
        copy_to_device(&d_boxes_U, host_U.data(),
                       box_elements * sizeof(double), "d_boxes_U");
        allocate_device(&d_Klo, work_elements * sizeof(double), "d_Klo");
        allocate_device(&d_Khi, work_elements * sizeof(double), "d_Khi");
        allocate_device(&d_vlo, work_elements * sizeof(double), "d_vlo");
        allocate_device(&d_vhi, work_elements * sizeof(double), "d_vhi");
        allocate_device(&d_ei, rows * sizeof(double), "d_ei");

        const int block = 128;
        const int grid = max(
            1, min((rows + block - 1) / block, prop.maxGridSize[0]));
        ei_hi_bounds_kernel<<<grid, block>>>(
            d_boxes_L, d_boxes_U, d_X, d_alpha, d_L, d_length,
            rows, input.n_train, input.d, input.sigma_f_2, input.y_min_scaled,
            d_Klo, d_Khi, d_vlo, d_vhi, d_ei);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(local_ei.data() + offset, d_ei, rows * sizeof(double),
                              cudaMemcpyDeviceToHost));

        free_device(d_ei);
        free_device(d_vhi);
        free_device(d_vlo);
        free_device(d_Khi);
        free_device(d_Klo);
        free_device(d_boxes_U);
        free_device(d_boxes_L);
    }

    free_device(d_length);
    free_device(d_L);
    free_device(d_alpha);
    free_device(d_X);
    return local_ei;
}

BestSample sample_best_streamed(
    const BoxStore& boxes,
    const vector<unsigned char>& local_mask,
    const vector<double>& lhs,
    const PartitionInput& input,
    int device_batch_rows,
    int rank,
    int size,
    int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const uint64_t n_samples = 1ULL << input.d;
    const uint64_t start = local_start_for_rank(boxes.rows(), rank, size);
    const uint64_t local_n = local_count_for_rank(boxes.rows(), rank, size);
    if (local_mask.size() != local_n) {
        throw runtime_error("local sample mask has the wrong size");
    }

    double *d_X = nullptr, *d_alpha = nullptr, *d_L = nullptr;
    double *d_length = nullptr, *d_lhs = nullptr;
    if (local_n != 0) {
        copy_to_device(&d_X, input.X_train.data(),
                       input.X_train.size() * sizeof(double), "d_X_train");
        copy_to_device(&d_alpha, input.alpha.data(),
                       input.alpha.size() * sizeof(double), "d_alpha");
        copy_to_device(&d_L, input.L.data(),
                       input.L.size() * sizeof(double), "d_cholesky_L");
        copy_to_device(&d_length, input.length_scale.data(),
                       input.length_scale.size() * sizeof(double), "d_length_scale");
        copy_to_device(&d_lhs, lhs.data(),
                       lhs.size() * sizeof(double), "d_lhs");
    }

    BestSample local_best;
    local_best.x.assign(input.d, 0.0);
    for (uint64_t offset = 0; offset < local_n; offset += device_batch_rows) {
        const int rows = static_cast<int>(min<uint64_t>(device_batch_rows, local_n - offset));
        size_t box_elements = rows;
        box_elements *= input.d;
        size_t work_elements = rows;
        work_elements *= input.n_train;
        vector<double> host_L(box_elements);
        vector<double> host_U(box_elements);
        vector<double> batch_ei(rows);
        vector<double> batch_points(box_elements);
        boxes.read_rows(start + offset, rows, host_L.data(), host_U.data());

        double *d_boxes_L = nullptr, *d_boxes_U = nullptr, *d_v = nullptr;
        double *d_best_ei = nullptr, *d_best_points = nullptr;
        unsigned char* d_mask = nullptr;
        copy_to_device(&d_boxes_L, host_L.data(),
                       box_elements * sizeof(double), "d_boxes_L");
        copy_to_device(&d_boxes_U, host_U.data(),
                       box_elements * sizeof(double), "d_boxes_U");
        copy_to_device(&d_mask, local_mask.data() + offset,
                       rows * sizeof(unsigned char), "d_mask");
        allocate_device(&d_v,
                        work_elements * sizeof(double), "d_v");
        allocate_device(&d_best_ei,
                        rows * sizeof(double), "d_best_ei");
        allocate_device(&d_best_points,
                        box_elements * sizeof(double), "d_best_points");

        const int block = 128;
        const int grid = max(
            1, min((rows + block - 1) / block, prop.maxGridSize[0]));
        sample_best_kernel<<<grid, block>>>(
            d_boxes_L, d_boxes_U, d_mask, d_X, d_alpha, d_L, d_length, d_lhs,
            rows, input.n_train, input.d, n_samples, input.sigma_f_2,
            input.y_min_scaled, d_v, d_best_ei, d_best_points);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(batch_ei.data(), d_best_ei, rows * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(batch_points.data(), d_best_points,
                              box_elements * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < rows; ++i) {
            if (batch_ei[i] > local_best.ei) {
                local_best.ei = batch_ei[i];
                local_best.box_idx = static_cast<long long>(start + offset + i);
                copy_n(batch_points.data() + i * input.d, input.d,
                            local_best.x.data());
            }
        }

        free_device(d_best_points);
        free_device(d_best_ei);
        free_device(d_v);
        free_device(d_mask);
        free_device(d_boxes_U);
        free_device(d_boxes_L);
    }

    free_device(d_lhs);
    free_device(d_length);
    free_device(d_L);
    free_device(d_alpha);
    free_device(d_X);

    struct {
        double value;
        int rank;
    } local_candidate{local_best.ei, rank}, winner{};
    MPI_Allreduce(&local_candidate, &winner, 1, MPI_DOUBLE_INT,
                  MPI_MAXLOC, MPI_COMM_WORLD);
    if (rank != winner.rank) {
        local_best.box_idx = -1;
        fill(local_best.x.begin(), local_best.x.end(), 0.0);
    }
    MPI_Bcast(&local_best.box_idx, 1, MPI_LONG_LONG, winner.rank, MPI_COMM_WORLD);
    MPI_Bcast(local_best.x.data(), input.d,
              MPI_DOUBLE, winner.rank, MPI_COMM_WORLD);
    local_best.ei = winner.value;
    return local_best;
}

// -----------------------------------------------------------------------------
// Convergence and memory planning
// -----------------------------------------------------------------------------

bool global_box_is_narrow(
    const BoxStore& boxes,
    long long box_idx,
    const vector<double>& epsilon_x,
    int rank,
    int size) {
    int local_narrow = 0;
    if (box_idx >= 0) {
        const uint64_t start = local_start_for_rank(boxes.rows(), rank, size);
        const uint64_t local_n = local_count_for_rank(boxes.rows(), rank, size);
        const uint64_t index = static_cast<uint64_t>(box_idx);
        if (index >= start && index < start + local_n) {
            vector<double> lower(boxes.dims());
            vector<double> upper(boxes.dims());
            boxes.read_rows(index, 1, lower.data(), upper.data());
            local_narrow = 1;
            for (uint64_t dim = 0; dim < boxes.dims(); ++dim) {
                if (!(upper[dim] - lower[dim] < epsilon_x[dim])) {
                    local_narrow = 0;
                    break;
                }
            }
        }
    }
    int narrow = 0;
    MPI_Allreduce(&local_narrow, &narrow, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return narrow != 0;
}

int ranks_sharing_device(int device) {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm device_comm;
    MPI_Comm_split(local_comm, device, 0, &device_comm);
    int count = 1;
    MPI_Comm_size(device_comm, &count);
    MPI_Comm_free(&device_comm);
    MPI_Comm_free(&local_comm);
    return count;
}

int choose_split_batch_parents(
    uint64_t local_targets,
    int d,
    int requested,
    int ranks_per_device,
    int ranks_on_node,
    int device) {
    if (local_targets == 0) {
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(device));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const uint64_t one_gib = 1ULL << 30;
    const uint64_t device_reserve =
        max<uint64_t>(
            one_gib, total_bytes / 10);
    uint64_t device_usable =
        free_bytes > device_reserve
            ? free_bytes - device_reserve
            : 0;
    device_usable /=
        max(ranks_per_device, 1);

    const uint64_t host_available =
        tamubo::exactbo::available_host_memory_bytes() /
        max(ranks_on_node, 1);
    const uint64_t host_reserve =
        max<uint64_t>(256ULL << 20, host_available / 10);
    const uint64_t host_usable =
        host_available > host_reserve ? host_available - host_reserve : 0;

    const uint64_t host_bytes_per_parent =
        32 * d * (d + 1);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int uses_host_page_tables = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &uses_host_page_tables,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
    const bool shared_physical_memory =
        prop.integrated || uses_host_page_tables != 0;
    const uint64_t device_bytes_per_parent =
        shared_physical_memory
            ? host_bytes_per_parent * 2
            : host_bytes_per_parent;
    const uint64_t device_slack = 64ULL << 20;
    if (device_usable <= device_slack ||
        device_usable - device_slack < device_bytes_per_parent ||
        host_usable < host_bytes_per_parent) {
        throw runtime_error(
            "not enough host/device memory for one compact split parent");
    }

    uint64_t batch =
        (device_usable - device_slack) / device_bytes_per_parent;
    batch = min(batch, host_usable / host_bytes_per_parent);
    batch = min<uint64_t>(batch, 1ULL << 20);
    if (requested != 0 && batch > requested) {
        batch = requested;
    }
    return static_cast<int>(max<uint64_t>(1, min(batch, local_targets)));
}

bool use_file_for_next_store(
    const BoxStore& current,
    uint64_t output_rows,
    int d,
    const Options& options,
    uint64_t host_limit_bytes) {
    if (options.box_storage == "file") {
        return true;
    }
    if (options.box_storage == "host") {
        return false;
    }
    if (current.file_backed()) {
        return true;
    }
    const uint64_t current_bytes =
        tamubo::exactbo::box_data_bytes(current.rows(), d);
    const uint64_t output_bytes =
        tamubo::exactbo::box_data_bytes(output_rows, d);
    if (current_bytes > host_limit_bytes ||
        output_bytes > host_limit_bytes - min(current_bytes, host_limit_bytes)) {
        return true;
    }
    const uint64_t output_elements =
        output_rows * d;
    return output_elements >
           numeric_limits<int>::max();
}

// -----------------------------------------------------------------------------
// Batched splitting into either host RAM or a shared spill file
// -----------------------------------------------------------------------------

unique_ptr<BoxStore> split_selected_streamed(
    const BoxStore& boxes,
    const vector<unsigned char>& local_target_mask,
    const PartitionInput& input,
    const Options& options,
    uint64_t host_limit_bytes,
    const fs::path& spill_run_dir,
    uint64_t generation,
    int rank,
    int size,
    int device,
    int ranks_per_device,
    int ranks_on_node) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const uint64_t start = local_start_for_rank(boxes.rows(), rank, size);
    const uint64_t local_n = local_count_for_rank(boxes.rows(), rank, size);
    if (local_target_mask.size() != local_n) {
        throw runtime_error("local target mask has the wrong size");
    }
    const int stride = 2 * input.d + 1;
    const uint64_t local_targets = count_mask(local_target_mask);
    const uint64_t global_targets = allreduce_sum_u64(local_targets);
    const uint64_t output_rows =
        global_targets * stride;
    const uint64_t local_output_rows =
        local_targets * stride;
    const uint64_t target_prefix = exscan_sum_u64(local_targets, rank);
    const uint64_t output_start =
        target_prefix * stride;

    bool file_output =
        use_file_for_next_store(boxes, output_rows, input.d, options,
                                host_limit_bytes);
    const uint64_t output_elements =
        output_rows * input.d;
    if (!file_output &&
        output_elements > numeric_limits<int>::max()) {
        if (options.box_storage == "host") {
            throw runtime_error(
                "forced host box storage exceeds MPI int-count limits; use file storage");
        }
        file_output = true;
    }

    // Keep all ranks on exactly the same collective path. The MAX also makes
    // auto mode fail safe: if any rank cannot safely host both generations,
    // every rank writes the shared file-backed generation.
    int local_file_output = file_output ? 1 : 0;
    int global_file_output = 0;
    MPI_Allreduce(&local_file_output, &global_file_output, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    file_output = global_file_output != 0;

    fs::path partial_path;
    fs::path final_path;
    if (file_output) {
        partial_path = spill_run_dir /
            ("boxes_" + to_string(generation) + ".partial");
        final_path = spill_run_dir /
            ("boxes_" + to_string(generation) + ".box");
        if (rank == 0) {
            fs::create_directories(spill_run_dir);
            const uint64_t required =
                tamubo::exactbo::box_data_bytes(output_rows, input.d) + 4096;
            const uint64_t available =
                tamubo::exactbo::available_filesystem_bytes(spill_run_dir.string());
            const uint64_t reserve =
                min<uint64_t>(1ULL << 30, available / 10);
            if (available < required ||
                available - required < reserve) {
                throw runtime_error(
                    "insufficient spill filesystem space: required=" +
                    to_string(required) + " available=" +
                    to_string(available));
            }
            tamubo::exactbo::initialize_box_file(
                partial_path.string(), output_rows, input.d);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0 && options.verbose) {
        cout << "  split_store=" << (file_output ? "file" : "host")
                  << " output_boxes=" << output_rows
                  << " output_bytes="
                  << format_bytes(tamubo::exactbo::box_data_bytes(output_rows, input.d));
        if (file_output) {
            cout << " spill=\"" << final_path.string() << "\"";
        }
        cout << "\n";
    }

    const int split_batch = choose_split_batch_parents(
        local_targets, input.d, options.split_batch_parents,
        ranks_per_device, ranks_on_node, device);
    vector<double> domain_width(input.d);
    for (int dim = 0; dim < input.d; ++dim) {
        domain_width[dim] = input.domain_U[dim] - input.domain_L[dim];
    }

    size_t parent_capacity_elements = split_batch;
    parent_capacity_elements *= input.d;
    size_t child_capacity_rows = split_batch;
    child_capacity_rows *= stride;
    size_t child_capacity_elements = child_capacity_rows;
    child_capacity_elements *= input.d;
    vector<double> parent_L(parent_capacity_elements);
    vector<double> parent_U(parent_capacity_elements);
    vector<double> child_L(child_capacity_elements);
    vector<double> child_U(child_capacity_elements);
    vector<double> local_output_L;
    vector<double> local_output_U;
    if (!file_output) {
        const size_t local_elements = local_output_rows * input.d;
        local_output_L.reserve(local_elements);
        local_output_U.reserve(local_elements);
    }

    unique_ptr<FileBoxWriter> file_writer;
    if (file_output) {
        file_writer = make_unique<FileBoxWriter>(partial_path.string());
    }

    double *d_parent_L = nullptr, *d_parent_U = nullptr;
    double *d_domain_width = nullptr, *d_child_L = nullptr, *d_child_U = nullptr;
    if (local_targets != 0) {
        allocate_device(&d_parent_L,
                        parent_capacity_elements * sizeof(double), "d_split_parent_L");
        allocate_device(&d_parent_U,
                        parent_capacity_elements * sizeof(double), "d_split_parent_U");
        copy_to_device(&d_domain_width, domain_width.data(),
                       domain_width.size() * sizeof(double), "d_domain_width");
        allocate_device(&d_child_L,
                        child_capacity_elements * sizeof(double), "d_split_child_L");
        allocate_device(&d_child_U,
                        child_capacity_elements * sizeof(double), "d_split_child_U");
    }

    int pending = 0;
    uint64_t written_rows = 0;
    auto flush = [&]() {
        if (pending == 0) {
            return;
        }
        size_t parent_elements = pending;
        parent_elements *= input.d;
        const int child_rows = pending * stride;
        size_t child_elements = child_rows;
        child_elements *= input.d;
        CUDA_CHECK(cudaMemcpy(d_parent_L, parent_L.data(),
                              parent_elements * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_parent_U, parent_U.data(),
                              parent_elements * sizeof(double),
                              cudaMemcpyHostToDevice));
        const int block = 128;
        const int grid = max(
            1, min((pending + block - 1) / block, prop.maxGridSize[0]));
        split_dense_boxes_kernel<<<grid, block>>>(
            d_parent_L, d_parent_U, d_domain_width, pending, input.d,
            stride, d_child_L, d_child_U);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(child_L.data(), d_child_L,
                              child_elements * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(child_U.data(), d_child_U,
                              child_elements * sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (file_output) {
            file_writer->write_rows(output_start + written_rows, child_rows,
                                    child_L.data(), child_U.data());
        } else {
            local_output_L.insert(local_output_L.end(), child_L.begin(),
                                  child_L.begin() + child_elements);
            local_output_U.insert(local_output_U.end(), child_U.begin(),
                                  child_U.begin() + child_elements);
        }
        written_rows += child_rows;
        pending = 0;
    };

    const int scan_rows = local_n == 0
        ? 1
        : static_cast<int>(min<uint64_t>(options.device_batch_rows, local_n));
    size_t scan_elements = scan_rows;
    scan_elements *= input.d;
    vector<double> scan_L(scan_elements);
    vector<double> scan_U(scan_L.size());
    for (uint64_t offset = 0; offset < local_n; offset += scan_rows) {
        const int rows = static_cast<int>(min<uint64_t>(scan_rows, local_n - offset));
        boxes.read_rows(start + offset, rows, scan_L.data(), scan_U.data());
        for (int row = 0; row < rows; ++row) {
            if (local_target_mask[offset + row] == 0) {
                continue;
            }
            copy_n(scan_L.data() + row * input.d, input.d,
                        parent_L.data() + pending * input.d);
            copy_n(scan_U.data() + row * input.d, input.d,
                        parent_U.data() + pending * input.d);
            ++pending;
            if (pending == split_batch) {
                flush();
            }
        }
    }
    flush();

    free_device(d_child_U);
    free_device(d_child_L);
    free_device(d_domain_width);
    free_device(d_parent_U);
    free_device(d_parent_L);

    if (written_rows != local_output_rows) {
        throw runtime_error("split writer produced the wrong local row count");
    }

    if (file_output) {
        file_writer->sync_and_drop_cache();
        file_writer.reset();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            tamubo::exactbo::finalize_box_file(
                partial_path.string(), final_path.string());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return make_unique<FileBoxStore>(final_path.string());
    }

    vector<double> gathered_L;
    vector<double> gathered_U;
    uint64_t total_L = 0;
    uint64_t total_U = 0;
    gather_double_rows(local_output_L, local_output_rows, input.d,
                       rank, size, gathered_L, total_L);
    gather_double_rows(local_output_U, local_output_rows, input.d,
                       rank, size, gathered_U, total_U);
    if (rank == 0 && (total_L != output_rows || total_U != output_rows)) {
        throw runtime_error("host split gather produced the wrong row count");
    }
    broadcast_double_vector(gathered_L, rank);
    broadcast_double_vector(gathered_U, rank);
    return make_unique<HostBoxStore>(
        output_rows, input.d, move(gathered_L), move(gathered_U));
}

}  // namespace

// -----------------------------------------------------------------------------
// Program entry point and ExactBO partition loop
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // Flush native messages immediately while Python tees the pipe to a log.
    cout.setf(ios::unitbuf);
    cerr.setf(ios::unitbuf);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fs::path spill_run_dir;
    Options options;
    try {
        options = parse_args(argc, argv);
        PartitionInput input = read_input(options.input_path);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw runtime_error("no CUDA devices visible");
        }
        const int local_rank = local_rank_on_node();
        const int device = local_rank % device_count;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        const int ranks_per_device = ranks_sharing_device(device);
        const int ranks_on_node = local_size_on_node();

        uint64_t host_limit =
            current_host_box_limit(options, ranks_on_node);

        string spill_run_path;
        if (rank == 0) {
            fs::path spill_base;
            if (!options.spill_dir.empty()) {
                spill_base = options.spill_dir;
            } else {
                fs::path output_path(options.output_path);
                spill_base = output_path.has_parent_path()
                    ? output_path.parent_path() / ".exactbo-spill"
                    : fs::path(".exactbo-spill");
            }
            fs::create_directories(spill_base);
            spill_base = fs::absolute(spill_base);
            const fs::path run_template =
                spill_base / "run_XXXXXX";
            const string template_text = run_template.string();
            vector<char> mutable_template(
                template_text.begin(), template_text.end());
            mutable_template.push_back('\0');
            char* created = ::mkdtemp(mutable_template.data());
            if (created == nullptr) {
                throw runtime_error(
                    "failed to create unique spill directory: " +
                    string(strerror(errno)));
            }
            spill_run_path = created;
        }
        broadcast_string(spill_run_path, rank);
        spill_run_dir = spill_run_path;

        if (rank == 0) {
            cout << "exactbo_partitioning: mpi_size=" << size
                      << " device_count=" << device_count
                      << " input=\"" << options.input_path << "\""
                      << " output=\"" << options.output_path << "\""
                      << " n_train=" << input.n_train
                      << " d=" << input.d
                      << " max_partitions=" << input.max_partitions
                      << " device_batch_rows=" << options.device_batch_rows
                      << " split_batch_parents=" << options.split_batch_parents
                      << " box_storage=" << options.box_storage
                      << " host_box_limit=" << format_bytes(
                             host_limit)
                      << " epsilon_ei=" << input.epsilon_ei << "\n";
        }
        cout << "rank " << rank << " local_rank=" << local_rank
                  << " using gpu_device=" << device
                  << " ranks_on_device=" << ranks_per_device
                  << " gpu_name=\"" << prop.name << "\""
                  << " launch_cpu=" << current_cpu() << "\n";
        print_device_memory(rank, "startup");

        unique_ptr<BoxStore> boxes = make_unique<HostBoxStore>(
            1, input.d, input.domain_L, input.domain_U);
        const uint64_t n_samples = 1ULL << input.d;
        const int stride = 2 * input.d + 1;
        const vector<double> lhs =
            centered_latin_hypercube_unit(n_samples, input.d);
        BestSample best;
        best.x.assign(input.d, 0.0);
        uint64_t partitions_done = 0;
        int converged = 0;
        uint64_t preserved_start = 0;
        uint64_t preserved_count = 1;

        for (int partition = 0;
             partition < input.max_partitions; ++partition) {
            const uint64_t n_boxes = boxes->rows();
            const uint64_t start =
                local_start_for_rank(n_boxes, rank, size);
            const uint64_t local_n =
                local_count_for_rank(n_boxes, rank, size);
            // Upper bound: the best EI that each box could possibly contain.
            vector<double> local_ei = compute_ei_hi_streamed(
                *boxes, input, options.device_batch_rows, rank, size, device);

            double local_max = -numeric_limits<double>::infinity();
            for (double value : local_ei) {
                local_max = max(local_max, value);
            }
            double max_ei_hi = 0.0;
            MPI_Allreduce(&local_max, &max_ei_hi, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD);

            // First sample boxes near the largest upper bound, plus the children
            // of the best box from the previous partition.
            vector<unsigned char> analyze_mask(local_n, 0);
            for (uint64_t i = 0; i < local_n; ++i) {
                const uint64_t global_i = start + i;
                const bool preserve =
                    global_i >= preserved_start &&
                    global_i < preserved_start + preserved_count;
                analyze_mask[i] =
                    local_ei[i] >= (max_ei_hi - input.epsilon_ei) ||
                    preserve;
            }
            const uint64_t n_analyze =
                allreduce_sum_u64(count_mask(analyze_mask));
            if (rank == 0 && options.verbose) {
                cout << "partition " << partition
                          << " boxes=" << n_boxes
                          << " max_ei_hi=" << max_ei_hi
                          << " analyze=" << n_analyze << "\n";
            }

            // Feasible lower bound: EI at an actual point in an analyzed box.
            BestSample best_analyze = sample_best_streamed(
                *boxes, analyze_mask, lhs, input,
                options.device_batch_rows, rank, size, device);
            best = best_analyze;

            // A box is active if its upper bound can beat the sampled point by
            // more than epsilon_ei.
            vector<unsigned char> active_mask(local_n, 0);
            for (uint64_t i = 0; i < local_n; ++i) {
                active_mask[i] =
                    local_ei[i] >
                    (best_analyze.ei + input.epsilon_ei);
            }
            const uint64_t n_active =
                allreduce_sum_u64(count_mask(active_mask));
            int stop =
                n_active == 0 &&
                global_box_is_narrow(*boxes, best_analyze.box_idx,
                                     input.epsilon_x, rank, size);
            if (stop) {
                converged = 1;
                partitions_done = partition + 1;
                break;
            }
            if (best_analyze.box_idx >= 0) {
                const uint64_t best_idx =
                    static_cast<uint64_t>(best_analyze.box_idx);
                if (best_idx >= start && best_idx < start + local_n) {
                    active_mask[best_idx - start] = 1;
                }
            }

            // Resample the full active set to improve the feasible lower bound.
            BestSample best_active = sample_best_streamed(
                *boxes, active_mask, lhs, input,
                options.device_batch_rows, rank, size, device);
            best = best_active;

            // Only target boxes still capable of a meaningful EI improvement.
            vector<unsigned char> target_mask(local_n, 0);
            for (uint64_t i = 0; i < local_n; ++i) {
                target_mask[i] =
                    local_ei[i] >
                    (best_active.ei + input.epsilon_ei);
            }
            const uint64_t n_target_before_force =
                allreduce_sum_u64(count_mask(target_mask));
            stop =
                n_target_before_force == 0 &&
                global_box_is_narrow(*boxes, best_active.box_idx,
                                     input.epsilon_x, rank, size);
            if (stop) {
                converged = 1;
                partitions_done = partition + 1;
                break;
            }

            const uint64_t best_idx =
                static_cast<uint64_t>(best_active.box_idx);
            if (best_active.box_idx >= 0 &&
                best_idx >= start && best_idx < start + local_n) {
                target_mask[best_idx - start] = 1;
            }
            uint64_t local_targets_before_best = 0;
            for (uint64_t i = 0; i < local_n; ++i) {
                if (start + i >= best_idx) {
                    break;
                }
                local_targets_before_best += target_mask[i] != 0;
            }
            // Children are written in parent order.  Remember the child range
            // produced by the best parent so it is analyzed next time.
            preserved_start =
                allreduce_sum_u64(local_targets_before_best) * stride;
            preserved_count = stride;

            const uint64_t n_target =
                allreduce_sum_u64(count_mask(target_mask));
            if (rank == 0 && options.verbose) {
                cout << "  best_ei=" << best_active.ei
                          << " best_box=" << best_active.box_idx
                          << " target=" << n_target << "\n";
            }

            partitions_done = partition + 1;
            if (partition + 1 == input.max_partitions) {
                break;
            }

            const string old_file =
                boxes->file_backed() ? boxes->path() : string();
            // Re-evaluate auto mode against current cgroup/node pressure.
            host_limit = current_host_box_limit(options, ranks_on_node);
            // Generate the next box population in RAM or in a spill file.
            unique_ptr<BoxStore> next = split_selected_streamed(
                *boxes, target_mask, input, options, host_limit,
                spill_run_dir, partition, rank, size, device,
                ranks_per_device, ranks_on_node);
            boxes = move(next);

            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0 && !old_file.empty() &&
                !options.keep_spill_files) {
                error_code error;
                fs::remove(old_file, error);
                if (error) {
                    throw runtime_error(
                        "failed to remove old spill file " + old_file +
                        ": " + error.message());
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            write_output(options.output_path, best, input.d,
                         partitions_done, boxes->rows(), converged);
            cout << "rank 0 wrote exactbo_partitioning output to "
                      << options.output_path
                      << " best_ei_scaled=" << best.ei
                      << " converged=" << converged
                      << " partitions_done=" << partitions_done
                      << " n_boxes_final=" << boxes->rows() << "\n";
        }

        const string final_file =
            boxes->file_backed() ? boxes->path() : string();
        boxes.reset();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            error_code error;
            if (!options.keep_spill_files && !final_file.empty()) {
                fs::remove(final_file, error);
                if (error) {
                    throw runtime_error(
                        "failed to remove final spill file " + final_file +
                        ": " + error.message());
                }
            }
            if (!options.keep_spill_files || final_file.empty()) {
                error.clear();
                fs::remove(spill_run_dir, error);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } catch (const exception& exc) {
        cerr << "rank " << rank
                  << " exactbo_partitioning: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    cudaDeviceReset();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

/*
Beginner-readability changes made in this revision
--------------------------------------------------
1. Imported frequently used standard-library names with `using std::...` and
   introduced the short `fs` alias for `std::filesystem`.
2. Added section dividers and comments around the main ExactBO decisions.
3. Inlined the binary input reads so the Python/C++ file layout is visible in
   one place. Removed read_value(), read_doubles(), and the optional trailing-
   byte helper.
4. Replaced checked_mul() calls with direct arithmetic. Runtime allocation and
   file-space failures are still reported, but malformed inputs that overflow a
   64-bit product are no longer handled as a separate special case.
5. Removed mpi_count(). Host-backed collectives now use direct int counts;
   global box totals and file-backed offsets remain 64-bit.
6. Replaced untyped CUDA allocation helpers with typed templates. This removes
   repeated reinterpret_cast<void**>() expressions from every call site; the
   one cast required by the C cudaMalloc API is now contained in one function.
7. Removed redundant numeric casts. Casts remain only where binary streams,
   CUDA indexing, MPI, or an intentional narrowing conversion requires them.
8. Changed training size, dimension, partition limit, device_batch_rows,
   split_batch_parents, and per-batch loop counters to ordinary int values.
   Kept uint64_t for on-disk fields, total/local box populations, global box
   indexes, memory totals, prefix sums, and spill-file offsets.
9. The GP mathematics, EI criteria, splitting geometry, MPI ownership, storage
   policy, and output format were intentionally left unchanged.
10. Enabled immediate native stdout/stderr flushing so Python can tee live output
    to the terminal and the run log.

Future performance work recorded after the 8M/16M batch comparison
------------------------------------------------------------------
1. Add measurements before changing the algorithm. Record wall-clock and CUDA-
   event timings for EI bounds, EI sampling, mask construction, splitting,
   allocations, copies, and file I/O. Also log the number of batches and the
   process/device memory high-water marks. nvtop can miss short kernels, so its
   utilization display is useful context but is not a phase-level profiler.
2. Reuse persistent CUDA work buffers sized for the largest required batch
   instead of calling cudaMalloc() and cudaFree() for every batch. Grow a buffer
   only when necessary and release it after the complete partitioning run.
3. Use asynchronous copies and at least two CUDA streams to double-buffer work:
   while the GPU evaluates batch N, the host can read/prepare batch N+1 and copy
   it into the other buffer. Benchmark pinned, mapped, and ordinary host memory
   on GB10 because its CPU and GPU share physical memory.
4. Construct analyze, active, and target masks on the GPU and compute their
   counts with device reductions. This avoids repeated full scans of local_ei
   on the CPU and repeated host/device synchronization.
5. Compact sparse analyze/active masks before EI sampling. The current sampling
   path launches over and copies results for every box even when only a few
   boxes are selected. A device prefix sum or selected-index array should make
   sampling proportional to the selected population.
6. Reduce sampled EI and its best point on the GPU. Copy one winning candidate
   per batch back to the host instead of copying per-box EI values and points.
7. Compact split parents on the GPU, then feed the compact array directly to
   the split kernel. The current implementation scans every box on the CPU to
   collect selected parents before each split batch.
8. Preserve exact ordering and determinism while making these changes. Compare
   the complete partition trace, proposed points, best EI, and binary output
   against the current implementation for several batch sizes.
9. Do not assume that a larger batch is faster. The 8,388,608-row and
   16,777,216-row ideal runs produced identical partition traces and took about
   133 and 134 seconds. The larger run used longer 70--90% GPU bursts separated
   by longer host-side gaps. Increasing the limit again would consume more
   shared GB10 memory without addressing the serialized host work.
10. Suggested implementation order: instrumentation, persistent allocations,
    GPU reductions/compaction, device-side best reduction, and finally
    double-buffered asynchronous execution. Re-profile after every step.
*/
