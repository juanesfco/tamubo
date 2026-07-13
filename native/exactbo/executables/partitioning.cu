#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(status));                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
        }                                                                       \
    } while (0)

namespace {

constexpr char kInputMagic[8] = {'T', 'P', 'A', 'R', 'I', 'N', '1', '!'};
constexpr char kOutputMagic[8] = {'T', 'P', 'A', 'R', 'O', 'U', '1', '!'};
constexpr double kInvSqrt2Pi = 0.39894228040143267794;
constexpr double kSqrt2 = 1.41421356237309504880;

struct Options {
    std::string input_path;
    std::string output_path;
    int verbose = 0;
    // Bound every CUDA launch/allocation by processing at most this many boxes.
    std::uint64_t device_batch_rows = 4096;
};

struct PartitionInput {
    std::uint64_t n_train = 0;
    std::uint64_t d = 0;
    std::uint64_t max_partitions = 0;
    double epsilon_ei = 0.0;
    double sigma_f_2 = 1.0;
    double sigma_n_2 = 0.0;
    double y_train_mean = 0.0;
    double y_train_std = 1.0;
    double y_min_scaled = 0.0;
    std::vector<double> epsilon_x;
    std::vector<double> domain_L;
    std::vector<double> domain_U;
    std::vector<double> X_train;
    std::vector<double> alpha;
    std::vector<double> L;
    std::vector<double> length_scale;
};

struct BestSample {
    double ei = -std::numeric_limits<double>::infinity();
    long long box_idx = -1;
    std::vector<double> x;
};

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

std::string format_bytes(std::size_t bytes) {
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << static_cast<double>(bytes) / gib << " GiB";
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
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            options.verbose = 1;
        } else if (std::strcmp(argv[i], "--device-batch-rows") == 0) {
            const char* value = need_value("--device-batch-rows");
            std::size_t parsed = 0;
            options.device_batch_rows = std::stoull(value, &parsed);
            if (value[parsed] != '\0' || options.device_batch_rows == 0) {
                throw std::runtime_error("--device-batch-rows must be a positive integer");
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: exactbo_partitioning --input input.bin --output output.bin\n"
                      << "       [--device-batch-rows N] [--verbose]\n";
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
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
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
        throw std::runtime_error(label + " is too large for MPI int counts");
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

PartitionInput read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kInputMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid exactbo_partitioning input magic");
    }

    PartitionInput input;
    read_value(in, input.n_train, "n_train");
    read_value(in, input.d, "d");
    read_value(in, input.max_partitions, "max_partitions");
    read_value(in, input.epsilon_ei, "epsilon_ei");
    read_value(in, input.sigma_f_2, "sigma_f_2");
    read_value(in, input.sigma_n_2, "sigma_n_2");
    read_value(in, input.y_train_mean, "y_train_mean");
    read_value(in, input.y_train_std, "y_train_std");
    read_value(in, input.y_min_scaled, "y_min_scaled");

    if (input.n_train == 0 || input.d == 0 || input.max_partitions == 0) {
        throw std::runtime_error("n_train, d, and max_partitions must be > 0");
    }
    if (input.d >= 63) {
        throw std::runtime_error("d is too large for m=2^d samples");
    }

    input.epsilon_x.resize(input.d);
    input.domain_L.resize(input.d);
    input.domain_U.resize(input.d);
    input.X_train.resize(checked_mul(input.n_train, input.d, "n_train*d"));
    input.alpha.resize(input.n_train);
    input.L.resize(checked_mul(input.n_train, input.n_train, "n_train*n_train"));
    input.length_scale.resize(input.d);

    read_doubles(in, input.epsilon_x, "epsilon_x");
    read_doubles(in, input.domain_L, "domain_L");
    read_doubles(in, input.domain_U, "domain_U");
    read_doubles(in, input.X_train, "X_train");
    read_doubles(in, input.alpha, "alpha");
    read_doubles(in, input.L, "L");
    read_doubles(in, input.length_scale, "length_scale");
    check_no_trailing_bytes(in);

    for (std::uint64_t dim = 0; dim < input.d; ++dim) {
        if (!std::isfinite(input.domain_L[dim]) || !std::isfinite(input.domain_U[dim]) ||
            !(input.domain_L[dim] < input.domain_U[dim])) {
            throw std::runtime_error("each domain lower bound must be finite and smaller than its upper bound");
        }
        if (!std::isfinite(input.epsilon_x[dim]) || input.epsilon_x[dim] < 0.0) {
            throw std::runtime_error("epsilon_x values must be finite and nonnegative");
        }
        if (!std::isfinite(input.length_scale[dim]) || !(input.length_scale[dim] > 0.0)) {
            throw std::runtime_error("length scales must be finite and positive");
        }
    }
    if (!std::isfinite(input.epsilon_ei) || input.epsilon_ei < 0.0 ||
        !std::isfinite(input.sigma_f_2) || !(input.sigma_f_2 > 0.0) ||
        !std::isfinite(input.y_min_scaled)) {
        throw std::runtime_error("invalid EI tolerance or GP scalar parameters");
    }
    for (std::uint64_t j = 0; j < input.n_train; ++j) {
        double diagonal = input.L[j * input.n_train + j];
        if (!std::isfinite(diagonal) || !(diagonal > 0.0)) {
            throw std::runtime_error("the Cholesky factor must have a finite positive diagonal");
        }
    }
    return input;
}

void write_output(
    const std::string& path,
    const BestSample& best,
    std::uint64_t d,
    std::uint64_t partitions_done,
    std::uint64_t n_boxes_final,
    int converged) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out.write(kOutputMagic, sizeof(kOutputMagic));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(&partitions_done), sizeof(partitions_done));
    out.write(reinterpret_cast<const char*>(&n_boxes_final), sizeof(n_boxes_final));
    out.write(reinterpret_cast<const char*>(&converged), sizeof(converged));
    out.write(reinterpret_cast<const char*>(&best.ei), sizeof(best.ei));
    out.write(reinterpret_cast<const char*>(best.x.data()), static_cast<std::streamsize>(d * sizeof(double)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

std::vector<double> centered_latin_hypercube_unit(std::uint64_t n_points, std::uint64_t d) {
    std::vector<double> lhs(checked_mul(n_points, d, "lhs elements"));
    std::vector<double> centers(n_points);
    for (std::uint64_t i = 0; i < n_points; ++i) {
        centers[i] = (static_cast<double>(i) + 0.5) / static_cast<double>(n_points);
    }
    for (std::uint64_t j = 0; j < d; ++j) {
        std::uint64_t step = 2 * j + 1;
        while (std::gcd(step, n_points) != 1) {
            step += 2;
        }
        for (std::uint64_t i = 0; i < n_points; ++i) {
            lhs[i * d + j] = centers[(i * step + j) % n_points];
        }
    }
    return lhs;
}

void check_cuda_alloc(cudaError_t status, const char* label, std::size_t bytes, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::fprintf(stderr,
                 "CUDA allocation error at %s:%d for %s: requested=%zu bytes free=%zu bytes total=%zu bytes error=%s\n",
                 file, line, label, bytes, free_bytes, total_bytes, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

#define CUDA_ALLOC(dst, bytes, label)                                                \
    do {                                                                             \
        std::size_t cuda_alloc_bytes = (bytes);                                      \
        cudaError_t cuda_alloc_status = cudaMalloc((dst), cuda_alloc_bytes);         \
        check_cuda_alloc(cuda_alloc_status, (label), cuda_alloc_bytes, __FILE__, __LINE__); \
    } while (0)

void copy_to_device(void** dst, const void* src, std::size_t bytes, const char* label) {
    if (bytes == 0) {
        *dst = nullptr;
        return;
    }
    CUDA_ALLOC(dst, bytes, label);
    CUDA_CHECK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
}

void allocate_device(void** dst, std::size_t bytes, const char* label) {
    if (bytes == 0) {
        *dst = nullptr;
        return;
    }
    CUDA_ALLOC(dst, bytes, label);
}

void free_device(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
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
    std::uint64_t local_n,
    std::uint64_t n_train,
    std::uint64_t d,
    double sigma_f_2,
    double y_min_scaled,
    double* K_lo,
    double* K_hi,
    double* v_lo,
    double* v_hi,
    double* ei_hi) {
    for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        double mu_lo = 0.0;
        double mu_hi = 0.0;
        for (std::uint64_t j = 0; j < n_train; ++j) {
            double dmin_sq = 0.0;
            double dmax_sq = 0.0;
            for (std::uint64_t dim = 0; dim < d; ++dim) {
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
        for (std::uint64_t j = 0; j < n_train; ++j) {
            double sum_lo = 0.0;
            double sum_hi = 0.0;
            for (std::uint64_t i = 0; i < j; ++i) {
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
    std::uint64_t local_n,
    std::uint64_t n_train,
    std::uint64_t d,
    std::uint64_t n_samples,
    double sigma_f_2,
    double y_min_scaled,
    double* v,
    double* best_ei,
    double* best_points) {
    for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < local_n;
         row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        if (mask[row] == 0) {
            best_ei[row] = -INFINITY;
            for (std::uint64_t dim = 0; dim < d; ++dim) {
                best_points[row * d + dim] = 0.0;
            }
            continue;
        }

        double row_best_ei = -INFINITY;
        std::uint64_t row_best_sample = 0;
        for (std::uint64_t sample = 0; sample < n_samples; ++sample) {
            double mu = 0.0;
            double q = 0.0;
            for (std::uint64_t j = 0; j < n_train; ++j) {
                double sqdist = 0.0;
                for (std::uint64_t dim = 0; dim < d; ++dim) {
                    double lo = boxes_L[row * d + dim];
                    double hi = boxes_U[row * d + dim];
                    double x = lo + (hi - lo) * lhs[sample * d + dim];
                    double diff = (x - X_train[j * d + dim]) / length_scale[dim];
                    sqdist += diff * diff;
                }
                double k = sigma_f_2 * exp(-0.5 * sqdist);
                mu += k * alpha[j];
                double sum = 0.0;
                for (std::uint64_t i = 0; i < j; ++i) {
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
        for (std::uint64_t dim = 0; dim < d; ++dim) {
            double lo = boxes_L[row * d + dim];
            double hi = boxes_U[row * d + dim];
            best_points[row * d + dim] = lo + (hi - lo) * lhs[row_best_sample * d + dim];
        }
    }
}

__device__ double normalized_width(const double* boxes_L, const double* boxes_U, const double* domain_width, std::uint64_t row, std::uint64_t d, std::uint64_t dim) {
    double width = boxes_U[row * d + dim] - boxes_L[row * d + dim];
    double scale = domain_width[dim];
    return scale != 0.0 ? width / scale : width;
}

__device__ std::uint64_t split_dim_for_rank(const double* boxes_L, const double* boxes_U, const double* domain_width, std::uint64_t row, std::uint64_t d, std::uint64_t rank) {
    for (std::uint64_t dim = 0; dim < d; ++dim) {
        double score = normalized_width(boxes_L, boxes_U, domain_width, row, d, dim);
        std::uint64_t order = 0;
        for (std::uint64_t other = 0; other < d; ++other) {
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

__global__ void split_selected_boxes_kernel(
    const double* boxes_L,
    const double* boxes_U,
    const double* domain_width,
    const std::uint64_t* selected_indices,
    std::uint64_t selected_count,
    std::uint64_t d,
    std::uint64_t stride,
    double* out_L,
    double* out_U) {
    for (std::uint64_t pos = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pos < selected_count;
         pos += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        std::uint64_t src = selected_indices[pos];
        std::uint64_t out_base = pos * stride;
        for (std::uint64_t child = 0; child < stride; ++child) {
            for (std::uint64_t dim = 0; dim < d; ++dim) {
                out_L[(out_base + child) * d + dim] = boxes_L[src * d + dim];
                out_U[(out_base + child) * d + dim] = boxes_U[src * d + dim];
            }
        }
        for (std::uint64_t rank = 0; rank < d; ++rank) {
            std::uint64_t dim = split_dim_for_rank(boxes_L, boxes_U, domain_width, src, d, rank);
            double low = boxes_L[src * d + dim];
            double high = boxes_U[src * d + dim];
            double third = (high - low) / 3.0;
            double lower_third = low + third;
            double upper_third = high - third;
            std::uint64_t lower_row = 2 * rank;
            std::uint64_t upper_row = lower_row + 1;
            out_U[(out_base + lower_row) * d + dim] = lower_third;
            out_L[(out_base + upper_row) * d + dim] = upper_third;
            for (std::uint64_t child = upper_row + 1; child < stride; ++child) {
                out_L[(out_base + child) * d + dim] = lower_third;
                out_U[(out_base + child) * d + dim] = upper_third;
            }
        }
    }
}

void gather_double_rows(
    const std::vector<double>& local,
    std::uint64_t local_rows,
    std::uint64_t cols,
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
            std::uint64_t elems = checked_mul(rows, cols, "MPI gather elements");
            counts[r] = checked_int_count(elems, "MPI gather elements");
            displs[r] = offset;
            offset += counts[r];
            total_rows += rows;
        }
        all.resize(checked_mul(total_rows, cols, "gathered output elements"));
    }
    int send_count = checked_int_count(checked_mul(local_rows, cols, "local gather elements"), "local gather elements");
    MPI_Gatherv(local.empty() ? nullptr : local.data(), send_count, MPI_DOUBLE,
                rank == 0 && !all.empty() ? all.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void broadcast_double_vector(std::vector<double>& values, int rank) {
    unsigned long long size_ull = rank == 0 ? static_cast<unsigned long long>(values.size()) : 0ULL;
    MPI_Bcast(&size_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        values.resize(static_cast<std::size_t>(size_ull));
    }
    if (size_ull != 0) {
        MPI_Bcast(values.data(), checked_int_count(static_cast<std::uint64_t>(size_ull), "broadcast vector"), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

std::vector<double> compute_ei_hi_global(
    const std::vector<double>& boxes_L,
    const std::vector<double>& boxes_U,
    const PartitionInput& input,
    std::uint64_t device_batch_rows,
    int rank,
    int size,
    int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::uint64_t n_boxes = static_cast<std::uint64_t>(boxes_L.size()) / input.d;
    const std::uint64_t start = local_start_for_rank(n_boxes, rank, size);
    const std::uint64_t local_n = local_count_for_rank(n_boxes, rank, size);
    const std::vector<double> local_L = slice_rows(boxes_L, start, local_n, input.d);
    const std::vector<double> local_U = slice_rows(boxes_U, start, local_n, input.d);
    std::vector<double> local_ei(local_n);

    // GP state is constant across batches, so upload it once for this phase.
    double *d_X = nullptr, *d_alpha = nullptr, *d_L = nullptr, *d_length = nullptr;
    if (local_n != 0) {
        copy_to_device(reinterpret_cast<void**>(&d_X), input.X_train.data(), input.X_train.size() * sizeof(double), "d_X_train");
        copy_to_device(reinterpret_cast<void**>(&d_alpha), input.alpha.data(), input.alpha.size() * sizeof(double), "d_alpha");
        copy_to_device(reinterpret_cast<void**>(&d_L), input.L.data(), input.L.size() * sizeof(double), "d_cholesky_L");
        copy_to_device(reinterpret_cast<void**>(&d_length), input.length_scale.data(), input.length_scale.size() * sizeof(double), "d_length_scale");
    }

    for (std::uint64_t offset = 0; offset < local_n; offset += device_batch_rows) {
        const std::uint64_t rows = std::min(device_batch_rows, local_n - offset);
        const std::size_t box_elements = checked_mul(rows, input.d, "EI batch box elements");
        const std::size_t work_elements = checked_mul(rows, input.n_train, "EI batch work elements");

        double *d_boxes_L = nullptr, *d_boxes_U = nullptr;
        double *d_Klo = nullptr, *d_Khi = nullptr, *d_vlo = nullptr, *d_vhi = nullptr, *d_ei = nullptr;
        copy_to_device(reinterpret_cast<void**>(&d_boxes_L), local_L.data() + offset * input.d,
                       box_elements * sizeof(double), "d_boxes_L");
        copy_to_device(reinterpret_cast<void**>(&d_boxes_U), local_U.data() + offset * input.d,
                       box_elements * sizeof(double), "d_boxes_U");
        allocate_device(reinterpret_cast<void**>(&d_Klo), work_elements * sizeof(double), "d_Klo");
        allocate_device(reinterpret_cast<void**>(&d_Khi), work_elements * sizeof(double), "d_Khi");
        allocate_device(reinterpret_cast<void**>(&d_vlo), work_elements * sizeof(double), "d_vlo");
        allocate_device(reinterpret_cast<void**>(&d_vhi), work_elements * sizeof(double), "d_vhi");
        allocate_device(reinterpret_cast<void**>(&d_ei), rows * sizeof(double), "d_ei");

        const int block = 128;
        const int grid = std::max(1, std::min(static_cast<int>((rows + block - 1) / block), prop.maxGridSize[0]));
        ei_hi_bounds_kernel<<<grid, block>>>(
            d_boxes_L, d_boxes_U, d_X, d_alpha, d_L, d_length,
            rows, input.n_train, input.d, input.sigma_f_2, input.y_min_scaled,
            d_Klo, d_Khi, d_vlo, d_vhi, d_ei);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(local_ei.data() + offset, d_ei, rows * sizeof(double), cudaMemcpyDeviceToHost));

        free_device(d_ei); free_device(d_vhi); free_device(d_vlo); free_device(d_Khi); free_device(d_Klo);
        free_device(d_boxes_U); free_device(d_boxes_L);
    }

    free_device(d_length); free_device(d_L); free_device(d_alpha); free_device(d_X);

    std::vector<double> all;
    std::uint64_t total = 0;
    gather_double_rows(local_ei, local_n, 1, rank, size, all, total);
    return all;
}

BestSample sample_best_global(
    const std::vector<double>& boxes_L,
    const std::vector<double>& boxes_U,
    const std::vector<unsigned char>& mask,
    const std::vector<double>& lhs,
    const PartitionInput& input,
    std::uint64_t device_batch_rows,
    int rank,
    int size,
    int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::uint64_t n_boxes = static_cast<std::uint64_t>(boxes_L.size()) / input.d;
    const std::uint64_t n_samples = 1ULL << input.d;
    const std::uint64_t start = local_start_for_rank(n_boxes, rank, size);
    const std::uint64_t local_n = local_count_for_rank(n_boxes, rank, size);
    const std::vector<double> local_L = slice_rows(boxes_L, start, local_n, input.d);
    const std::vector<double> local_U = slice_rows(boxes_U, start, local_n, input.d);
    std::vector<unsigned char> local_mask(local_n);
    std::copy_n(mask.data() + start, local_n, local_mask.data());

    std::vector<double> local_best_ei(local_n, -std::numeric_limits<double>::infinity());
    std::vector<double> local_best_points(checked_mul(local_n, input.d, "local best point elements"));

    // Constant GP/LHS arrays are uploaded once. Per-box work is allocated in
    // bounded batches and released immediately after each kernel finishes.
    double *d_X = nullptr, *d_alpha = nullptr, *d_L = nullptr, *d_length = nullptr, *d_lhs = nullptr;
    if (local_n != 0) {
        copy_to_device(reinterpret_cast<void**>(&d_X), input.X_train.data(), input.X_train.size() * sizeof(double), "d_X_train");
        copy_to_device(reinterpret_cast<void**>(&d_alpha), input.alpha.data(), input.alpha.size() * sizeof(double), "d_alpha");
        copy_to_device(reinterpret_cast<void**>(&d_L), input.L.data(), input.L.size() * sizeof(double), "d_cholesky_L");
        copy_to_device(reinterpret_cast<void**>(&d_length), input.length_scale.data(), input.length_scale.size() * sizeof(double), "d_length_scale");
        copy_to_device(reinterpret_cast<void**>(&d_lhs), lhs.data(), lhs.size() * sizeof(double), "d_lhs");
    }

    for (std::uint64_t offset = 0; offset < local_n; offset += device_batch_rows) {
        const std::uint64_t rows = std::min(device_batch_rows, local_n - offset);
        const std::size_t box_elements = checked_mul(rows, input.d, "sample batch box elements");
        const std::size_t work_elements = checked_mul(rows, input.n_train, "sample batch work elements");

        double *d_boxes_L = nullptr, *d_boxes_U = nullptr, *d_v = nullptr;
        double *d_best_ei = nullptr, *d_best_points = nullptr;
        unsigned char* d_mask = nullptr;
        copy_to_device(reinterpret_cast<void**>(&d_boxes_L), local_L.data() + offset * input.d,
                       box_elements * sizeof(double), "d_boxes_L");
        copy_to_device(reinterpret_cast<void**>(&d_boxes_U), local_U.data() + offset * input.d,
                       box_elements * sizeof(double), "d_boxes_U");
        copy_to_device(reinterpret_cast<void**>(&d_mask), local_mask.data() + offset,
                       rows * sizeof(unsigned char), "d_mask");
        allocate_device(reinterpret_cast<void**>(&d_v), work_elements * sizeof(double), "d_v");
        allocate_device(reinterpret_cast<void**>(&d_best_ei), rows * sizeof(double), "d_best_ei");
        allocate_device(reinterpret_cast<void**>(&d_best_points), box_elements * sizeof(double), "d_best_points");

        const int block = 128;
        const int grid = std::max(1, std::min(static_cast<int>((rows + block - 1) / block), prop.maxGridSize[0]));
        sample_best_kernel<<<grid, block>>>(
            d_boxes_L, d_boxes_U, d_mask, d_X, d_alpha, d_L, d_length, d_lhs,
            rows, input.n_train, input.d, n_samples, input.sigma_f_2, input.y_min_scaled,
            d_v, d_best_ei, d_best_points);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(local_best_ei.data() + offset, d_best_ei,
                              rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_best_points.data() + offset * input.d, d_best_points,
                              box_elements * sizeof(double), cudaMemcpyDeviceToHost));

        free_device(d_best_points); free_device(d_best_ei); free_device(d_v);
        free_device(d_mask); free_device(d_boxes_U); free_device(d_boxes_L);
    }

    free_device(d_lhs); free_device(d_length); free_device(d_L); free_device(d_alpha); free_device(d_X);

    BestSample local_best;
    local_best.x.assign(input.d, 0.0);
    for (std::uint64_t i = 0; i < local_n; ++i) {
        if (local_best_ei[i] > local_best.ei) {
            local_best.ei = local_best_ei[i];
            local_best.box_idx = static_cast<long long>(start + i);
            std::copy_n(local_best_points.data() + i * input.d, input.d, local_best.x.data());
        }
    }

    // MAXLOC chooses the lowest rank on an EI tie. Contiguous rank slices then
    // reproduce NumPy argmax's "first global box" tie behavior exactly.
    struct { double value; int rank; } local_candidate{local_best.ei, rank}, winner{};
    MPI_Allreduce(&local_candidate, &winner, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    if (rank != winner.rank) {
        local_best.box_idx = -1;
        std::fill(local_best.x.begin(), local_best.x.end(), 0.0);
    }
    MPI_Bcast(&local_best.box_idx, 1, MPI_LONG_LONG, winner.rank, MPI_COMM_WORLD);
    MPI_Bcast(local_best.x.data(), checked_int_count(input.d, "best point dimension"), MPI_DOUBLE,
              winner.rank, MPI_COMM_WORLD);
    local_best.ei = winner.value;
    return local_best;
}

void broadcast_mask(std::vector<unsigned char>& mask, int rank) {
    unsigned long long size_ull = rank == 0 ? static_cast<unsigned long long>(mask.size()) : 0ULL;
    MPI_Bcast(&size_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        mask.resize(static_cast<std::size_t>(size_ull));
    }
    if (size_ull != 0) {
        MPI_Bcast(mask.data(), checked_int_count(static_cast<std::uint64_t>(size_ull), "mask broadcast"), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
}

void split_selected_global(
    std::vector<double>& boxes_L,
    std::vector<double>& boxes_U,
    const std::vector<unsigned char>& target_mask,
    const PartitionInput& input,
    int rank,
    int size,
    int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::uint64_t n_boxes = static_cast<std::uint64_t>(boxes_L.size()) / input.d;
    std::uint64_t stride = 2 * input.d + 1;
    std::uint64_t start = local_start_for_rank(n_boxes, rank, size);
    std::uint64_t local_n = local_count_for_rank(n_boxes, rank, size);
    std::vector<double> local_L = slice_rows(boxes_L, start, local_n, input.d);
    std::vector<double> local_U = slice_rows(boxes_U, start, local_n, input.d);
    std::vector<std::uint64_t> selected_indices;
    for (std::uint64_t i = 0; i < local_n; ++i) {
        if (target_mask[start + i] != 0) {
            selected_indices.push_back(i);
        }
    }
    std::uint64_t local_out_rows = checked_mul(static_cast<std::uint64_t>(selected_indices.size()), stride, "split local rows");
    std::vector<double> local_out_L(checked_mul(local_out_rows, input.d, "split local output elements"));
    std::vector<double> local_out_U(local_out_L.size());

    if (!selected_indices.empty()) {
        double *d_boxes_L = nullptr, *d_boxes_U = nullptr, *d_domain_width = nullptr, *d_out_L = nullptr, *d_out_U = nullptr;
        std::uint64_t* d_selected = nullptr;
        std::vector<double> domain_width(input.d);
        for (std::uint64_t dim = 0; dim < input.d; ++dim) {
            domain_width[dim] = input.domain_U[dim] - input.domain_L[dim];
        }
        copy_to_device(reinterpret_cast<void**>(&d_boxes_L), local_L.data(), local_L.size() * sizeof(double), "d_boxes_L");
        copy_to_device(reinterpret_cast<void**>(&d_boxes_U), local_U.data(), local_U.size() * sizeof(double), "d_boxes_U");
        copy_to_device(reinterpret_cast<void**>(&d_domain_width), domain_width.data(), domain_width.size() * sizeof(double), "d_domain_width");
        copy_to_device(reinterpret_cast<void**>(&d_selected), selected_indices.data(), selected_indices.size() * sizeof(std::uint64_t), "d_selected");
        allocate_device(reinterpret_cast<void**>(&d_out_L), local_out_L.size() * sizeof(double), "d_out_L");
        allocate_device(reinterpret_cast<void**>(&d_out_U), local_out_U.size() * sizeof(double), "d_out_U");
        int block = 128;
        int grid = std::max(1, std::min(static_cast<int>((selected_indices.size() + block - 1) / block), prop.maxGridSize[0]));
        split_selected_boxes_kernel<<<grid, block>>>(
            d_boxes_L, d_boxes_U, d_domain_width, d_selected,
            static_cast<std::uint64_t>(selected_indices.size()), input.d, stride,
            d_out_L, d_out_U);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(local_out_L.data(), d_out_L, local_out_L.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_out_U.data(), d_out_U, local_out_U.size() * sizeof(double), cudaMemcpyDeviceToHost));
        free_device(d_out_U); free_device(d_out_L); free_device(d_selected); free_device(d_domain_width); free_device(d_boxes_U); free_device(d_boxes_L);
    }

    std::vector<double> gathered_L;
    std::vector<double> gathered_U;
    std::uint64_t total_rows_L = 0;
    std::uint64_t total_rows_U = 0;
    gather_double_rows(local_out_L, local_out_rows, input.d, rank, size, gathered_L, total_rows_L);
    gather_double_rows(local_out_U, local_out_rows, input.d, rank, size, gathered_U, total_rows_U);
    if (rank == 0) {
        if (total_rows_L != total_rows_U) {
            throw std::runtime_error("split lower/upper row counts do not match");
        }
        boxes_L.swap(gathered_L);
        boxes_U.swap(gathered_U);
    }
    broadcast_double_vector(boxes_L, rank);
    broadcast_double_vector(boxes_U, rank);
}

bool all_widths_less_than_epsilon(
    const std::vector<double>& boxes_L,
    const std::vector<double>& boxes_U,
    long long idx,
    const std::vector<double>& epsilon_x,
    std::uint64_t d) {
    if (idx < 0) {
        return false;
    }
    for (std::uint64_t dim = 0; dim < d; ++dim) {
        double width = boxes_U[static_cast<std::uint64_t>(idx) * d + dim] - boxes_L[static_cast<std::uint64_t>(idx) * d + dim];
        if (!(width < epsilon_x[dim])) {
            return false;
        }
    }
    return true;
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
        PartitionInput input = read_input(options.input_path);
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA devices visible");
        }
        int local_rank = local_rank_on_node();
        int device = local_rank % device_count;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (rank == 0) {
            std::cout << "exactbo_partitioning: mpi_size=" << size
                      << " device_count=" << device_count
                      << " input=\"" << options.input_path << "\""
                      << " output=\"" << options.output_path << "\""
                      << " n_train=" << input.n_train
                      << " d=" << input.d
                      << " max_partitions=" << input.max_partitions
                      << " device_batch_rows=" << options.device_batch_rows
                      << " epsilon_ei=" << input.epsilon_ei << "\n";
        }
        std::cout << "rank " << rank << " local_rank=" << local_rank
                  << " using gpu_device=" << device
                  << " gpu_name=\"" << prop.name << "\""
                  << " launch_cpu=" << current_cpu() << "\n";
        print_device_memory(rank, "startup");

        std::vector<double> boxes_L = input.domain_L;
        std::vector<double> boxes_U = input.domain_U;
        std::uint64_t n_samples = 1ULL << input.d;
        std::uint64_t stride = 2 * input.d + 1;
        std::vector<double> lhs = centered_latin_hypercube_unit(n_samples, input.d);
        BestSample best;
        best.x.assign(input.d, 0.0);
        std::uint64_t partitions_done = 0;
        int converged = 0;
        std::uint64_t preserved_start = 0;
        std::uint64_t preserved_count = 1;

        for (std::uint64_t partition = 0; partition < input.max_partitions; ++partition) {
            std::uint64_t n_boxes = static_cast<std::uint64_t>(boxes_L.size()) / input.d;
            std::vector<double> ei_hi = compute_ei_hi_global(
                boxes_L, boxes_U, input, options.device_batch_rows, rank, size, device);

            std::vector<unsigned char> analyze_mask;
            if (rank == 0) {
                double max_ei_hi = *std::max_element(ei_hi.begin(), ei_hi.end());
                analyze_mask.assign(n_boxes, 0);
                for (std::uint64_t i = 0; i < n_boxes; ++i) {
                    analyze_mask[i] = ei_hi[i] >= (max_ei_hi - input.epsilon_ei) ? 1 : 0;
                }
                for (std::uint64_t i = preserved_start; i < std::min(n_boxes, preserved_start + preserved_count); ++i) {
                    analyze_mask[i] = 1;
                }
                if (options.verbose) {
                    std::uint64_t n_analyze = 0;
                    for (auto value : analyze_mask) n_analyze += value != 0;
                    std::cout << "partition " << partition << " boxes=" << n_boxes
                              << " max_ei_hi=" << max_ei_hi
                              << " analyze=" << n_analyze << "\n";
                }
            }
            broadcast_mask(analyze_mask, rank);
            BestSample best_analyze = sample_best_global(
                boxes_L, boxes_U, analyze_mask, lhs, input, options.device_batch_rows, rank, size, device);
            best = best_analyze;

            int stop = 0;
            std::vector<unsigned char> active_mask;
            if (rank == 0) {
                active_mask.assign(n_boxes, 0);
                std::uint64_t n_active = 0;
                for (std::uint64_t i = 0; i < n_boxes; ++i) {
                    active_mask[i] = ei_hi[i] > (best_analyze.ei + input.epsilon_ei) ? 1 : 0;
                    n_active += active_mask[i] != 0;
                }
                if (n_active == 0 && all_widths_less_than_epsilon(boxes_L, boxes_U, best_analyze.box_idx, input.epsilon_x, input.d)) {
                    stop = 1;
                    converged = 1;
                }
                if (best_analyze.box_idx >= 0) {
                    active_mask[static_cast<std::uint64_t>(best_analyze.box_idx)] = 1;
                }
            }
            MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (stop) {
                partitions_done = partition + 1;
                break;
            }

            broadcast_mask(active_mask, rank);
            BestSample best_active = sample_best_global(
                boxes_L, boxes_U, active_mask, lhs, input, options.device_batch_rows, rank, size, device);
            best = best_active;

            std::vector<unsigned char> target_mask;
            if (rank == 0) {
                target_mask.assign(n_boxes, 0);
                std::uint64_t n_target = 0;
                for (std::uint64_t i = 0; i < n_boxes; ++i) {
                    target_mask[i] = ei_hi[i] > (best_active.ei + input.epsilon_ei) ? 1 : 0;
                    n_target += target_mask[i] != 0;
                }
                if (n_target == 0 && all_widths_less_than_epsilon(boxes_L, boxes_U, best_active.box_idx, input.epsilon_x, input.d)) {
                    stop = 1;
                    converged = 1;
                }
                if (best_active.box_idx >= 0) {
                    target_mask[static_cast<std::uint64_t>(best_active.box_idx)] = 1;
                }
                preserved_start = 0;
                for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(best_active.box_idx); ++i) {
                    preserved_start += target_mask[i] != 0;
                }
                preserved_start *= stride;
                preserved_count = stride;
                if (options.verbose) {
                    std::uint64_t n_target_after = 0;
                    for (auto value : target_mask) n_target_after += value != 0;
                    std::cout << "  best_ei=" << best_active.ei
                              << " best_box=" << best_active.box_idx
                              << " target=" << n_target_after << "\n";
                }
            }
            MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (stop) {
                partitions_done = partition + 1;
                break;
            }

            partitions_done = partition + 1;
            if (partition + 1 == input.max_partitions) {
                break;
            }

            broadcast_mask(target_mask, rank);
            split_selected_global(boxes_L, boxes_U, target_mask, input, rank, size, device);
            MPI_Bcast(&preserved_start, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(&preserved_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            std::uint64_t n_boxes_final = static_cast<std::uint64_t>(boxes_L.size()) / input.d;
            write_output(options.output_path, best, input.d, partitions_done, n_boxes_final, converged);
            std::cout << "rank 0 wrote exactbo_partitioning output to " << options.output_path
                      << " best_ei_scaled=" << best.ei
                      << " converged=" << converged
                      << " partitions_done=" << partitions_done
                      << " n_boxes_final=" << n_boxes_final << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " exactbo_partitioning: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    cudaDeviceReset();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
