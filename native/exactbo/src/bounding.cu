#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>

using std::cerr;
using std::cout;
using std::fixed;
using std::setprecision;
using std::setw;
using std::vector;
using std::string;
using std::ifstream;
using std::runtime_error;
using std::ios;
using std::uint64_t;
using std::numeric_limits;

constexpr char kInputMagic[8] = {'T', 'P', 'A', 'R', 'I', 'N', '1', '!'};

constexpr int D = 2;
constexpr int N = 4;
constexpr int BOXES = 100000000;
constexpr int THREADS_PER_BLOCK = 256;
constexpr double TOL = 0.0;
constexpr double SQRT_2 = 1.4142135623730951;
constexpr double INV_SQRT_2_PI = 0.3989422804014327;

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

struct Interval {
    double low;
    double high;
};

struct Results {
    double low[BOXES * D];
    double high[BOXES * D];
    double center_ei[BOXES];
    double upper_ei[BOXES];
};

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
    // Binary streams read bytes through char pointers.  These casts are required
    // by the C++ stream API; they do not change the stored values.
    in.read(reinterpret_cast<char*>(&input.n_train), sizeof(input.n_train));
    in.read(reinterpret_cast<char*>(&input.d), sizeof(input.d));
    in.read(reinterpret_cast<char*>(&input.max_partitions), sizeof(input.max_partitions));
    in.read(reinterpret_cast<char*>(&input.epsilon_ei), sizeof(input.epsilon_ei));
    in.read(reinterpret_cast<char*>(&input.sigma_f_2), sizeof(input.sigma_f_2));
    in.read(reinterpret_cast<char*>(&input.sigma_n_2), sizeof(input.sigma_n_2));
    in.read(reinterpret_cast<char*>(&input.y_train_mean), sizeof(input.y_train_mean));
    in.read(reinterpret_cast<char*>(&input.y_train_std), sizeof(input.y_train_std));
    in.read(reinterpret_cast<char*>(&input.y_min_scaled), sizeof(input.y_min_scaled));
    
    if (!in) {
        throw runtime_error("input file ended while reading GP settings");
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
    return input;
}

// Fixed GP used only by this test.
__device__ __constant__ double X_TRAIN[N * D] = {
    0.25, 0.25, 
    0.25, 0.75, 
    0.75, 0.25, 
    0.75, 0.75
};
__device__ __constant__ double ALPHA[N] = {
    1.2241744200151572, 
    -0.39566515487285686, 
    -1.4146724317268353, 
    0.5861631665845355
};
__device__ __constant__ double CHOLESKY[N * N] = {
    0.9999975546562031, 0.0, 0.0, 0.0,
    6.874197171131952e-11, 0.9999975546562031, 0.0, 0.0,
    0.05, 0.02, 1.0, 0.0,
    0.01, 0.03, 0.04, 1.0,
};
__device__ __constant__ double LENGTH_SCALE[D] = {0.4, 0.6};

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t error = call;                                           \
        if (error != cudaSuccess) {                                         \
            cerr << cudaGetErrorString(error) << "\n";                     \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__device__ double normal_cdf(double z) {
    return 0.5 * (1.0 + erf(z / SQRT_2));
}

__device__ double normal_pdf(double z) {
    return INV_SQRT_2_PI * exp(-0.5 * z * z);
}

__device__ Interval multiply(Interval a, Interval b) {
    const double p1 = a.low * b.low;
    const double p2 = a.low * b.high;
    const double p3 = a.high * b.low;
    const double p4 = a.high * b.high;
    return {
        fmin(fmin(p1, p2), fmin(p3, p4)),
        fmax(fmax(p1, p2), fmax(p3, p4)),
    };
}

__device__ double expected_improvement(double mean, double sigma) {
    const double best_y = -0.4;
    sigma = fmax(sigma, 1.0e-12);
    const double improvement = best_y - mean;
    const double z = improvement / sigma;
    return improvement * normal_cdf(z) + sigma * normal_pdf(z);
}

__device__ double upper_expected_improvement(Interval mean, Interval sigma) {
    const double best_y = -0.4;
    const Interval improvement = {best_y - mean.high, best_y - mean.low};
    const Interval inverse_sigma = {1.0 / sigma.high, 1.0 / sigma.low};
    const Interval z = multiply(improvement, inverse_sigma);
    const Interval cdf = {normal_cdf(z.low), normal_cdf(z.high)};

    const double pdf_low_end = normal_pdf(z.low);
    const double pdf_high_end = normal_pdf(z.high);
    const Interval pdf = {
        fmin(pdf_low_end, pdf_high_end),
        z.low <= 0.0 && z.high >= 0.0
            ? INV_SQRT_2_PI
            : fmax(pdf_low_end, pdf_high_end),
    };

    return fmax(
        multiply(improvement, cdf).high + multiply(sigma, pdf).high,
        0.0);
}

__device__ void gp_at_point(const double* x, double& mean, double& sigma) {
    mean = 0.0;
    double squared_norm = 0.0;
    double solve[N];

    for (int row = 0; row < N; ++row) {
        double squared_distance = 0.0;
        for (int dim = 0; dim < D; ++dim) {
            const double difference =
                (x[dim] - X_TRAIN[row * D + dim]) / LENGTH_SCALE[dim];
            squared_distance += difference * difference;
        }

        const double kernel = exp(-0.5 * squared_distance);
        mean += kernel * ALPHA[row];

        double previous = 0.0;
        for (int column = 0; column < row; ++column) {
            previous += CHOLESKY[row * N + column] * solve[column];
        }
        solve[row] = (kernel - previous) / CHOLESKY[row * N + row];
        squared_norm += solve[row] * solve[row];
    }

    sigma = sqrt(fmax(1.0 - squared_norm, 1.0e-12));
}

__device__ double bound_box(const double* low, const double* high) {
    double kernel_low[N];
    double kernel_high[N];
    Interval mean = {0.0, 0.0};

    // RBF-kernel and posterior-mean bounds.
    for (int point = 0; point < N; ++point) {
        double nearest_squared = 0.0;
        double farthest_squared = 0.0;

        for (int dim = 0; dim < D; ++dim) {
            const double xi = X_TRAIN[point * D + dim];
            const double from_low = (low[dim] - xi) / LENGTH_SCALE[dim];
            const double from_high = (xi - high[dim]) / LENGTH_SCALE[dim];
            const double nearest = fmax(fmax(from_low, from_high), 0.0);
            const double farthest = fmax(fabs(from_low), fabs(from_high));
            nearest_squared += nearest * nearest;
            farthest_squared += farthest * farthest;
        }

        kernel_low[point] = exp(-0.5 * farthest_squared);
        kernel_high[point] = exp(-0.5 * nearest_squared);

        if (ALPHA[point] >= 0.0) {
            mean.low += kernel_low[point] * ALPHA[point];
            mean.high += kernel_high[point] * ALPHA[point];
        } else {
            mean.low += kernel_high[point] * ALPHA[point];
            mean.high += kernel_low[point] * ALPHA[point];
        }
    }

    // Cholesky-solve and posterior-sigma bounds.
    double solve_low[N];
    double solve_high[N];
    double norm_low = 0.0;
    double norm_high = 0.0;

    for (int row = 0; row < N; ++row) {
        double sum_low = 0.0;
        double sum_high = 0.0;

        for (int column = 0; column < row; ++column) {
            const double value = CHOLESKY[row * N + column];
            sum_low += value >= 0.0
                ? value * solve_low[column] : value * solve_high[column];
            sum_high += value >= 0.0
                ? value * solve_high[column] : value * solve_low[column];
        }

        const double diagonal = CHOLESKY[row * N + row];
        solve_low[row] = (kernel_low[row] - sum_high) / diagonal;
        solve_high[row] = (kernel_high[row] - sum_low) / diagonal;

        const double square_low = solve_low[row] * solve_low[row];
        const double square_high = solve_high[row] * solve_high[row];
        norm_high += fmax(square_low, square_high);
        norm_low += solve_low[row] <= 0.0 && solve_high[row] >= 0.0
            ? 0.0 : fmin(square_low, square_high);
    }

    const Interval sigma = {
        sqrt(fmax(1.0 - norm_high, 1.0e-12)),
        sqrt(fmax(1.0 - norm_low, 1.0e-12)),
    };
    return upper_expected_improvement(mean, sigma);
}

__global__ void evaluate_boxes(Results* results) {
    const int box = blockIdx.x * blockDim.x + threadIdx.x;
    if (box >= BOXES) {
        return;
    }

    const double* low = results->low + box * D;
    const double* high = results->high + box * D;
    const double center[D] = {
        0.5 * (low[0] + high[0]),
        0.5 * (low[1] + high[1]),
    };

    double mean = 0.0;
    double sigma = 0.0;
    gp_at_point(center, mean, sigma);
    results->center_ei[box] = expected_improvement(mean, sigma);
    results->upper_ei[box] = bound_box(low, high);
}

int main() {

    const string input_path = "data/logs/checkBounding/input.bin";
    PartitionInput input = read_input(input_path);

    cout << "Read input from " << input_path << "\n"
         << " n_train: " << input.n_train << "\n"
         << " d: " << input.d << "\n"
         << " max_partitions: " << input.max_partitions << "\n"
         << " epsilon_ei: " << input.epsilon_ei << "\n"
         << " sigma_f_2: " << input.sigma_f_2 << "\n"
         << " sigma_n_2: " << input.sigma_n_2 << "\n"
         << " y_train_mean: " << input.y_train_mean << "\n"
         << " y_train_std: " << input.y_train_std << "\n"
         << " y_min_scaled: " << input.y_min_scaled << "\n"
         << " epsilon_x: ";
            for (int dim = 0; dim < input.d; ++dim) {
                cout << input.epsilon_x[dim] << " ";
            }
            cout << "\n"
        << " domain_L: ";
            for (int dim = 0; dim < input.d; ++dim) {
                cout << input.domain_L[dim] << " ";
            }
            cout << "\n"
        << " domain_U: ";
            for (int dim = 0; dim < input.d; ++dim) {
                cout << input.domain_U[dim] << " ";
            }
            cout << "\n"
        << " X_train: ";
            for (int i = 0; i < input.n_train; ++i) {
                if (i == 0) {
                    for (int dim = 0; dim < input.d; ++dim) {
                        cout << input.X_train[i * input.d + dim] << " ";
                    }
                } else {
                    cout << "          ";
                    for (int dim = 0; dim < input.d; ++dim) {
                        cout << input.X_train[i * input.d + dim] << " ";
                    }
                }
                if (i < input.n_train - 1) {
                    cout << "\n";
                }
            }
            cout << "\n"
        << " alpha: ";
            for (int i = 0; i < input.n_train; ++i) {
                cout << input.alpha[i] << " ";
            }
            cout << "\n"
        << " L: ";
            for (int i = 0; i < input.n_train; ++i) {
                if (i == 0) {
                    for (int j = 0; j < input.n_train; ++j) {
                        cout << input.L[i * input.n_train + j] << " ";
                    }
                } else {
                    cout << "    ";
                    for (int j = 0; j < input.n_train; ++j) {
                        cout << input.L[i * input.n_train + j] << " ";
                    }
                }
                if (i < input.n_train - 1) {
                    cout << "\n";
                }
            }
            cout << "\n"
        << " length_scale: ";
            for (int dim = 0; dim < input.d; ++dim) {
                cout << input.length_scale[dim] << " ";
            }
            cout << "\n";

    Results* results = nullptr;
    CUDA_CHECK(cudaMallocManaged(&results, sizeof(Results)));

    const double center[D] = {0.5, 0.4};
    double half_width[D] = {0.3, 0.3};

    // Create nested boxes with a common center.
    for (int box = 0; box < BOXES; ++box) {
        for (int dim = 0; dim < D; ++dim) {
            const int index = box * D + dim;
            results->low[index] = center[dim] - half_width[dim];
            results->high[index] = center[dim] + half_width[dim];
            half_width[dim] /= 1.0000001;
        }
    }

    const int blocks = (BOXES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warm up the GPU to avoid measuring kernel launch overhead.
    // evaluate_boxes<<<blocks, THREADS_PER_BLOCK>>>(results);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // Profile representative kernel launches.
    // for (int i = 0; i < 10; ++i) {
    //     evaluate_boxes<<<blocks, THREADS_PER_BLOCK>>>(results);
    //    CUDA_CHECK(cudaGetLastError());
    //    CUDA_CHECK(cudaDeviceSynchronize());
    // }

    cout << fixed << setprecision(8);
    cout << "box        low.x       high.x      low.y       high.y      EI(c)       EI_hi       gap\n";

    bool passed = true;
    double first_gap = 0.0;
    double previous_upper = INFINITY;
    double previous_gap = INFINITY;

    for (int box = 0; box < BOXES; ++box) {
        const double gap = results->upper_ei[box] - results->center_ei[box];

        if (box < 8) {
            cout << setw(9) << box << "  "
                 << setw(10) << results->low[box * D + 0] << "  "
                 << setw(10) << results->high[box * D + 0] << "  "
                 << setw(10) << results->low[box * D + 1] << "  "
                 << setw(10) << results->high[box * D + 1] << "  "
                 << setw(10) << results->center_ei[box] << "  "
                 << setw(10) << results->upper_ei[box] << "  "
                 << setw(10) << gap << "\n";
        }

        if (box > BOXES - 5) {
            cout << setw(9) << box << "  "
                 << setw(10) << results->low[box * D + 0] << "  "
                 << setw(10) << results->high[box * D + 0] << "  "
                 << setw(10) << results->low[box * D + 1] << "  "
                 << setw(10) << results->high[box * D + 1] << "  "
                 << setw(10) << results->center_ei[box] << "  "
                 << setw(10) << results->upper_ei[box] << "  "
                 << setw(10) << gap << "\n";
        }

        if (box == 0) {
            first_gap = gap;
        }
        passed &= results->center_ei[box] <= results->upper_ei[box] + TOL;
        passed &= results->upper_ei[box] <= previous_upper + TOL;
        passed &= gap <= previous_gap + TOL;
        previous_upper = results->upper_ei[box];
        previous_gap = gap;
    }

    passed &= previous_gap < first_gap;
    cout << "Evaluated " << BOXES << " boxes with " << blocks
         << " blocks of " << THREADS_PER_BLOCK << " threads.\n";
    cout << "Result: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(results));
    return passed ? 0 : 1;
}
