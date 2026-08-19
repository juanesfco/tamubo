#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cstddef>
#include <cuda_runtime.h>

using std::cerr;
using std::cout;
using std::fixed;
using std::setprecision;
using std::setw;
using std::string;
using std::ifstream;
using std::runtime_error;
using std::ios;
using std::uint64_t;
using std::numeric_limits;
using std::size_t;

constexpr char kInputMagic[8] = {'T', 'P', 'A', 'R', 'I', 'N', '1', '!'};

constexpr int BOXES = 100000000;
constexpr int THREADS_PER_BLOCK = 256;
constexpr double SQRT_2 = 1.4142135623730951;
constexpr double INV_SQRT_2_PI = 0.3989422804014327;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t error = call;                                           \
        if (error != cudaSuccess) {                                         \
            cerr << cudaGetErrorString(error) << "\n";                     \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

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
    double* epsilon_x = nullptr;
    double* domain_L = nullptr;
    double* domain_U = nullptr;
    double* X_train = nullptr;
    double* alpha = nullptr;
    double* L = nullptr;
    double* length_scale = nullptr;
};

struct Interval {
    double low;
    double high;
};

struct Results {
    double* low = nullptr;
    double* high = nullptr;
    double* center_ei = nullptr;
    double* upper_ei = nullptr;
};

PartitionInput* read_input(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) {
        throw runtime_error("failed to open input file: " + path);
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || memcmp(magic, kInputMagic, sizeof(magic)) != 0) {
        throw runtime_error("invalid exactbo_partitioning input magic");
    }

    PartitionInput* input = nullptr;
    CUDA_CHECK(cudaMallocManaged(&input, sizeof(PartitionInput)));
    // Binary streams read bytes through char pointers.  These casts are required
    // by the C++ stream API; they do not change the stored values.
    in.read(reinterpret_cast<char*>(&input->n_train), sizeof(input->n_train));
    in.read(reinterpret_cast<char*>(&input->d), sizeof(input->d));
    in.read(reinterpret_cast<char*>(&input->max_partitions), sizeof(input->max_partitions));
    in.read(reinterpret_cast<char*>(&input->epsilon_ei), sizeof(input->epsilon_ei));
    in.read(reinterpret_cast<char*>(&input->sigma_f_2), sizeof(input->sigma_f_2));
    in.read(reinterpret_cast<char*>(&input->sigma_n_2), sizeof(input->sigma_n_2));
    in.read(reinterpret_cast<char*>(&input->y_train_mean), sizeof(input->y_train_mean));
    in.read(reinterpret_cast<char*>(&input->y_train_std), sizeof(input->y_train_std));
    in.read(reinterpret_cast<char*>(&input->y_min_scaled), sizeof(input->y_min_scaled));
    
    if (!in) {
        throw runtime_error("input file ended while reading GP settings");
    }

    CUDA_CHECK(cudaMallocManaged(&input->epsilon_x, input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&input->domain_L, input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&input->domain_U, input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&input->X_train, input->n_train * input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&input->alpha, input->n_train * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&input->L, input->n_train * input->n_train * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&input->length_scale, input->d * sizeof(double)));

    in.read(reinterpret_cast<char*>(input->epsilon_x), input->d * sizeof(double));
    in.read(reinterpret_cast<char*>(input->domain_L), input->d * sizeof(double));
    in.read(reinterpret_cast<char*>(input->domain_U), input->d * sizeof(double));
    in.read(reinterpret_cast<char*>(input->X_train), input->n_train * input->d * sizeof(double));
    in.read(reinterpret_cast<char*>(input->alpha), input->n_train * sizeof(double));
    in.read(reinterpret_cast<char*>(input->L), input->n_train * input->n_train * sizeof(double));
    in.read(reinterpret_cast<char*>(input->length_scale), input->d * sizeof(double));
    if (!in) {
        throw runtime_error("input file ended while reading GP arrays");
    }
    return input;
}

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

__device__ void ei_at_point(const double* low, const double* high, double* center_ei, const int d, const int n, const double* x_train, const double* alpha, const double* L, const double* length_scale) {
    
}

__device__ void bound_box(const double* low, const double* high, double* upper_ei, const int d, const int n, const double* x_train, const double* alpha, const double* L, const double* length_scale) {
    
}

__global__ void evaluate_boxes(Results* results, PartitionInput* input) {
    const int box = blockIdx.x * blockDim.x + threadIdx.x;
    if (box >= BOXES) {
        return;
    }

    const double* low = results->low + box * input->d;
    const double* high = results->high + box * input->d;
    double* center_ei = results->center_ei + box;
    double* upper_ei = results->upper_ei + box;

    ei_at_point(low, high, center_ei, input->d, input->n_train, input->X_train, input->alpha, input->L, input->length_scale);
    bound_box(low, high, upper_ei, input->d, input->n_train, input->X_train, input->alpha, input->L, input->length_scale);
}

void print_input(const PartitionInput& input, const string& input_path) {
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
}

int main() {

    const string input_path = "data/logs/checkBounding/input.bin";
    PartitionInput* input = read_input(input_path);

    print_input(*input, input_path);

    Results* results = nullptr;
    CUDA_CHECK(cudaMallocManaged(&results, sizeof(Results)));
    CUDA_CHECK(cudaMallocManaged(&results->low, BOXES * input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&results->high, BOXES * input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&results->center_ei, BOXES * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&results->upper_ei, BOXES * sizeof(double)));

    const double center[input->d] = {0.5, 0.4};
    double half_width[input->d] = {0.3, 0.3};

    // Create nested boxes with a common center.
    for (int box = 0; box < BOXES; ++box) {
        for (int dim = 0; dim < input->d; ++dim) {
            const int index = box * input->d + dim;
            results->low[index] = center[dim] - half_width[dim];
            results->high[index] = center[dim] + half_width[dim];
            half_width[dim] /= 1.0000001;
        }
    }

    const int blocks = (BOXES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warm up the GPU to avoid measuring kernel launch overhead.
    evaluate_boxes<<<blocks, THREADS_PER_BLOCK>>>(results, input);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profile representative kernel launches.
    for (int i = 0; i < 10; ++i) {
        evaluate_boxes<<<blocks, THREADS_PER_BLOCK>>>(results, input);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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
                 << setw(10) << results->low[box * input->d + 0] << "  "
                 << setw(10) << results->high[box * input->d + 0] << "  "
                 << setw(10) << results->low[box * input->d + 1] << "  "
                 << setw(10) << results->high[box * input->d + 1] << "  "
                 << setw(10) << results->center_ei[box] << "  "
                 << setw(10) << results->upper_ei[box] << "  "
                 << setw(10) << gap << "\n";
        }

        if (box > BOXES - 5) {
            cout << setw(9) << box << "  "
                 << setw(10) << results->low[box * input->d + 0] << "  "
                 << setw(10) << results->high[box * input->d + 0] << "  "
                 << setw(10) << results->low[box * input->d + 1] << "  "
                 << setw(10) << results->high[box * input->d + 1] << "  "
                 << setw(10) << results->center_ei[box] << "  "
                 << setw(10) << results->upper_ei[box] << "  "
                 << setw(10) << gap << "\n";
        }

        if (box == 0) {
            first_gap = gap;
        }
        passed &= results->center_ei[box] <= results->upper_ei[box];
        passed &= results->upper_ei[box] <= previous_upper;
        passed &= gap <= previous_gap;
        previous_upper = results->upper_ei[box];
        previous_gap = gap;
    }

    passed &= previous_gap < first_gap;
    cout << "Evaluated " << BOXES << " boxes with " << blocks
         << " blocks of " << THREADS_PER_BLOCK << " threads.\n";
    cout << "Result: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaFree(input->epsilon_x));
    CUDA_CHECK(cudaFree(input->domain_L));
    CUDA_CHECK(cudaFree(input->domain_U));
    CUDA_CHECK(cudaFree(input->X_train));
    CUDA_CHECK(cudaFree(input->alpha));
    CUDA_CHECK(cudaFree(input->L));
    CUDA_CHECK(cudaFree(input->length_scale));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(results->low));
    CUDA_CHECK(cudaFree(results->high));
    CUDA_CHECK(cudaFree(results->center_ei));
    CUDA_CHECK(cudaFree(results->upper_ei));
    CUDA_CHECK(cudaFree(results));
    return passed ? 0 : 1;
}
