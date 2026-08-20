#include <cmath>
using std::exp;
using std::sqrt;
using std::erf;
using std::fabs;
using std::fmin;
using std::fmax;
using std::log;

#include <cstdlib>
using std::exit;
using std::size_t;

#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;

#include <string>
using std::string;

#include <fstream>
using std::ifstream;
using std::ios;

#include <stdexcept>
using std::runtime_error;

#include <cuda_runtime.h>

constexpr char kInputMagic[8] = {'T', 'P', 'A', 'R', 'I', 'N', '1', '!'};

constexpr int PROFILE_REPS = 0;
constexpr double BOX_WIDTH_DICREASE_FACTOR = 1.0000003;
constexpr size_t BOXES = 100000000;
constexpr int THREADS_PER_BLOCK = 128;
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
    double y_min = 0.0;
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
    double* center_logei = nullptr;
    double* upper_logei = nullptr;
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
    in.read(reinterpret_cast<char*>(&input->y_min), sizeof(input->y_min));
    
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
         << " y_min: " << input.y_min << "\n"
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

__device__ void logei_at_point(double* center_logei, const double* low, const double* high, const int d, const int n, const double* x_train, const double* length_scale, const double sigma_f_2, const double sigma_n_2, const double* alpha, const double y_train_mean, const double y_train_std, const double* L, const double y_min, double* workspace) {
    // Calculate the kernel values between the center of the box and each training point.
    for (int i = 0; i < n; ++i) {
        double squared_distance = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            const double difference = ((low[dim] + high[dim]) / 2.0 - x_train[i * d + dim]) / length_scale[dim];
            squared_distance += difference * difference;
        }
        workspace[i] = sigma_f_2 * exp(-0.5 * squared_distance);
    }

    // Calculate and unnormalize the mean.
    double mean = 0.0;
    for (int i = 0; i < n; ++i) { 
        mean += alpha[i] * workspace[i];
    }
    mean = y_train_mean + y_train_std * mean;
    
    // Left divide L by workspace to solve for v in L * v = workspace, store in workspace
    for (int i = 0; i < n; ++i) {
        workspace[i] /= L[i * n + i];
        for (int j = i + 1; j < n; ++j) {
            workspace[j] -= L[j * n + i] * workspace[i];
        }
    }

    // Calculate the normalized variance.
    double variance = sigma_f_2 + sigma_n_2;
    for (int i = 0; i < n; ++i) {
        variance -= workspace[i] * workspace[i];
    }

    // If the variance is negative due to numerical issues, set EI to zero.
    if (variance < 0.0) {
        *center_logei = -INFINITY;
    } else {
        const double standard_deviation = sqrt(variance * y_train_std * y_train_std);
        const double improvement = y_min - mean;
        const double z = improvement / standard_deviation;
        *center_logei = log(improvement * normal_cdf(z) + standard_deviation * normal_pdf(z));
    }
}

__device__ void bound_box(double* upper_logei, const double* low, const double* high, const int d, const int n, const double* x_train, const double* length_scale, const double sigma_f_2, const double sigma_n_2, const double* alpha, const double y_train_mean, const double y_train_std, const double* L, const double y_min, double* workspace_low, double* workspace_high) {
    // Bound the normalized mean over the box.
    double mean_low = 0.0;
    double mean_high = 0.0;
    for (int i = 0; i < n; ++i) {
        double min_distance_2 = 0.0;
        double max_distance_2 = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            const double distance_low = (low[dim] - x_train[i * d + dim]) / length_scale[dim];
            const double distance_high = (x_train[i * d + dim] - high[dim]) / length_scale[dim];
            const double min_distance = fmax(fmax(distance_low, distance_high), 0.0);
            const double max_distance = fmax(fabs(distance_low), fabs(distance_high));
            min_distance_2 += min_distance * min_distance;
            max_distance_2 += max_distance * max_distance;
        }

        const double kernel_low = sigma_f_2 * exp(-0.5 * max_distance_2);
        const double kernel_high = sigma_f_2 * exp(-0.5 * min_distance_2);
        workspace_low[i] = kernel_low;
        workspace_high[i] = kernel_high;
        if (alpha[i] >= 0.0) {
            mean_low += alpha[i] * kernel_low;
            mean_high += alpha[i] * kernel_high;
        } else {
            mean_low += alpha[i] * kernel_high;
            mean_high += alpha[i] * kernel_low;
        }
    }

    // Undo the mean normalization.
    mean_low = y_train_mean + y_train_std * mean_low;
    mean_high = y_train_mean + y_train_std * mean_high;

    // Bound v = L^-1 k, overwriting the kernel bounds with the v bounds.
    double q_low = 0.0;
    double q_high = 0.0;
    for (int row = 0; row < n; ++row) {
        double sum_low = 0.0;
        double sum_high = 0.0;
        for (int i = 0; i < row; ++i) {
            const double coefficient = L[row * n + i];
            if (coefficient >= 0.0) {
                sum_low += coefficient * workspace_low[i];
                sum_high += coefficient * workspace_high[i];
            } else {
                sum_low += coefficient * workspace_high[i];
                sum_high += coefficient * workspace_low[i];
            }
        }

        const double diagonal = L[row * n + row];
        const double v_low = (workspace_low[row] - sum_high) / diagonal;
        const double v_high = (workspace_high[row] - sum_low) / diagonal;
        workspace_low[row] = v_low;
        workspace_high[row] = v_high;

        const double square_1 = v_low * v_low;
        const double square_2 = v_high * v_high;
        q_high += fmax(square_1, square_2);
        if (v_low <= 0.0 && v_high >= 0.0) {
            q_low += 0.0;
        } else {
            q_low += fmin(square_1, square_2);
        }
    }

    // Bound and unnormalize the posterior standard deviation.
    const double variance_base = sigma_f_2 + sigma_n_2;
    Interval standard_deviation = {
        y_train_std * sqrt(fmax(variance_base - q_high, 1.0e-12)),
        y_train_std * sqrt(fmax(variance_base - q_low, 1.0e-12)),
    };

    // Interval extension of EI = (y_min - mean) Phi(z) + sigma phi(z).
    const Interval improvement = {y_min - mean_high, y_min - mean_low};
    const Interval inverse_standard_deviation = {
        standard_deviation.high != 0.0 ? 1.0 / standard_deviation.high : 1.0e12,
        standard_deviation.low != 0.0 ? 1.0 / standard_deviation.low : 1.0e12,
    };
    const Interval z = multiply(improvement, inverse_standard_deviation);
    const Interval cdf = {normal_cdf(z.low), normal_cdf(z.high)};

    Interval pdf = {
        fmin(normal_pdf(z.low), normal_pdf(z.high)),
        fmax(normal_pdf(z.low), normal_pdf(z.high)),
    };
    if (z.low <= 0.0 && z.high >= 0.0) {
        pdf.high = INV_SQRT_2_PI;
    }

    const Interval first_term = multiply(improvement, cdf);
    const Interval second_term = multiply(standard_deviation, pdf);
    *upper_logei = log(fmax(first_term.high + second_term.high, 0.0));
    if (standard_deviation.high == 0.0) {
        *upper_logei = -INFINITY;
    }
}

__global__ void evaluate_boxes(Results* results, PartitionInput* input) {
    extern __shared__ double shared_workspace[];

    const size_t box = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (box >= BOXES) {
        return;
    }

    const double* low = results->low + box * input->d;
    const double* high = results->high + box * input->d;
    double* center_logei = results->center_logei + box;
    double* upper_logei = results->upper_logei + box;

    const size_t values_per_thread = 2 * static_cast<size_t>(input->n_train) + 1;
    double* workspace_low_for_box = shared_workspace + threadIdx.x * values_per_thread;
    double* workspace_high_for_box = workspace_low_for_box + input->n_train;

    logei_at_point(center_logei, low, high, input->d, input->n_train, input->X_train, input->length_scale, input->sigma_f_2, input->sigma_n_2, input->alpha, input->y_train_mean, input->y_train_std, input->L, input->y_min, workspace_low_for_box);
    bound_box(upper_logei, low, high, input->d, input->n_train, input->X_train, input->length_scale, input->sigma_f_2, input->sigma_n_2, input->alpha, input->y_train_mean, input->y_train_std, input->L, input->y_min, workspace_low_for_box, workspace_high_for_box);
}

int main() {
    // Read input from binary file and allocate memory on device for the input data.
    const string input_path = "data/logs/checkBounding/input5d.bin";
    PartitionInput* input = read_input(input_path);

    // Print the input data to the console for verification.
    print_input(*input, input_path);

    // Allocate memory on the device for the results of the box evaluations.
    Results* results = nullptr;
    CUDA_CHECK(cudaMallocManaged(&results, sizeof(Results)));
    CUDA_CHECK(cudaMallocManaged(&results->low, BOXES * input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&results->high, BOXES * input->d * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&results->center_logei, BOXES * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&results->upper_logei, BOXES * sizeof(double)));

    // Define the center and half-width of the boxes to be evaluated.
    double center[input->d];
    double half_width[input->d];

    if (input->d == 2) {
        center[0] = 0.5;
        center[1] = 0.4;
        half_width[0] = 0.3;
        half_width[1] = 0.3;
    } else if (input->d == 5) {
        center[0] = 0.8;
        center[1] = 0.6;
        center[2] = 0.4;
        center[3] = 0.2;
        center[4] = 0.1;
        half_width[0] = 0.2;
        half_width[1] = 0.3;
        half_width[2] = 0.5;
        half_width[3] = 0.3;
        half_width[4] = 0.1;
    } else if (input->d == 10) {
        center[0] = 0.9;
        center[1] = 0.8;
        center[2] = 0.7;
        center[3] = 0.6;
        center[4] = 0.5;
        center[5] = 0.5;
        center[6] = 0.4;
        center[7] = 0.3;
        center[8] = 0.2;
        center[9] = 0.1;
        half_width[0] = 0.1;
        half_width[1] = 0.3;
        half_width[2] = 0.4;
        half_width[3] = 0.3;
        half_width[4] = 0.6;
        half_width[5] = 0.3;
        half_width[6] = 0.3;
        half_width[7] = 0.3;
        half_width[8] = 0.2;
        half_width[9] = 0.1;
    } else {
        cerr << "Unsupported dimension: " << input->d << "\n";
        return 1;
    }
    

    // Create nested boxes with a common center.
    for (size_t box = 0; box < BOXES; ++box) {
        for (int dim = 0; dim < input->d; ++dim) {
            const size_t index = box * input->d + dim;
            results->low[index] = center[dim] - half_width[dim];
            results->high[index] = center[dim] + half_width[dim];
            half_width[dim] /= BOX_WIDTH_DICREASE_FACTOR;
        }
    }

    // Calculate the number of blocks needed to evaluate all boxes with the specified number of threads per block.
    const unsigned int blocks = static_cast<unsigned int>((BOXES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const size_t shared_memory_bytes = THREADS_PER_BLOCK * (2 * static_cast<size_t>(input->n_train) + 1) * sizeof(double);

    // Warm up the GPU to avoid measuring kernel launch overhead.
    evaluate_boxes<<<blocks, THREADS_PER_BLOCK, shared_memory_bytes>>>(results, input);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profile representative kernel launches.
    for (int i = 0; i < PROFILE_REPS; ++i) {
        evaluate_boxes<<<blocks, THREADS_PER_BLOCK, shared_memory_bytes>>>(results, input);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Print the results of the box evaluations to the console for verification.
    cout << fixed << setprecision(12);
    cout << "box            low.x_1         high.x_1        low.x_d         high.x_d        logEI(c)        logEI_hi        gap\n";

    bool passed = true;
    double first_gap = 0.0;
    double previous_upper = INFINITY;
    double previous_gap = INFINITY;

    for (size_t box = 0; box < BOXES; ++box) {
        const double gap = results->upper_logei[box] - results->center_logei[box];

        if (box < 8) {
            cout << setw(9) << box << "  "
                 << setw(10) << results->low[box * input->d + 0] << "  "
                 << setw(10) << results->high[box * input->d + 0] << "  "
                 << setw(10) << results->low[box * input->d + (input->d - 1)] << "  "
                 << setw(10) << results->high[box * input->d + (input->d - 1)] << "  "
                 << setw(10) << results->center_logei[box] << "  "
                 << setw(10) << results->upper_logei[box] << "  "
                 << setw(10) << gap << "\n";
        }

        if (box > BOXES - 5) {
            cout << setw(9) << box << "  "
                 << setw(10) << results->low[box * input->d + 0] << "  "
                 << setw(10) << results->high[box * input->d + 0] << "  "
                 << setw(10) << results->low[box * input->d + (input->d - 1)] << "  "
                 << setw(10) << results->high[box * input->d + (input->d - 1)] << "  "
                 << setw(10) << results->center_logei[box] << "  "
                 << setw(10) << results->upper_logei[box] << "  "
                 << setw(10) << gap << "\n";
        }

        if (box == 0) {
            first_gap = gap;
        }
        passed &= results->center_logei[box] <= results->upper_logei[box];
        passed &= results->upper_logei[box] <= previous_upper;
        passed &= gap <= previous_gap;
        previous_upper = results->upper_logei[box];
        previous_gap = gap;
    }

    // Check that the gaps are decreasing and that the first gap is greater than the last gap.
    passed &= previous_gap < first_gap;
    cout << "Evaluated " << BOXES << " boxes with " << blocks
         << " blocks of " << THREADS_PER_BLOCK << " threads.\n";
    cout << "Result: " << (passed ? "PASS" : "FAIL") << "\n";

    // Free the allocated memory on the device for the input data and results.
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
    CUDA_CHECK(cudaFree(results->center_logei));
    CUDA_CHECK(cudaFree(results->upper_logei));
    CUDA_CHECK(cudaFree(results));

    // Return 0 if the test passed, or 1 if it failed.
    return passed ? 0 : 1;
}
