#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

using std::copy;
using std::cerr;
using std::cout;
using std::fixed;
using std::max;
using std::setprecision;

// A small, standalone version of the bounding mathematics in partitioning.cu.
//
// It deliberately uses:
//   * one hyperbox,
//   * two GP training points,
//   * one CUDA thread, and
//   * CUDA managed memory.
//
// This is not performance code. Its purpose is to make every calculation fit
// on the terminal so it can be checked by hand.

constexpr int DIMENSIONS = 2;
constexpr int TRAINING_POINTS = 2;
constexpr int SAMPLE_POINTS = 9;
constexpr double INV_SQRT_2_PI = 0.39894228040143267794;
constexpr double SQRT_2 = 1.41421356237309504880;

struct Demo {
    // Inputs.
    double box_low[DIMENSIONS];
    double box_high[DIMENSIONS];
    double training_x[TRAINING_POINTS * DIMENSIONS];
    double alpha[TRAINING_POINTS];
    double cholesky[TRAINING_POINTS * TRAINING_POINTS];
    double length_scale[DIMENSIONS];
    double signal_variance;
    double best_observed_y;
    double sample_x[SAMPLE_POINTS * DIMENSIONS];

    // Box-bound outputs.
    double kernel_low[TRAINING_POINTS];
    double kernel_high[TRAINING_POINTS];
    double mean_low;
    double mean_high;
    double solve_low[TRAINING_POINTS];
    double solve_high[TRAINING_POINTS];
    double sigma_low;
    double sigma_high;
    double ei_high;

    // Exact outputs at the nine test points.
    double point_mean[SAMPLE_POINTS];
    double point_sigma[SAMPLE_POINTS];
    double point_ei[SAMPLE_POINTS];
};

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        cerr << "CUDA failed while " << action << ": "
             << cudaGetErrorString(status) << "\n";
        exit(1);
    }
}

__device__ double normal_cdf(double z) {
    // Same erf approximation used by partitioning.cu.
    const double p = 0.3275911;
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;

    const double argument = z / SQRT_2;
    const double sign = (argument > 0.0) - (argument < 0.0);
    const double x = fabs(argument);
    const double t = 1.0 / (1.0 + p * x);
    const double polynomial =
        (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t);
    const double erf_value =
        sign * (1.0 - polynomial * exp(-x * x));
    return 0.5 * (1.0 + erf_value);
}

__device__ double normal_pdf(double z) {
    return INV_SQRT_2_PI * exp(-0.5 * z * z);
}

__device__ void multiply_intervals(
    double a_low,
    double a_high,
    double b_low,
    double b_high,
    double& result_low,
    double& result_high) {
    const double product_1 = a_low * b_low;
    const double product_2 = a_low * b_high;
    const double product_3 = a_high * b_low;
    const double product_4 = a_high * b_high;

    result_low = fmin(
        fmin(product_1, product_2),
        fmin(product_3, product_4));
    result_high = fmax(
        fmax(product_1, product_2),
        fmax(product_3, product_4));
}

__device__ double expected_improvement(
    double mean,
    double sigma,
    double best_observed_y) {
    sigma = fmax(sigma, 1.0e-12);
    const double improvement = best_observed_y - mean;
    const double z = improvement / sigma;
    return improvement * normal_cdf(z) + sigma * normal_pdf(z);
}

__device__ double expected_improvement_upper_bound(
    double mean_low,
    double mean_high,
    double sigma_low,
    double sigma_high,
    double best_observed_y) {
    // EI = N * Phi(Z) + sigma * phi(Z)
    // N = best_observed_y - mean
    // Z = N / sigma
    const double improvement_low = best_observed_y - mean_high;
    const double improvement_high = best_observed_y - mean_low;

    const double inverse_sigma_low = 1.0 / sigma_high;
    const double inverse_sigma_high = 1.0 / sigma_low;

    double z_low = 0.0;
    double z_high = 0.0;
    multiply_intervals(
        improvement_low,
        improvement_high,
        inverse_sigma_low,
        inverse_sigma_high,
        z_low,
        z_high);

    const double cdf_low = normal_cdf(z_low);
    const double cdf_high = normal_cdf(z_high);

    const double pdf_at_low = normal_pdf(z_low);
    const double pdf_at_high = normal_pdf(z_high);
    const double pdf_low = fmin(pdf_at_low, pdf_at_high);
    double pdf_high = fmax(pdf_at_low, pdf_at_high);
    if (z_low <= 0.0 && z_high >= 0.0) {
        pdf_high = INV_SQRT_2_PI;
    }

    double first_low = 0.0;
    double first_high = 0.0;
    double second_low = 0.0;
    double second_high = 0.0;
    multiply_intervals(
        improvement_low,
        improvement_high,
        cdf_low,
        cdf_high,
        first_low,
        first_high);
    multiply_intervals(
        sigma_low,
        sigma_high,
        pdf_low,
        pdf_high,
        second_low,
        second_high);

    return fmax(first_high + second_high, 0.0);
}

__global__ void run_demo(Demo* demo) {
    // This teaching kernel executes the stages in source order.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // Step 1: bound every RBF kernel value over the complete box.
    demo->mean_low = 0.0;
    demo->mean_high = 0.0;

    for (int point = 0; point < TRAINING_POINTS; ++point) {
        double nearest_distance_squared = 0.0;
        double farthest_distance_squared = 0.0;

        for (int dimension = 0; dimension < DIMENSIONS; ++dimension) {
            const double training_value =
                demo->training_x[point * DIMENSIONS + dimension];
            const double scale = demo->length_scale[dimension];

            const double distance_from_low =
                (demo->box_low[dimension] - training_value) / scale;
            const double distance_from_high =
                (training_value - demo->box_high[dimension]) / scale;

            // Nearest distance is zero when the training value is inside the
            // interval. Farthest distance reaches one of the interval ends.
            const double nearest = fmax(
                fmax(distance_from_low, distance_from_high),
                0.0);
            const double farthest = fmax(
                fabs(distance_from_low),
                fabs(distance_from_high));

            nearest_distance_squared += nearest * nearest;
            farthest_distance_squared += farthest * farthest;
        }

        demo->kernel_low[point] =
            demo->signal_variance *
            exp(-0.5 * farthest_distance_squared);
        demo->kernel_high[point] =
            demo->signal_variance *
            exp(-0.5 * nearest_distance_squared);

        // Step 2: multiply the kernel interval by alpha.
        if (demo->alpha[point] >= 0.0) {
            demo->mean_low +=
                demo->kernel_low[point] * demo->alpha[point];
            demo->mean_high +=
                demo->kernel_high[point] * demo->alpha[point];
        } else {
            demo->mean_low +=
                demo->kernel_high[point] * demo->alpha[point];
            demo->mean_high +=
                demo->kernel_low[point] * demo->alpha[point];
        }
    }

    // Step 3: interval forward substitution through the Cholesky factor.
    double squared_norm_low = 0.0;
    double squared_norm_high = 0.0;

    for (int row = 0; row < TRAINING_POINTS; ++row) {
        double sum_low = 0.0;
        double sum_high = 0.0;

        for (int column = 0; column < row; ++column) {
            const double coefficient =
                demo->cholesky[row * TRAINING_POINTS + column];

            if (coefficient >= 0.0) {
                sum_low += coefficient * demo->solve_low[column];
                sum_high += coefficient * demo->solve_high[column];
            } else {
                sum_low += coefficient * demo->solve_high[column];
                sum_high += coefficient * demo->solve_low[column];
            }
        }

        const double diagonal =
            demo->cholesky[row * TRAINING_POINTS + row];
        demo->solve_low[row] =
            (demo->kernel_low[row] - sum_high) / diagonal;
        demo->solve_high[row] =
            (demo->kernel_high[row] - sum_low) / diagonal;

        const double low_squared =
            demo->solve_low[row] * demo->solve_low[row];
        const double high_squared =
            demo->solve_high[row] * demo->solve_high[row];

        squared_norm_high += fmax(low_squared, high_squared);
        if (demo->solve_low[row] <= 0.0 &&
            demo->solve_high[row] >= 0.0) {
            squared_norm_low += 0.0;
        } else {
            squared_norm_low += fmin(low_squared, high_squared);
        }
    }

    // Step 4: posterior variance = signal variance - ||v||^2.
    demo->sigma_low = sqrt(fmax(
        demo->signal_variance - squared_norm_high,
        1.0e-12));
    demo->sigma_high = sqrt(fmax(
        demo->signal_variance - squared_norm_low,
        1.0e-12));

    // Step 5: turn the mean and sigma intervals into an EI upper bound.
    demo->ei_high = expected_improvement_upper_bound(
        demo->mean_low,
        demo->mean_high,
        demo->sigma_low,
        demo->sigma_high,
        demo->best_observed_y);

    // Step 6: evaluate exact GP mean, sigma, and EI at nine test points.
    for (int sample = 0; sample < SAMPLE_POINTS; ++sample) {
        double mean = 0.0;
        double squared_norm = 0.0;
        double solve[TRAINING_POINTS];

        for (int row = 0; row < TRAINING_POINTS; ++row) {
            double distance_squared = 0.0;

            for (int dimension = 0; dimension < DIMENSIONS; ++dimension) {
                const double difference =
                    (demo->sample_x[sample * DIMENSIONS + dimension] -
                     demo->training_x[row * DIMENSIONS + dimension]) /
                    demo->length_scale[dimension];
                distance_squared += difference * difference;
            }

            const double kernel =
                demo->signal_variance *
                exp(-0.5 * distance_squared);
            mean += kernel * demo->alpha[row];

            double previous_rows = 0.0;
            for (int column = 0; column < row; ++column) {
                previous_rows +=
                    demo->cholesky[row * TRAINING_POINTS + column] *
                    solve[column];
            }

            solve[row] =
                (kernel - previous_rows) /
                demo->cholesky[row * TRAINING_POINTS + row];
            squared_norm += solve[row] * solve[row];
        }

        const double sigma = sqrt(fmax(
            demo->signal_variance - squared_norm,
            1.0e-12));

        demo->point_mean[sample] = mean;
        demo->point_sigma[sample] = sigma;
        demo->point_ei[sample] = expected_improvement(
            mean,
            sigma,
            demo->best_observed_y);
    }
}

int main() {
    Demo* demo = nullptr;
    check_cuda(
        cudaMallocManaged(&demo, sizeof(Demo)),
        "allocating the teaching example");

    // The two-by-two Cholesky matrix is stored row by row:
    // [ L00,   0 ]
    // [ L10, L11 ]
    const double box_low[DIMENSIONS] = {0.2, 0.1};
    const double box_high[DIMENSIONS] = {0.8, 0.7};
    const double training_x[TRAINING_POINTS * DIMENSIONS] = {
        0.1, 0.2,
        0.9, 0.8,
    };
    const double alpha[TRAINING_POINTS] = {-0.8, 0.5};
    const double cholesky[TRAINING_POINTS * TRAINING_POINTS] = {
        1.0049875621, 0.0,
        0.0816778680, 1.0016616507,
    };
    const double length_scale[DIMENSIONS] = {0.4, 0.6};
    const double sample_x[SAMPLE_POINTS * DIMENSIONS] = {
        0.2, 0.1,  0.5, 0.1,  0.8, 0.1,
        0.2, 0.4,  0.5, 0.4,  0.8, 0.4,
        0.2, 0.7,  0.5, 0.7,  0.8, 0.7,
    };

    copy(box_low, box_low + DIMENSIONS, demo->box_low);
    copy(box_high, box_high + DIMENSIONS, demo->box_high);
    copy(
        training_x,
        training_x + TRAINING_POINTS * DIMENSIONS,
        demo->training_x);
    copy(alpha, alpha + TRAINING_POINTS, demo->alpha);
    copy(
        cholesky,
        cholesky + TRAINING_POINTS * TRAINING_POINTS,
        demo->cholesky);
    copy(
        length_scale,
        length_scale + DIMENSIONS,
        demo->length_scale);
    copy(
        sample_x,
        sample_x + SAMPLE_POINTS * DIMENSIONS,
        demo->sample_x);
    demo->signal_variance = 1.0;
    demo->best_observed_y = -0.4;

    run_demo<<<1, 1>>>(demo);
    check_cuda(cudaGetLastError(), "launching run_demo");
    check_cuda(cudaDeviceSynchronize(), "waiting for run_demo");

    cout << fixed << setprecision(9);

    cout << "Step 1 - RBF kernel intervals\n";
    for (int point = 0; point < TRAINING_POINTS; ++point) {
        cout << "  K[" << point << "]: ["
             << demo->kernel_low[point] << ", "
             << demo->kernel_high[point] << "]\n";
    }

    cout << "Step 2 - posterior mean interval\n"
         << "  mean: [" << demo->mean_low << ", "
         << demo->mean_high << "]\n";

    cout << "Step 3 - Cholesky solve intervals\n";
    for (int row = 0; row < TRAINING_POINTS; ++row) {
        cout << "  v[" << row << "]: ["
             << demo->solve_low[row] << ", "
             << demo->solve_high[row] << "]\n";
    }

    cout << "Step 4 - posterior sigma interval\n"
         << "  sigma: [" << demo->sigma_low << ", "
         << demo->sigma_high << "]\n";

    cout << "Step 5 - EI upper bound for the complete box\n"
         << "  EI_hi: " << demo->ei_high << "\n";

    cout << "Step 6 - exact EI at nine points inside the box\n";
    double largest_sampled_ei = 0.0;
    bool every_point_is_inside_intervals = true;

    for (int sample = 0; sample < SAMPLE_POINTS; ++sample) {
        largest_sampled_ei = max(
            largest_sampled_ei,
            demo->point_ei[sample]);

        const bool mean_is_inside =
            demo->point_mean[sample] >= demo->mean_low - 1.0e-10 &&
            demo->point_mean[sample] <= demo->mean_high + 1.0e-10;
        const bool sigma_is_inside =
            demo->point_sigma[sample] >= demo->sigma_low - 1.0e-10 &&
            demo->point_sigma[sample] <= demo->sigma_high + 1.0e-10;
        const bool ei_is_below =
            demo->point_ei[sample] <= demo->ei_high + 1.0e-10;
        every_point_is_inside_intervals &=
            mean_is_inside && sigma_is_inside && ei_is_below;

        cout << "  x=("
             << demo->sample_x[sample * DIMENSIONS] << ", "
             << demo->sample_x[sample * DIMENSIONS + 1] << ")"
             << " mean=" << demo->point_mean[sample]
             << " sigma=" << demo->point_sigma[sample]
             << " EI=" << demo->point_ei[sample] << "\n";
    }

    cout << "  largest sampled EI: " << largest_sampled_ei << "\n";

    const bool intervals_are_ordered =
        demo->kernel_low[0] <= demo->kernel_high[0] &&
        demo->kernel_low[1] <= demo->kernel_high[1] &&
        demo->mean_low <= demo->mean_high &&
        demo->sigma_low <= demo->sigma_high;
    const bool passed =
        intervals_are_ordered && every_point_is_inside_intervals;

    cout << "Result: " << (passed ? "PASS" : "FAIL") << "\n";
    cout << "Reminder: sampled EI is a feasible lower bound. If later boxes "
            "use different sample points, their sampled maximum can decrease "
            "unless the previous winner is retained as an incumbent.\n";

    check_cuda(cudaFree(demo), "freeing the teaching example");
    return passed ? 0 : 1;
}
