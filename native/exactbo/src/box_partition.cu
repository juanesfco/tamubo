#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

using std::cerr;
using std::cout;
using std::fixed;
using std::setprecision;

// A small, standalone version of split_dense_boxes_kernel in partitioning.cu.
//
// It deliberately uses one two-dimensional parent box and one CUDA thread.
// The production concerns (MPI, masks, batches, BoxStore, and spill files) are
// absent so the five resulting child boxes can be checked by hand.

constexpr int DIMENSIONS = 2;
constexpr int CHILDREN = 2 * DIMENSIONS + 1;
constexpr double TOLERANCE = 1.0e-12;

struct Demo {
    double parent_low[DIMENSIONS];
    double parent_high[DIMENSIONS];
    double domain_width[DIMENSIONS];

    int split_order[DIMENSIONS];
    double children_low[CHILDREN * DIMENSIONS];
    double children_high[CHILDREN * DIMENSIONS];
};

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        cerr << "CUDA failed while " << action << ": "
             << cudaGetErrorString(status) << "\n";
        exit(1);
    }
}

__device__ double normalized_width(
    const Demo* demo,
    int dimension) {
    const double box_width =
        demo->parent_high[dimension] - demo->parent_low[dimension];
    return box_width / demo->domain_width[dimension];
}

__device__ int dimension_at_order(
    const Demo* demo,
    int wanted_order) {
    // Count how many dimensions are wider than each candidate dimension.
    // Exact ties are resolved by the lower dimension index.
    for (int dimension = 0; dimension < DIMENSIONS; ++dimension) {
        const double width = normalized_width(demo, dimension);
        int order = 0;

        for (int other = 0; other < DIMENSIONS; ++other) {
            const double other_width = normalized_width(demo, other);
            if (other_width > width ||
                (other_width == width && other < dimension)) {
                ++order;
            }
        }

        if (order == wanted_order) {
            return dimension;
        }
    }

    return DIMENSIONS - 1;
}

__global__ void split_one_box(Demo* demo) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // Step 1: start with five identical copies of the parent box.
    for (int child = 0; child < CHILDREN; ++child) {
        for (int dimension = 0;
             dimension < DIMENSIONS;
             ++dimension) {
            demo->children_low[child * DIMENSIONS + dimension] =
                demo->parent_low[dimension];
            demo->children_high[child * DIMENSIONS + dimension] =
                demo->parent_high[dimension];
        }
    }

    // Step 2: process dimensions from widest to narrowest.
    for (int order = 0; order < DIMENSIONS; ++order) {
        const int dimension = dimension_at_order(demo, order);
        demo->split_order[order] = dimension;

        const double low = demo->parent_low[dimension];
        const double high = demo->parent_high[dimension];
        const double one_third = (high - low) / 3.0;
        const double lower_cut = low + one_third;
        const double upper_cut = high - one_third;

        const int lower_child = 2 * order;
        const int upper_child = lower_child + 1;

        // Create the two side children for this dimension.
        demo->children_high[
            lower_child * DIMENSIONS + dimension] = lower_cut;
        demo->children_low[
            upper_child * DIMENSIONS + dimension] = upper_cut;

        // Children used by later dimensions stay in this dimension's center.
        for (int child = upper_child + 1;
             child < CHILDREN;
             ++child) {
            demo->children_low[
                child * DIMENSIONS + dimension] = lower_cut;
            demo->children_high[
                child * DIMENSIONS + dimension] = upper_cut;
        }
    }
}

double box_volume(
    const double* low,
    const double* high) {
    double volume = 1.0;
    for (int dimension = 0;
         dimension < DIMENSIONS;
         ++dimension) {
        volume *= high[dimension] - low[dimension];
    }
    return volume;
}

double overlap_volume(
    const double* first_low,
    const double* first_high,
    const double* second_low,
    const double* second_high) {
    double volume = 1.0;
    for (int dimension = 0;
         dimension < DIMENSIONS;
         ++dimension) {
        const double overlap_low =
            fmax(first_low[dimension], second_low[dimension]);
        const double overlap_high =
            fmin(first_high[dimension], second_high[dimension]);
        volume *= fmax(overlap_high - overlap_low, 0.0);
    }
    return volume;
}

int main() {
    Demo* demo = nullptr;
    check_cuda(
        cudaMallocManaged(&demo, sizeof(Demo)),
        "allocating the teaching example");

    // x occupies 9/9 = 1 of its search-domain width.
    // y occupies 3/6 = 0.5 of its search-domain width.
    // Therefore the split order must be x, then y.
    demo->parent_low[0] = 0.0;
    demo->parent_low[1] = 0.0;
    demo->parent_high[0] = 9.0;
    demo->parent_high[1] = 3.0;
    demo->domain_width[0] = 9.0;
    demo->domain_width[1] = 6.0;

    split_one_box<<<1, 1>>>(demo);
    check_cuda(cudaGetLastError(), "launching split_one_box");
    check_cuda(cudaDeviceSynchronize(), "waiting for split_one_box");

    const char* dimension_name[DIMENSIONS] = {"x", "y"};
    const char* child_name[CHILDREN] = {
        "lower side of first dimension",
        "upper side of first dimension",
        "lower side of second dimension",
        "upper side of second dimension",
        "center",
    };

    cout << fixed << setprecision(3);

    cout << "Step 1 - normalized parent widths\n";
    for (int dimension = 0;
         dimension < DIMENSIONS;
         ++dimension) {
        const double normalized =
            (demo->parent_high[dimension] -
             demo->parent_low[dimension]) /
            demo->domain_width[dimension];
        cout << "  " << dimension_name[dimension]
             << ": " << normalized << "\n";
    }

    cout << "Step 2 - split order\n";
    for (int order = 0; order < DIMENSIONS; ++order) {
        cout << "  order " << order << ": "
             << dimension_name[demo->split_order[order]] << "\n";
    }

    cout << "Step 3 - generated children\n";
    double sum_of_child_volumes = 0.0;
    bool every_child_is_inside = true;

    for (int child = 0; child < CHILDREN; ++child) {
        const double* low =
            demo->children_low + child * DIMENSIONS;
        const double* high =
            demo->children_high + child * DIMENSIONS;
        const double volume = box_volume(low, high);
        sum_of_child_volumes += volume;

        cout << "  child " << child
             << " (" << child_name[child] << ")"
             << ": [(" << low[0] << ", " << low[1] << "), ("
             << high[0] << ", " << high[1] << ")]"
             << " volume=" << volume << "\n";

        for (int dimension = 0;
             dimension < DIMENSIONS;
             ++dimension) {
            if (low[dimension] <
                    demo->parent_low[dimension] - TOLERANCE ||
                high[dimension] >
                    demo->parent_high[dimension] + TOLERANCE ||
                low[dimension] > high[dimension]) {
                every_child_is_inside = false;
            }
        }
    }

    // Distinct children may touch at boundaries. They must not share a region
    // with positive volume.
    bool child_interiors_do_not_overlap = true;
    for (int first = 0; first < CHILDREN; ++first) {
        for (int second = first + 1;
             second < CHILDREN;
             ++second) {
            const double overlap = overlap_volume(
                demo->children_low + first * DIMENSIONS,
                demo->children_high + first * DIMENSIONS,
                demo->children_low + second * DIMENSIONS,
                demo->children_high + second * DIMENSIONS);
            if (overlap > TOLERANCE) {
                child_interiors_do_not_overlap = false;
            }
        }
    }

    const double parent_volume = box_volume(
        demo->parent_low,
        demo->parent_high);
    const bool volume_is_preserved =
        fabs(sum_of_child_volumes - parent_volume) <= TOLERANCE;

    cout << "Step 4 - geometry checks\n"
         << "  parent volume:              " << parent_volume << "\n"
         << "  sum of child volumes:       "
         << sum_of_child_volumes << "\n"
         << "  all children inside parent: "
         << (every_child_is_inside ? "yes" : "no") << "\n"
         << "  child interiors overlap:    "
         << (child_interiors_do_not_overlap ? "no" : "yes") << "\n";

    const bool passed =
        every_child_is_inside &&
        child_interiors_do_not_overlap &&
        volume_is_preserved;

    cout << "Result: " << (passed ? "PASS" : "FAIL") << "\n";
    cout << "Reminder: the children cover the old box, but a new LHS design "
            "does not necessarily contain the previous best sample point.\n";

    check_cuda(cudaFree(demo), "freeing the teaching example");
    return passed ? 0 : 1;
}
