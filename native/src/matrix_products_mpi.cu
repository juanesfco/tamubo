#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

constexpr int kTile = 16;

struct Matrix {
    std::size_t n = 0;
    std::vector<double> values;
};

struct Options {
    std::string a_path;
    std::string b_path;
    std::string out_ab_path = "AB.bin";
    std::string out_ba_path = "BA.bin";
    int hold_ms = 0;
    int repeats = 1;
    int cycles = 1;
};

__global__ void matmul_kernel(const double* a, const double* b, double* c, int n) {
    __shared__ double tile_a[kTile][kTile];
    __shared__ double tile_b[kTile][kTile];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int base = 0; base < n; base += kTile) {
        int a_col = base + threadIdx.x;
        int b_row = base + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? a[row * n + a_col] : 0.0;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < kTile; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

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

Matrix read_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path);
    }

    std::uint64_t n64 = 0;
    in.read(reinterpret_cast<char*>(&n64), sizeof(n64));
    if (!in || n64 == 0) {
        throw std::runtime_error("invalid matrix header in: " + path);
    }

    Matrix matrix;
    matrix.n = static_cast<std::size_t>(n64);
    matrix.values.resize(matrix.n * matrix.n);
    in.read(reinterpret_cast<char*>(matrix.values.data()),
            static_cast<std::streamsize>(matrix.values.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("invalid matrix payload in: " + path);
    }

    return matrix;
}

void write_matrix(const std::string& path, std::size_t n, const std::vector<double>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    std::uint64_t n64 = static_cast<std::uint64_t>(n);
    out.write(reinterpret_cast<const char*>(&n64), sizeof(n64));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path);
    }
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

        if (std::strcmp(argv[i], "--a") == 0) {
            options.a_path = need_value("--a");
        } else if (std::strcmp(argv[i], "--b") == 0) {
            options.b_path = need_value("--b");
        } else if (std::strcmp(argv[i], "--out-ab") == 0) {
            options.out_ab_path = need_value("--out-ab");
        } else if (std::strcmp(argv[i], "--out-ba") == 0) {
            options.out_ba_path = need_value("--out-ba");
        } else if (std::strcmp(argv[i], "--hold-ms") == 0) {
            options.hold_ms = std::stoi(need_value("--hold-ms"));
        } else if (std::strcmp(argv[i], "--repeats") == 0) {
            options.repeats = std::stoi(need_value("--repeats"));
        } else if (std::strcmp(argv[i], "--cycles") == 0) {
            options.cycles = std::stoi(need_value("--cycles"));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Usage: matrix_products_mpi --a A.bin --b B.bin "
                << "[--out-ab AB.bin] [--out-ba BA.bin] "
                << "[--hold-ms ms] [--repeats n] [--cycles n]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }

    if (options.a_path.empty() || options.b_path.empty()) {
        throw std::runtime_error("--a and --b are required");
    }
    if (options.hold_ms < 0) {
        throw std::runtime_error("--hold-ms must be >= 0");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("--repeats must be > 0");
    }
    if (options.cycles <= 0) {
        throw std::runtime_error("--cycles must be > 0");
    }

    return options;
}

std::vector<double> run_product(
    const std::string& label,
    const double* left,
    const double* right,
    std::size_t n,
    int rank,
    int device,
    const Options& options) {
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::size_t elements = n * n;
    std::size_t bytes = elements * sizeof(double);
    int n_int = static_cast<int>(n);
    if (static_cast<std::size_t>(n_int) != n) {
        throw std::runtime_error("matrix dimension is too large for this simple kernel");
    }

    std::vector<double> host_result(elements);

    std::cout << "rank " << rank << " starting " << label
              << " on pid=" << getpid()
              << " launch_cpu=" << current_cpu()
              << " gpu_device=" << device
              << " gpu_name=\"" << prop.name << "\""
              << " matrix=" << n << "x" << n
              << " bytes_per_matrix=" << format_bytes(bytes)
              << " repeats=" << options.repeats
              << " cycles=" << options.cycles << "\n";

    dim3 block(kTile, kTile);
    dim3 grid((n_int + block.x - 1) / block.x, (n_int + block.y - 1) / block.y);
    double total_seconds = 0.0;

    for (int cycle = 1; cycle <= options.cycles; ++cycle) {
        bool final_cycle = cycle == options.cycles;
        std::string cycle_label = label;
        if (options.cycles > 1) {
            cycle_label += " cycle " + std::to_string(cycle) + "/" + std::to_string(options.cycles);
        }

        print_device_memory(rank, cycle_label + " before cudaMalloc");

        double* d_left = nullptr;
        double* d_right = nullptr;
        double* d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_left, bytes));
        CUDA_CHECK(cudaMalloc(&d_right, bytes));
        CUDA_CHECK(cudaMalloc(&d_result, bytes));
        print_device_memory(rank, cycle_label + " after cudaMalloc left/right/result");

        CUDA_CHECK(cudaMemcpy(d_left, left, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_right, right, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_result, 0, bytes));
        print_device_memory(rank, cycle_label + " after host-to-device copies");

        auto t0 = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            matmul_kernel<<<grid, block>>>(d_left, d_right, d_result, n_int);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        total_seconds += seconds;
        print_device_memory(rank, cycle_label + " after kernel");

        if (options.hold_ms > 0) {
            std::cout << "rank " << rank << " holding " << cycle_label
                      << " allocations for " << options.hold_ms << " ms\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(options.hold_ms));
        }

        if (final_cycle) {
            CUDA_CHECK(cudaMemcpy(host_result.data(), d_result, bytes, cudaMemcpyDeviceToHost));
            print_device_memory(rank, cycle_label + " after device-to-host copy");
        }

        CUDA_CHECK(cudaFree(d_result));
        CUDA_CHECK(cudaFree(d_right));
        CUDA_CHECK(cudaFree(d_left));
        CUDA_CHECK(cudaDeviceSynchronize());
        print_device_memory(rank, cycle_label + " after cudaFree");

        double cycle_gflops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                              static_cast<double>(n) * static_cast<double>(options.repeats) /
                              seconds / 1.0e9;
        std::cout << "rank " << rank << " finished " << cycle_label
                  << " seconds=" << seconds
                  << " approx_gflop_s=" << cycle_gflops << "\n";
    }

    double total_gflops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                          static_cast<double>(n) * static_cast<double>(options.repeats) *
                          static_cast<double>(options.cycles) / total_seconds / 1.0e9;
    std::cout << "rank " << rank << " finished all " << label
              << " cycles total_kernel_seconds=" << total_seconds
              << " aggregate_approx_gflop_s=" << total_gflops << "\n";

    return host_result;
}

void compute_and_save(
    const std::string& label,
    const Matrix& left,
    const Matrix& right,
    const std::string& output_path,
    int rank,
    int device,
    const Options& options) {
    std::vector<double> result =
        run_product(label, left.values.data(), right.values.data(), left.n, rank, device, options);
    write_matrix(output_path, left.n, result);
    std::cout << "rank " << rank << " wrote " << label << " to " << output_path << "\n";
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

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("no CUDA devices visible");
        }

        bool concurrent = (size >= 2 && device_count >= 2);
        if (rank == 0) {
            std::cout << "matrix_products_mpi: mpi_size=" << size
                      << " device_count=" << device_count
                      << " execution=" << (concurrent ? "concurrent AB/BA" : "sequential AB then BA")
                      << "\n";
        }

        bool active_rank = concurrent ? (rank == 0 || rank == 1) : (rank == 0);
        if (!active_rank) {
            MPI_Finalize();
            return EXIT_SUCCESS;
        }

        Matrix a = read_matrix(options.a_path);
        Matrix b = read_matrix(options.b_path);
        if (a.n != b.n) {
            throw std::runtime_error("A and B must have the same square dimension");
        }

        if (concurrent) {
            if (rank == 0) {
                compute_and_save("AB", a, b, options.out_ab_path, rank, 0, options);
            } else if (rank == 1) {
                compute_and_save("BA", b, a, options.out_ba_path, rank, 1, options);
            }
        } else if (rank == 0) {
            compute_and_save("AB", a, b, options.out_ab_path, rank, 0, options);
            compute_and_save("BA", b, a, options.out_ba_path, rank, 0, options);
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank " << rank << " matrix_products_mpi: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
