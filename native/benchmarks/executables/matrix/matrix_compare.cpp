#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Matrix {
    std::size_t n = 0;
    std::vector<double> values;
};

struct Options {
    std::string expected_path;
    std::string actual_path;
    double atol = 1e-8;
    double rtol = 1e-8;
};

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

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* flag) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--expected") == 0) {
            options.expected_path = need_value("--expected");
        } else if (std::strcmp(argv[i], "--actual") == 0) {
            options.actual_path = need_value("--actual");
        } else if (std::strcmp(argv[i], "--atol") == 0) {
            options.atol = std::stod(need_value("--atol"));
        } else if (std::strcmp(argv[i], "--rtol") == 0) {
            options.rtol = std::stod(need_value("--rtol"));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Usage: matrix_compare --expected A.bin --actual B.bin "
                << "[--atol tol] [--rtol tol]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }

    if (options.expected_path.empty() || options.actual_path.empty()) {
        throw std::runtime_error("--expected and --actual are required");
    }
    if (options.atol < 0.0 || options.rtol < 0.0) {
        throw std::runtime_error("--atol and --rtol must be nonnegative");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        Matrix expected = read_matrix(options.expected_path);
        Matrix actual = read_matrix(options.actual_path);

        if (expected.n != actual.n) {
            throw std::runtime_error("matrix dimensions differ");
        }

        double max_abs = 0.0;
        double max_rel = 0.0;
        double sum_sq = 0.0;
        std::size_t max_index = 0;
        std::size_t failures = 0;

        for (std::size_t idx = 0; idx < expected.values.size(); ++idx) {
            double e = expected.values[idx];
            double a = actual.values[idx];
            double abs_err = std::abs(a - e);
            double denom = std::max(std::abs(e), std::numeric_limits<double>::min());
            double rel_err = abs_err / denom;
            double allowed = options.atol + options.rtol * std::abs(e);

            if (abs_err > max_abs) {
                max_abs = abs_err;
                max_rel = rel_err;
                max_index = idx;
            }
            sum_sq += abs_err * abs_err;
            if (abs_err > allowed) {
                ++failures;
            }
        }

        double rmse = std::sqrt(sum_sq / static_cast<double>(expected.values.size()));
        std::size_t row = max_index / expected.n;
        std::size_t col = max_index % expected.n;

        std::cout << "matrix_compare: n=" << expected.n
                  << " elements=" << expected.values.size()
                  << " max_abs=" << max_abs
                  << " max_rel_at_max_abs=" << max_rel
                  << " rmse=" << rmse
                  << " max_abs_index=(" << row << "," << col << ")"
                  << " failures=" << failures
                  << " atol=" << options.atol
                  << " rtol=" << options.rtol << "\n";

        if (failures != 0) {
            std::cerr << "matrix_compare: comparison failed\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << "matrix_compare: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
}
