#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parse_size(const char* text) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0) {
        throw std::runtime_error("matrix size must be a positive integer");
    }
    return static_cast<std::size_t>(value);
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: matrix_make_inputs <n> <A.bin> <B.bin>\n";
        return EXIT_FAILURE;
    }

    try {
        std::size_t n = parse_size(argv[1]);
        std::vector<double> a(n * n);
        std::vector<double> b(n * n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                a[i * n + j] = static_cast<double>((i + 2 * j) % 17) / 17.0;
                b[i * n + j] = static_cast<double>((3 * i + j + 1) % 19) / 19.0;
            }
        }

        write_matrix(argv[2], n, a);
        write_matrix(argv[3], n, b);
        std::cout << "wrote " << n << "x" << n << " matrices\n";
    } catch (const std::exception& exc) {
        std::cerr << "matrix_make_inputs: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
