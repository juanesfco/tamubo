#include <vector>
#include <numeric>
#include <string>
#include <iostream>

std::vector<double> centered_latin_hypercube_unit(int n, int d) {
    std::vector<double> lhs(n*d);
    std::vector<double> centers(n);
    for (int i = 0; i < n; ++i) {
        centers[i] = (i + 0.5) / n;
    }
    for (int j = 0; j < d; ++j) {
        int step = 2 * j + 1;
        while (std::gcd(step, n) != 1) {
            step += 2;
        }
        for (int i = 0; i < n; ++i) {
            lhs[i*d + j] = centers[(i * step + j) % n];
        }
    }
    return lhs;
}

int main(int argc, char *argv[]) {
    int n = std::stoi(argv[1]);
    int d = std::stoi(argv[2]);
    std::vector<double> lhs = centered_latin_hypercube_unit(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            std::cout << lhs[i*d + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}

