import math
import unittest
import numpy as np

from tests.support import oracle_v11 as oracle
from autoia.trusted import RigorousBinary64Backend
from autoia.polynomial import algebra as new_alg
from autoia.polynomial import ranging as new_rng


def bits(x):
    return int(np.float64(x).view(np.uint64))


def assert_interval_bits(testcase, got, expected, msg=None):
    testcase.assertEqual(bits(got.lo), bits(expected.lo), msg)
    testcase.assertEqual(bits(got.hi), bits(expected.hi), msg)


def assert_poly_bits(testcase, got, expected):
    testcase.assertEqual(set(got), set(expected))
    for alpha in got:
        testcase.assertEqual(bits(got[alpha]), bits(expected[alpha]), f"coefficient {alpha}")


class TestPolynomialRegression(unittest.TestCase):
    def setUp(self):
        self.new_backend = RigorousBinary64Backend()
        self.old_backend = oracle.RigorousBinary64Backend()

    def random_poly(self, rng, m, max_degree, density=0.55):
        p = {}
        # Include a controlled constant so reciprocal-predictor tests are stable.
        p[(0,) * m] = float(rng.uniform(0.5, 1.5))
        for _ in range(40):
            alpha = tuple(int(v) for v in rng.integers(0, max_degree + 1, size=m))
            if sum(alpha) > max_degree or alpha == (0,) * m:
                continue
            if rng.random() <= density:
                c = float(rng.normal())
                if c != 0.0:
                    p[alpha] = c
        return p

    def test_multiindex_and_cleaning_match_oracle(self):
        self.assertEqual(new_alg.zero_alpha(4), oracle.zero_alpha(4))
        for i in range(4):
            self.assertEqual(new_alg.basis_alpha(4, i), oracle.basis_alpha(4, i))
        self.assertEqual(
            list(new_alg.multi_indices_exact_degree(3, 4)),
            list(oracle.multi_indices_exact_degree(3, 4)),
        )
        p = {(0, 0): 0.0, (1, 0): 1.25, (0, 1): -0.0, (2, 0): -3.5}
        self.assertEqual(new_alg.clean_poly(p), oracle.clean_poly(p))

    def test_monomial_bounds_are_bit_identical(self):
        h = np.array([0.3, 0.7, 1.1], dtype=float)
        for alpha in [(0, 0, 0), (1, 0, 2), (2, 2, 0), (3, 1, 1), (4, 0, 0)]:
            a = new_alg.monomial_magnitude_upper(alpha, h, self.new_backend)
            b = oracle.monomial_magnitude_upper(alpha, h, self.old_backend)
            self.assertEqual(bits(a), bits(b))
            assert_interval_bits(
                self,
                new_alg.monomial_range_symmetric(alpha, h, self.new_backend),
                oracle.monomial_range_symmetric(alpha, h, self.old_backend),
            )

    def test_polynomial_addition_and_multiplication_match_oracle(self):
        rng = np.random.default_rng(20260829)
        for m in (1, 2, 3):
            h = rng.uniform(0.05, 1.0, size=m)
            for _ in range(20):
                p = self.random_poly(rng, m, 3)
                q = self.random_poly(rng, m, 3)

                ng, ne = new_alg.rigorous_poly_add_point(p, q, h, self.new_backend)
                og, oe = oracle.rigorous_poly_add_point(p, q, h, self.old_backend)
                assert_poly_bits(self, ng, og)
                assert_interval_bits(self, ne, oe)

                ng, ne = new_alg.rigorous_poly_mul_truncated(p, q, 3, h, self.new_backend)
                og, oe = oracle.rigorous_poly_mul_truncated(p, q, 3, h, self.old_backend)
                assert_poly_bits(self, ng, og)
                assert_interval_bits(self, ne, oe)

    def test_reciprocal_predictor_is_bit_identical(self):
        rng = np.random.default_rng(1451)
        for m in (1, 2, 3):
            for _ in range(20):
                p = self.random_poly(rng, m, 3)
                n = new_alg.build_reciprocal_predictor_point(p, m, 3, fallback_center=1.1)
                o = oracle.build_reciprocal_predictor_point(p, m, 3, fallback_center=1.1)
                assert_poly_bits(self, n, o)

    def test_primary_range_methods_are_bit_identical(self):
        rng = np.random.default_rng(73019)
        methods = ("termwise", "monotonicity", "bernstein", "portfolio")
        for m in (1, 2, 3):
            for _ in range(12):
                h = rng.uniform(0.05, 0.8, size=m)
                p = self.random_poly(rng, m, 3)
                for method in methods:
                    nr, nm = new_rng.range_polynomial(p, h, self.new_backend, method)
                    or_, om = oracle.range_polynomial(p, h, self.old_backend, method)
                    assert_interval_bits(self, nr, or_, f"{m=} {method=}")
                    self.assertEqual(nm["method"], om["method"])

    def test_derivative_and_monotonicity_metadata_match_oracle(self):
        rng = np.random.default_rng(9991)
        for m in (2, 3, 4):
            h = rng.uniform(0.1, 0.7, size=m)
            for _ in range(8):
                p = self.random_poly(rng, m, 3)
                nr, nm = new_rng.range_monotonicity_scalar(p, h, self.new_backend)
                or_, om = oracle.range_monotonicity_scalar(p, h, self.old_backend)
                assert_interval_bits(self, nr, or_)
                self.assertEqual(tuple(nm["signs"]), tuple(om["signs"]))
                self.assertEqual(len(nm["derivative_ranges"]), len(om["derivative_ranges"]))
                for a, b in zip(nm["derivative_ranges"], om["derivative_ranges"]):
                    assert_interval_bits(self, a, b)

    def test_reduced_bernstein_is_bit_identical(self):
        # Include both fully monotone and unresolved-direction cases.
        cases = [
            (
                {(0, 0, 0): 0.2, (1, 0, 0): 1.0, (0, 1, 0): -0.7,
                 (0, 0, 1): 0.5, (2, 0, 1): 0.1},
                np.array([0.2, 0.3, 0.25]),
            ),
            (
                {(0, 0): 0.1, (2, 0): 1.0, (0, 2): -0.8, (1, 1): 0.6,
                 (3, 0): 0.2},
                np.array([0.8, 0.7]),
            ),
        ]
        for p, h in cases:
            nmono, nm = new_rng.range_monotonicity_scalar(p, h, self.new_backend)
            omono, om = oracle.range_monotonicity_scalar(p, h, self.old_backend)
            assert_interval_bits(self, nmono, omono)
            nr, nmeta = new_rng.range_monotonicity_reduced_bernstein_scalar(
                p, h, self.new_backend, monotonicity_meta=nm
            )
            or_, ometa = oracle.range_monotonicity_reduced_bernstein_scalar(
                p, h, self.old_backend, monotonicity_meta=om
            )
            assert_interval_bits(self, nr, or_)
            for key in (
                "unresolved_variables", "num_coefficients",
                "full_bernstein_coefficients", "degree_vector",
                "monotonicity_complete",
            ):
                self.assertEqual(nmeta[key], ometa[key], key)

    def test_degree_and_reduced_cost_estimators_match_oracle(self):
        p = {
            (0, 0, 0, 0): 1.0,
            (3, 0, 0, 0): 0.2,
            (0, 2, 0, 1): -0.4,
            (1, 1, 1, 0): 0.7,
        }
        nd = new_rng._degree_vector_and_bernstein_cost(p, 4)
        od = oracle._degree_vector_and_bernstein_cost(p, 4)
        self.assertEqual(nd, od)
        for signs in [(1, 0, -1, 0), (0, 0, 0, 0), (1, -1, 1, -1)]:
            self.assertEqual(
                new_rng.estimate_monotonicity_reduced_bernstein_cost(p, 4, signs),
                oracle.estimate_monotonicity_reduced_bernstein_cost(p, 4, signs),
            )


if __name__ == "__main__":
    unittest.main()
