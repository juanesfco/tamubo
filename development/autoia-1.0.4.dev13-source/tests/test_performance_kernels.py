"""Step-9 regression tests for rigorous performance kernels."""

from __future__ import annotations

from fractions import Fraction
import unittest
import numpy as np

from autoia.trusted import RigorousBinary64Backend as TrustedBackend, rigorous_dot_point, nonnegative_dot_upper
from autoia import RigorousBinary64Backend
from autoia.exceptions import InvalidInputError
from autoia.semantic.linear_solve import build_solution_polynomials_general
from tests.support.exact_binary64 import exact_float, interval_contains_exact, upper_endpoint_ge_exact


class CountingSolve:
    def __init__(self):
        self.calls = 0
        self.rhs_shapes = []

    def __call__(self, A, b):
        self.calls += 1
        self.rhs_shapes.append(np.asarray(b).shape)
        return np.linalg.solve(A, b)


def exact_dot(a, b):
    return sum((exact_float(x) * exact_float(y) for x, y in zip(a, b)), Fraction(0, 1))


class TestVectorizedTrustedReductions(unittest.TestCase):
    def test_vectorized_dot_contains_exact_rational_long_reductions(self):
        rng = np.random.default_rng(2026082911)
        backend = TrustedBackend(reduction_method="vectorized")
        for n in (16, 31, 64, 127, 256):
            for _ in range(8):
                a = rng.normal(size=n)
                b = rng.normal(size=n)
                got = rigorous_dot_point(a, b, backend)
                self.assertTrue(interval_contains_exact(got, exact_dot(a, b)))
        self.assertGreater(backend.reduction_stats["vectorized_calls"], 0)

    def test_vectorized_nonnegative_upper_contains_exact(self):
        rng = np.random.default_rng(2026082912)
        backend = TrustedBackend(reduction_method="vectorized")
        for n in (16, 73, 256):
            a = np.abs(rng.normal(size=n))
            b = np.abs(rng.normal(size=n))
            got = nonnegative_dot_upper(a, b, backend)
            self.assertTrue(upper_endpoint_ge_exact(got, exact_dot(a, b)))

    def test_auto_mode_keeps_small_oracle_path_and_switches_for_long_reductions(self):
        backend = TrustedBackend(
            reduction_method="auto", vectorized_reduction_min_terms=16
        )
        rigorous_dot_point(np.arange(8.0), np.ones(8), backend)
        rigorous_dot_point(np.arange(32.0), np.ones(32), backend)
        stats = backend.reduction_stats
        self.assertEqual(stats["reference_calls"], 1)
        self.assertEqual(stats["vectorized_calls"], 1)

    def test_public_backend_reduction_configuration_is_typed(self):
        with self.assertRaises(InvalidInputError):
            RigorousBinary64Backend(reduction_method="mystery")
        with self.assertRaises(InvalidInputError):
            RigorousBinary64Backend(vectorized_reduction_min_terms=0)


class TestBatchedProposalFactorization(unittest.TestCase):
    def test_polynomial_predictor_batches_one_solve_per_total_degree(self):
        m = 4
        order = 4
        n = 5
        z = (0,) * m
        A0 = np.diag(np.linspace(2.0, 3.0, n))
        A_coeff = {z: A0}
        # Add first-order matrix dependence; the recurrence still only couples
        # to already-computed lower degrees.
        for i in range(m):
            alpha = tuple(1 if j == i else 0 for j in range(m))
            A_coeff[alpha] = np.eye(n) * (0.01 * (i + 1))

        b_coeff = {z: np.linspace(1.0, 2.0, n)}
        # Populate several RHS coefficients so every degree does useful work.
        from autoia.polynomial import multi_indices_exact_degree
        for degree in range(1, order + 1):
            for k, alpha in enumerate(multi_indices_exact_degree(m, degree)):
                b_coeff[alpha] = np.full(n, 0.001 * (degree + k + 1))

        counter = CountingSolve()
        coeff = build_solution_polynomials_general(
            A_coeff, b_coeff, m, order, original_solve=counter
        )
        self.assertEqual(counter.calls, order + 1)
        self.assertTrue(any(len(shape) == 2 and shape[1] > 1 for shape in counter.rhs_shapes))
        self.assertEqual(len(coeff), sum(1 for d in range(order + 1) for _ in multi_indices_exact_degree(m, d)))


if __name__ == "__main__":
    unittest.main()
