"""Independent exact-rational validation of proof-critical reductions."""

from __future__ import annotations

from fractions import Fraction
import math
import sys
import unittest
import numpy as np

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    rigorous_dot_point,
    rigorous_matmul_point,
    rigorous_matvec_point,
    rigorous_matvec_point_interval,
    rigorous_matmul_point_interval,
    interval_matvec,
    nonnegative_dot_upper,
    nonnegative_matvec_upper,
    nonnegative_matmul_upper,
    row_sum_norm_upper,
    matrix_product_abs_upper,
    identity_minus_product_abs_upper,
    add_nonnegative_matrices_upper,
    scale_nonnegative_matrix_upper,
)
from tests.support.exact_binary64 import (
    exact_float,
    interval_contains_exact,
    lower_endpoint_le_exact,
    upper_endpoint_ge_exact,
)


def exact_dot(a, b) -> Fraction:
    return sum((exact_float(ai) * exact_float(bi) for ai, bi in zip(a, b)), Fraction(0, 1))


def exact_interval_linear_combination(coeffs, intervals) -> tuple[Fraction, Fraction]:
    lo = Fraction(0, 1)
    hi = Fraction(0, 1)
    for a, I in zip(coeffs, intervals):
        aq = exact_float(a)
        ilo = exact_float(I.lo)
        ihi = exact_float(I.hi)
        if aq >= 0:
            lo += aq * ilo
            hi += aq * ihi
        else:
            lo += aq * ihi
            hi += aq * ilo
    return lo, hi


def exact_interval_dot(intervals_a, intervals_b) -> tuple[Fraction, Fraction]:
    lo = Fraction(0, 1)
    hi = Fraction(0, 1)
    for A, B in zip(intervals_a, intervals_b):
        vals = [
            exact_float(a) * exact_float(b)
            for a in (A.lo, A.hi)
            for b in (B.lo, B.hi)
        ]
        lo += min(vals)
        hi += max(vals)
    return lo, hi


class TestExactReductions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = RigorousBinary64Backend()

    def assert_interval_contains_range(self, I, lo, hi):
        self.assertTrue(lower_endpoint_le_exact(I.lo, lo), (I, lo, hi))
        self.assertTrue(upper_endpoint_ge_exact(I.hi, hi), (I, lo, hi))

    def test_adversarial_cancellation_dot_products(self):
        cases = [
            (np.array([1.0e16, 1.0, -1.0e16]), np.ones(3)),
            (np.array([2.0**500, 1.0, -(2.0**500)]), np.ones(3)),
            (np.array([1.0, -1.0] * 100 + [2.0**-1000]), np.ones(201)),
            (np.array([2.0**-500, 2.0**500, -2.0**500]), np.ones(3)),
        ]
        for a, b in cases:
            got = rigorous_dot_point(a, b, self.backend)
            exact = exact_dot(a, b)
            self.assertTrue(interval_contains_exact(got, exact), (a, b, exact, got))

    def test_random_dot_products_against_exact_rationals(self):
        rng = np.random.default_rng(2026082903)
        for n in (0, 1, 2, 5, 17, 64):
            for _ in range(100):
                exponents_a = rng.integers(-900, 900, size=n)
                exponents_b = rng.integers(-100, 100, size=n)
                mant_a = rng.uniform(-1.0, 1.0, size=n)
                mant_b = rng.uniform(-1.0, 1.0, size=n)
                a = np.ldexp(mant_a, exponents_a).astype(float)
                b = np.ldexp(mant_b, exponents_b).astype(float)
                # Avoid individual exact products outside finite binary64 range;
                # this test is for accumulation/cancellation, not fail-closed overflow.
                safe = True
                for ai, bi in zip(a, b):
                    if abs(exact_float(ai) * exact_float(bi)) > exact_float(sys.float_info.max):
                        safe = False
                        break
                if not safe:
                    continue
                got = rigorous_dot_point(a, b, self.backend)
                exact = exact_dot(a, b)
                self.assertTrue(interval_contains_exact(got, exact), (n, exact, got))

    def test_point_matrix_products_against_exact_rationals(self):
        A = np.array([
            [1.0e16, 1.0, -1.0e16],
            [2.0**-500, 2.0**500, -2.0**500],
        ])
        B = np.array([
            [1.0, 2.0],
            [1.0, -3.0],
            [1.0, 2.0],
        ])
        mm = rigorous_matmul_point(A, B, self.backend)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                self.assertTrue(interval_contains_exact(mm[i, j], exact_dot(A[i], B[:, j])))
        x = np.array([1.0, 1.0, 1.0])
        mv = rigorous_matvec_point(A, x, self.backend)
        for i in range(A.shape[0]):
            self.assertTrue(interval_contains_exact(mv[i], exact_dot(A[i], x)))

    def test_point_times_interval_reductions_against_exact_extrema(self):
        A = np.array([
            [1.0e16, 1.0, -1.0e16],
            [-2.0, 0.5, 3.0],
        ])
        xI = np.array([
            Interval(1.0, 1.0),
            Interval(-1.0, 2.0),
            Interval(1.0, 1.0),
        ], dtype=object)
        got = rigorous_matvec_point_interval(A, xI, self.backend)
        for i in range(A.shape[0]):
            lo, hi = exact_interval_linear_combination(A[i], xI)
            self.assert_interval_contains_range(got[i], lo, hi)

        BI = np.array([
            [Interval(1.0, 1.0), Interval(-1.0, 1.0)],
            [Interval(-1.0, 2.0), Interval(3.0, 4.0)],
            [Interval(1.0, 1.0), Interval(-2.0, -1.0)],
        ], dtype=object)
        got_mm = rigorous_matmul_point_interval(A, BI, self.backend)
        for i in range(A.shape[0]):
            for j in range(BI.shape[1]):
                lo, hi = exact_interval_linear_combination(A[i], BI[:, j])
                self.assert_interval_contains_range(got_mm[i, j], lo, hi)

    def test_interval_matvec_against_exact_extrema(self):
        AI = np.array([
            [Interval(-2.0, 3.0), Interval(1.0, 2.0)],
            [Interval(-1.0e16, -1.0e16), Interval(1.0e16, 1.0e16)],
        ], dtype=object)
        xI = np.array([Interval(-4.0, 5.0), Interval(1.0, 1.0)], dtype=object)
        got = interval_matvec(AI, xI, self.backend)
        for i in range(AI.shape[0]):
            lo, hi = exact_interval_dot(AI[i], xI)
            self.assert_interval_contains_range(got[i], lo, hi)

    def test_nonnegative_upper_reductions_against_exact_rationals(self):
        rng = np.random.default_rng(2026082904)
        for n in (1, 3, 20, 100):
            a = np.abs(rng.normal(size=n))
            b = np.abs(rng.normal(size=n))
            got = nonnegative_dot_upper(a, b, self.backend)
            exact = exact_dot(a, b)
            self.assertTrue(upper_endpoint_ge_exact(got, exact), (n, got, exact))

        A = np.abs(rng.normal(size=(6, 8)))
        x = np.abs(rng.normal(size=8))
        mv = nonnegative_matvec_upper(A, x, self.backend)
        for i in range(A.shape[0]):
            self.assertTrue(upper_endpoint_ge_exact(mv[i], exact_dot(A[i], x)))

        B = np.abs(rng.normal(size=(8, 5)))
        mm = nonnegative_matmul_upper(A, B, self.backend)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                self.assertTrue(upper_endpoint_ge_exact(mm[i, j], exact_dot(A[i], B[:, j])))

        rho = row_sum_norm_upper(A, self.backend)
        exact_rho = max(sum((exact_float(v) for v in row), Fraction(0, 1)) for row in A)
        self.assertTrue(upper_endpoint_ge_exact(rho, exact_rho))

    def test_certificate_matrix_helpers_against_exact_rationals(self):
        C = np.array([[1.0, -2.0], [0.5, 3.0]])
        A = np.array([[2.0, 1.0], [-1.0, 4.0]])
        abs_prod = matrix_product_abs_upper(C, A, self.backend)
        for i in range(2):
            for j in range(2):
                exact = abs(exact_dot(C[i], A[:, j]))
                self.assertTrue(upper_endpoint_ge_exact(abs_prod[i, j], exact))

        ident = identity_minus_product_abs_upper(C, A, self.backend)
        for i in range(2):
            for j in range(2):
                delta = Fraction(1 if i == j else 0, 1)
                exact = abs(delta - exact_dot(C[i], A[:, j]))
                self.assertTrue(upper_endpoint_ge_exact(ident[i, j], exact))

        P = np.array([[1.0, 2.0], [3.0, 4.0]])
        Q = np.array([[0.5, 0.25], [0.125, 0.0625]])
        added = add_nonnegative_matrices_upper(P, Q, self.backend)
        scaled = scale_nonnegative_matrix_upper(0.25, P, self.backend)
        for idx in np.ndindex(P.shape):
            self.assertTrue(upper_endpoint_ge_exact(
                added[idx], exact_float(P[idx]) + exact_float(Q[idx])
            ))
            self.assertTrue(upper_endpoint_ge_exact(
                scaled[idx], exact_float(0.25) * exact_float(P[idx])
            ))


if __name__ == "__main__":
    unittest.main()
