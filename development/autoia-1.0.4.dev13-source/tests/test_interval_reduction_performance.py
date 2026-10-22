"""Step-9b tests for vectorized point×interval and interval×interval reductions."""

from __future__ import annotations

from fractions import Fraction
import unittest
import numpy as np

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    rigorous_dot_point_interval,
    rigorous_dot_interval,
    rigorous_matvec_point_interval,
    rigorous_matmul_point,
    nonnegative_matmul_upper,
    interval_matmul,
)
from tests.support.exact_binary64 import exact_float, lower_endpoint_le_exact, upper_endpoint_ge_exact


def exact_point_interval_dot(a, xI):
    lo = Fraction(0, 1); hi = Fraction(0, 1)
    for ai, I in zip(a, xI):
        aq = exact_float(ai); il = exact_float(I.lo); ih = exact_float(I.hi)
        if aq >= 0:
            lo += aq * il; hi += aq * ih
        else:
            lo += aq * ih; hi += aq * il
    return lo, hi


def exact_interval_dot(aI, bI):
    lo = Fraction(0, 1); hi = Fraction(0, 1)
    for A, B in zip(aI, bI):
        vals = [
            exact_float(x) * exact_float(y)
            for x in (A.lo, A.hi)
            for y in (B.lo, B.hi)
        ]
        lo += min(vals); hi += max(vals)
    return lo, hi


class TestVectorizedIntervalReductions(unittest.TestCase):
    def assert_contains(self, got, lo, hi):
        self.assertTrue(lower_endpoint_le_exact(got.lo, lo), (got, lo, hi))
        self.assertTrue(upper_endpoint_ge_exact(got.hi, hi), (got, lo, hi))

    def test_long_point_interval_dot_contains_exact_extrema(self):
        rng = np.random.default_rng(2026082914)
        backend = RigorousBinary64Backend(reduction_method="vectorized")
        for n in (16, 31, 64, 127, 256):
            for _ in range(10):
                a = rng.normal(size=n)
                left = rng.normal(size=n)
                right = left + np.abs(rng.normal(size=n)) * 0.3
                xI = np.array([Interval(l, h) for l, h in zip(left, right)], dtype=object)
                got = rigorous_dot_point_interval(a, xI, backend)
                self.assert_contains(got, *exact_point_interval_dot(a, xI))
        self.assertGreater(backend.reduction_stats["by_kind"]["dot-point-interval"]["vectorized"], 0)

    def test_long_interval_dot_contains_exact_extrema(self):
        rng = np.random.default_rng(2026082915)
        backend = RigorousBinary64Backend(reduction_method="vectorized")
        for n in (16, 37, 128, 255):
            for _ in range(10):
                al = rng.normal(size=n); ah = al + np.abs(rng.normal(size=n)) * 0.25
                bl = rng.normal(size=n); bh = bl + np.abs(rng.normal(size=n)) * 0.25
                aI = np.array([Interval(l, h) for l, h in zip(al, ah)], dtype=object)
                bI = np.array([Interval(l, h) for l, h in zip(bl, bh)], dtype=object)
                got = rigorous_dot_interval(aI, bI, backend)
                self.assert_contains(got, *exact_interval_dot(aI, bI))
        self.assertGreater(backend.reduction_stats["by_kind"]["dot-interval"]["vectorized"], 0)

    def test_auto_mode_preserves_short_reference_and_switches_long(self):
        backend = RigorousBinary64Backend(reduction_method="auto", vectorized_reduction_min_terms=16)
        short_a = np.arange(8.0)
        short_i = np.array([Interval(-1, 2) for _ in range(8)], dtype=object)
        long_a = np.arange(32.0)
        long_i = np.array([Interval(-1, 2) for _ in range(32)], dtype=object)
        rigorous_dot_point_interval(short_a, short_i, backend)
        rigorous_dot_point_interval(long_a, long_i, backend)
        stats = backend.reduction_stats["by_kind"]["dot-point-interval"]
        self.assertEqual(stats["reference"], 1)
        self.assertEqual(stats["vectorized"], 1)

    def test_matrix_entry_points_use_new_reducers(self):
        rng = np.random.default_rng(2026082916)
        n = 32
        A = rng.normal(size=(3, n))
        xI = np.array([Interval(x - 0.1, x + 0.1) for x in rng.normal(size=n)], dtype=object)
        backend = RigorousBinary64Backend(reduction_method="vectorized")
        got = rigorous_matvec_point_interval(A, xI, backend)
        for i in range(3):
            self.assert_contains(got[i], *exact_point_interval_dot(A[i], xI))
        self.assertEqual(backend.reduction_stats["by_kind"]["dot-point-interval"]["vectorized"], 3)


    def test_batched_point_matrix_product_contains_exact_dots(self):
        rng = np.random.default_rng(2026082918)
        A = rng.normal(size=(5, 32))
        B = rng.normal(size=(32, 4))
        backend = RigorousBinary64Backend(reduction_method="vectorized")
        got = rigorous_matmul_point(A, B, backend)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                exact = sum((exact_float(x)*exact_float(y) for x,y in zip(A[i], B[:,j])), Fraction(0,1))
                self.assertTrue(lower_endpoint_le_exact(got[i,j].lo, exact))
                self.assertTrue(upper_endpoint_ge_exact(got[i,j].hi, exact))

    def test_batched_nonnegative_matrix_product_is_upward(self):
        rng = np.random.default_rng(2026082919)
        A = np.abs(rng.normal(size=(4, 40)))
        B = np.abs(rng.normal(size=(40, 3)))
        backend = RigorousBinary64Backend(reduction_method="vectorized")
        got = nonnegative_matmul_upper(A, B, backend)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                exact = sum((exact_float(x)*exact_float(y) for x,y in zip(A[i], B[:,j])), Fraction(0,1))
                self.assertTrue(upper_endpoint_ge_exact(got[i,j], exact))

    def test_interval_matrix_product_contains_exact_entry_ranges(self):
        rng = np.random.default_rng(2026082917)
        m, n, k = 2, 20, 3
        AI = np.empty((m, n), dtype=object)
        BI = np.empty((n, k), dtype=object)
        for idx in np.ndindex(AI.shape):
            c = float(rng.normal()); r = float(abs(rng.normal()) * 0.05)
            AI[idx] = Interval(c-r, c+r)
        for idx in np.ndindex(BI.shape):
            c = float(rng.normal()); r = float(abs(rng.normal()) * 0.05)
            BI[idx] = Interval(c-r, c+r)
        backend = RigorousBinary64Backend(reduction_method="vectorized")
        got = interval_matmul(AI, BI, backend)
        for i in range(m):
            for j in range(k):
                self.assert_contains(got[i, j], *exact_interval_dot(AI[i], BI[:, j]))


if __name__ == "__main__":
    unittest.main()
