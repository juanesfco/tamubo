"""Regression tests for the Step-1 trusted-kernel extraction.

These tests compare the extracted implementation directly against the verbatim
Version 0.11 oracle snapshot.  During this refactor checkpoint we require exact
binary64 endpoint identity, not merely mathematical inclusion.
"""

from __future__ import annotations

import unittest
import numpy as np

from tests.support import oracle_v11 as oracle
from autoia.trusted import (
    Interval,
    point_interval,
    integer_interval,
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


def assert_float_bits_equal(testcase: unittest.TestCase, a: float, b: float):
    testcase.assertEqual(float(a).hex(), float(b).hex())


def assert_interval_bits_equal(testcase, new, old):
    assert_float_bits_equal(testcase, new.lo, old.lo)
    assert_float_bits_equal(testcase, new.hi, old.hi)


def old_interval(x):
    return oracle.Interval(float(x.lo), float(x.hi))


def old_interval_array(arr):
    arr = np.asarray(arr, dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        out[idx] = old_interval(arr[idx])
    return out


class TestTrustedKernelRegression(unittest.TestCase):
    def setUp(self):
        self.new = RigorousBinary64Backend()
        self.old = oracle.RigorousBinary64Backend()
        self.rng = np.random.default_rng(20260829)

    def test_scalar_binary64_backend_is_bit_identical_to_oracle(self):
        # Include exact identities plus randomized finite intervals.
        cases = [
            (Interval(0.0, 0.0), Interval(1.0, 1.0)),
            (Interval(-1.0, -1.0), Interval(-3.0, 4.0)),
            (Interval(2.0, 2.0), Interval(0.5, 0.5)),
            (Interval(-3.0, -2.0), Interval(4.0, 5.0)),
        ]
        for _ in range(250):
            a = np.sort(self.rng.uniform(-10.0, 10.0, size=2))
            b = np.sort(self.rng.uniform(-10.0, 10.0, size=2))
            cases.append((Interval(a[0], a[1]), Interval(b[0], b[1])))

        for x, y in cases:
            ox, oy = old_interval(x), old_interval(y)
            for name in ("add", "sub", "mul"):
                assert_interval_bits_equal(
                    self,
                    getattr(self.new, name)(x, y),
                    getattr(self.old, name)(ox, oy),
                )
            if not (y.lo <= 0.0 <= y.hi):
                assert_interval_bits_equal(
                    self,
                    self.new.div(x, y),
                    self.old.div(ox, oy),
                )

    def test_integer_interval_is_bit_identical_to_oracle(self):
        values = [0, 1, -1, 2**53, 2**53 + 1, -(2**53 + 1), 10**30]
        for n in values:
            assert_interval_bits_equal(self, integer_interval(n), oracle.integer_interval(n))

    def test_point_reductions_are_bit_identical_to_oracle(self):
        for n in (1, 2, 5, 9):
            a = self.rng.normal(size=n)
            b = self.rng.normal(size=n)
            assert_interval_bits_equal(
                self,
                rigorous_dot_point(a, b, self.new),
                oracle.rigorous_dot_point(a, b, self.old),
            )

        A = self.rng.normal(size=(4, 5))
        B = self.rng.normal(size=(5, 3))
        x = self.rng.normal(size=5)
        new_mm = rigorous_matmul_point(A, B, self.new)
        old_mm = oracle.rigorous_matmul_point(A, B, self.old)
        for idx in np.ndindex(new_mm.shape):
            assert_interval_bits_equal(self, new_mm[idx], old_mm[idx])
        new_mv = rigorous_matvec_point(A, x, self.new)
        old_mv = oracle.rigorous_matvec_point(A, x, self.old)
        for idx in np.ndindex(new_mv.shape):
            assert_interval_bits_equal(self, new_mv[idx], old_mv[idx])

    def test_interval_linear_algebra_is_bit_identical_to_oracle(self):
        A = self.rng.normal(size=(3, 4))
        xI = np.empty(4, dtype=object)
        for i in range(4):
            c = self.rng.normal()
            r = abs(self.rng.normal()) * 0.1
            xI[i] = Interval(c-r, c+r)
        BI = np.empty((4, 2), dtype=object)
        for idx in np.ndindex(BI.shape):
            c = self.rng.normal()
            r = abs(self.rng.normal()) * 0.1
            BI[idx] = Interval(c-r, c+r)

        old_xI = old_interval_array(xI)
        old_BI = old_interval_array(BI)

        new_mv = rigorous_matvec_point_interval(A, xI, self.new)
        old_mv = oracle.rigorous_matvec_point_interval(A, old_xI, self.old)
        for idx in np.ndindex(new_mv.shape):
            assert_interval_bits_equal(self, new_mv[idx], old_mv[idx])

        new_mm = rigorous_matmul_point_interval(A, BI, self.new)
        old_mm = oracle.rigorous_matmul_point_interval(A, old_BI, self.old)
        for idx in np.ndindex(new_mm.shape):
            assert_interval_bits_equal(self, new_mm[idx], old_mm[idx])

        AI = np.empty((3, 4), dtype=object)
        for idx in np.ndindex(AI.shape):
            c = self.rng.normal()
            r = abs(self.rng.normal()) * 0.05
            AI[idx] = Interval(c-r, c+r)
        old_AI = old_interval_array(AI)
        new_imv = interval_matvec(AI, xI, self.new)
        old_imv = oracle.interval_matvec(old_AI, old_xI, self.old)
        for idx in np.ndindex(new_imv.shape):
            assert_interval_bits_equal(self, new_imv[idx], old_imv[idx])

    def test_nonnegative_reductions_are_bit_identical_to_oracle(self):
        A = np.abs(self.rng.normal(size=(4, 5)))
        B = np.abs(self.rng.normal(size=(5, 3)))
        x = np.abs(self.rng.normal(size=5))

        assert_float_bits_equal(
            self,
            nonnegative_dot_upper(A[0], x, self.new),
            oracle.nonnegative_dot_upper(A[0], x, self.old),
        )
        np.testing.assert_array_equal(
            nonnegative_matvec_upper(A, x, self.new).view(np.uint64),
            oracle.nonnegative_matvec_upper(A, x, self.old).view(np.uint64),
        )
        np.testing.assert_array_equal(
            nonnegative_matmul_upper(A, B, self.new).view(np.uint64),
            oracle.nonnegative_matmul_upper(A, B, self.old).view(np.uint64),
        )
        assert_float_bits_equal(
            self,
            row_sum_norm_upper(A, self.new),
            oracle.row_sum_norm_upper(A, self.old),
        )

    def test_certificate_matrix_helpers_are_bit_identical_to_oracle(self):
        C = self.rng.normal(size=(3, 3))
        A = self.rng.normal(size=(3, 3))
        P = np.abs(self.rng.normal(size=(3, 3)))
        Q = np.abs(self.rng.normal(size=(3, 3)))

        np.testing.assert_array_equal(
            matrix_product_abs_upper(C, A, self.new).view(np.uint64),
            oracle.matrix_product_abs_upper(C, A, self.old).view(np.uint64),
        )
        np.testing.assert_array_equal(
            identity_minus_product_abs_upper(C, A, self.new).view(np.uint64),
            oracle.identity_minus_product_abs_upper(C, A, self.old).view(np.uint64),
        )
        np.testing.assert_array_equal(
            add_nonnegative_matrices_upper(P, Q, self.new).view(np.uint64),
            oracle.add_nonnegative_matrices_upper(P, Q, self.old).view(np.uint64),
        )
        np.testing.assert_array_equal(
            scale_nonnegative_matrix_upper(0.25, P, self.new).view(np.uint64),
            oracle.scale_nonnegative_matrix_upper(0.25, P, self.old).view(np.uint64),
        )


if __name__ == "__main__":
    unittest.main()
