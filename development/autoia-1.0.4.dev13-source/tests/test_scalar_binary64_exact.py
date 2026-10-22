"""Independent exact-rational validation of AutoIA scalar binary64 arithmetic."""

from __future__ import annotations

import math
import struct
import sys
import unittest
import numpy as np

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    audit_binary64_environment,
    integer_interval,
)
from tests.support.exact_binary64 import (
    exact_float,
    exact_binary_op,
    interval_contains_exact,
    exact_interval_extrema,
    interval_contains_exact_range,
)


MAX_FINITE_Q = exact_float(sys.float_info.max)


def finite_float_from_bits(bits: int) -> float:
    return struct.unpack(">d", int(bits).to_bytes(8, "big"))[0]


class TestBinary64Environment(unittest.TestCase):
    def test_environment_audit_passes_on_test_platform(self):
        report = audit_binary64_environment()
        self.assertTrue(report.supported, report.summary())
        self.assertGreaterEqual(len(report.checks), 10)

    def test_trusted_interval_canonicalizes_endpoints_to_binary64_float(self):
        I = Interval(np.float32(1.25e30), 2**500)
        self.assertIs(type(I.lo), float)
        self.assertIs(type(I.hi), float)
        self.assertEqual(I.lo, float(np.float32(1.25e30)))
        self.assertEqual(I.hi, float(2**500))

    def test_integer_interval_contains_exact_integer_across_rounding_boundary(self):
        for n in (-(2**53 + 1), -2**53, -1, 0, 1, 2**53, 2**53 + 1, 10**30):
            I = integer_interval(n)
            from fractions import Fraction
            self.assertTrue(interval_contains_exact(I, Fraction(n, 1)), (n, I))


class TestScalarExactRationalAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = RigorousBinary64Backend()

    def assert_point_operation_sound(self, a: float, b: float, op: str):
        if op == "div" and b == 0.0:
            return
        exact = exact_binary_op(a, b, op)
        x = Interval(a, a)
        y = Interval(b, b)
        try:
            got = getattr(self.backend, op)(x, y)
        except FloatingPointError:
            # The reference kernel deliberately fails closed when the hardware
            # result overflows to infinity.  Such a failure is acceptable only
            # when the exact real result lies outside finite binary64 range.
            self.assertGreater(abs(exact), MAX_FINITE_Q, (op, a, b, exact))
            return
        self.assertTrue(interval_contains_exact(got, exact), (op, a, b, exact, got))

    def test_edge_case_point_operations_against_exact_rationals(self):
        min_sub = float.fromhex("0x0.0000000000001p-1022")
        min_normal = sys.float_info.min
        max_finite = sys.float_info.max
        values = [
            0.0, -0.0,
            min_sub, -min_sub,
            2.0 * min_sub, -2.0 * min_sub,
            float(np.nextafter(min_normal, 0.0)),
            -float(np.nextafter(min_normal, 0.0)),
            min_normal, -min_normal,
            0.5, -0.5, 1.0, -1.0,
            float(np.nextafter(1.0, math.inf)),
            float(np.nextafter(1.0, -math.inf)),
            float(2.0**-500), -float(2.0**-500),
            float(2.0**500), -float(2.0**500),
            float(np.nextafter(max_finite, 0.0)),
            -float(np.nextafter(max_finite, 0.0)),
            max_finite, -max_finite,
        ]
        for a in values:
            for b in values:
                for op in ("add", "sub", "mul", "div"):
                    with self.subTest(a=a.hex(), b=b.hex(), op=op):
                        self.assert_point_operation_sound(a, b, op)

    def test_random_raw_binary64_point_operations_against_exact_rationals(self):
        rng = np.random.default_rng(2026082901)
        values: list[float] = []
        while len(values) < 3000:
            bits = int(rng.integers(0, 2**64, dtype=np.uint64))
            x = finite_float_from_bits(bits)
            if math.isfinite(x):
                values.append(x)

        for _ in range(12000):
            a = values[int(rng.integers(0, len(values)))]
            b = values[int(rng.integers(0, len(values)))]
            op = ("add", "sub", "mul", "div")[int(rng.integers(0, 4))]
            self.assert_point_operation_sound(a, b, op)

    def test_random_interval_operations_against_exact_endpoint_extrema(self):
        rng = np.random.default_rng(2026082902)
        for _ in range(2500):
            aa = np.sort(rng.uniform(-1.0e150, 1.0e150, size=2))
            bb = np.sort(rng.uniform(-1.0e150, 1.0e150, size=2))
            x = Interval(float(aa[0]), float(aa[1]))
            y = Interval(float(bb[0]), float(bb[1]))
            for op in ("add", "sub", "mul"):
                got = getattr(self.backend, op)(x, y)
                lo, hi = exact_interval_extrema(x, y, op)
                self.assertTrue(interval_contains_exact_range(got, lo, hi), (op, x, y, got))
            if not (y.lo <= 0.0 <= y.hi):
                got = self.backend.div(x, y)
                lo, hi = exact_interval_extrema(x, y, "div")
                self.assertTrue(interval_contains_exact_range(got, lo, hi), ("div", x, y, got))

    def test_signed_zero_and_underflow_neighborhood(self):
        min_sub = float.fromhex("0x0.0000000000001p-1022")
        cases = [
            (min_sub, min_sub, "mul"),
            (min_sub, sys.float_info.max, "div"),
            (min_sub, 2.0, "div"),
            (-min_sub, 2.0, "div"),
            (min_sub, -min_sub, "add"),
            (min_sub, min_sub, "sub"),
            (sys.float_info.min, 0.5, "mul"),
        ]
        for a, b, op in cases:
            self.assert_point_operation_sound(a, b, op)

    def test_division_by_interval_containing_zero_fails_closed(self):
        with self.assertRaises(ZeroDivisionError):
            self.backend.div(Interval(1.0, 2.0), Interval(-0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
