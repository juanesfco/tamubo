"""Version-1 adversarial proof-hardening regression tests.

These tests emphasize binary64 lattice boundaries and cross-layer situations that
are easy to miss with ordinary random data.  They are deterministic and do not
form part of the proof itself; they independently check the implementation's
proof obligations and fail-closed behavior.
"""

from __future__ import annotations

from fractions import Fraction
import math
import sys
import unittest

import numpy as np

from autoia import autoia, Interval, CertificationError
from autoia.trusted import (
    RigorousBinary64Backend,
    audit_binary64_environment,
    rigorous_dot_point,
    sqrt_point_interval,
    log_point_interval,
)
from tests.support.exact_binary64 import (
    exact_binary_op,
    exact_float,
    interval_contains_exact,
    lower_endpoint_le_exact,
    upper_endpoint_ge_exact,
)

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - optional audit dependency
    mp = None


MAX_FINITE_Q = exact_float(sys.float_info.max)
MIN_SUB = float.fromhex("0x0.0000000000001p-1022")


def _critical_binary64_values() -> list[float]:
    vals = {
        0.0,
        -0.0,
        MIN_SUB,
        -MIN_SUB,
        2.0 * MIN_SUB,
        -2.0 * MIN_SUB,
        float(np.nextafter(sys.float_info.min, 0.0)),
        -float(np.nextafter(sys.float_info.min, 0.0)),
        sys.float_info.min,
        -sys.float_info.min,
        0.5,
        -0.5,
        1.0,
        -1.0,
        float(np.nextafter(1.0, -math.inf)),
        float(np.nextafter(1.0, math.inf)),
        sys.float_info.max,
        -sys.float_info.max,
        float(np.nextafter(sys.float_info.max, 0.0)),
        -float(np.nextafter(sys.float_info.max, 0.0)),
    }
    for exponent in (-1000, -500, -1, 0, 1, 500, 1000):
        x = math.ldexp(1.0, exponent)
        vals.update((x, -x, np.nextafter(x, 0.0), np.nextafter(x, math.inf)))
    return sorted(float(x) for x in vals)


def _exact_reciprocal_interval(lo: float, hi: float) -> tuple[Fraction, Fraction]:
    """Exact range of 1/x on a positive binary64 interval."""
    lo_q = exact_float(lo)
    hi_q = exact_float(hi)
    return Fraction(1, 1) / hi_q, Fraction(1, 1) / lo_q


class TestReleaseHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = RigorousBinary64Backend()

    def test_environment_exercises_strengthened_cross_platform_contract(self):
        report = audit_binary64_environment()
        self.assertTrue(report.supported, report.summary())
        names = {name for name, _, _ in report.checks}
        self.assertIn("binary64-division-rounding", names)
        self.assertIn("subnormal-tie-to-even", names)

    def test_critical_binary64_lattice_scalar_operations(self):
        values = _critical_binary64_values()
        for a in values:
            for b in values:
                for op in ("add", "sub", "mul", "div"):
                    if op == "div" and b == 0.0:
                        continue
                    exact = exact_binary_op(a, b, op)
                    try:
                        got = getattr(self.backend, op)(Interval(a, a), Interval(b, b))
                    except FloatingPointError:
                        self.assertGreater(abs(exact), MAX_FINITE_Q, (op, a, b, exact))
                        continue
                    self.assertTrue(
                        interval_contains_exact(got, exact),
                        (op, a.hex(), b.hex(), exact, got),
                    )

    def test_reduction_threshold_and_subnormal_cancellation_cases(self):
        for n in (15, 16, 17, 31, 32, 33, 63, 64, 65):
            # Alternating large terms plus one tiny residual stresses both
            # cancellation and every pairwise-tree shape around powers of two.
            a = np.empty(n, dtype=float)
            a[:-1] = np.resize(np.array([2.0**500, -(2.0**500)], dtype=float), n - 1)
            a[-1] = 2.0**-1000
            b = np.ones(n, dtype=float)
            exact = sum((exact_float(x) for x in a), Fraction(0, 1))
            for mode in ("reference", "vectorized", "auto"):
                backend = RigorousBinary64Backend(reduction_method=mode)
                got = rigorous_dot_point(a, b, backend)
                self.assertTrue(interval_contains_exact(got, exact), (n, mode, got, exact))

            # Every exact product is half a minimum subnormal and rounds to
            # zero in binary64; the outward product intervals must still retain
            # the exact positive sum.
            a = np.full(n, MIN_SUB, dtype=float)
            b = np.full(n, 0.5, dtype=float)
            exact = Fraction(n, 2) * exact_float(MIN_SUB)
            for mode in ("reference", "vectorized", "auto"):
                backend = RigorousBinary64Backend(reduction_method=mode)
                got = rigorous_dot_point(a, b, backend)
                self.assertTrue(interval_contains_exact(got, exact), (n, mode, got, exact))

    def test_extreme_sqrt_endpoints_satisfy_exact_square_inequalities(self):
        for x in (MIN_SUB, sys.float_info.min, sys.float_info.max):
            got = sqrt_point_interval(x, self.backend)
            xq = exact_float(x)
            loq = exact_float(got.lo)
            hiq = exact_float(got.hi)
            self.assertLessEqual(loq * loq, xq, (x, got))
            self.assertGreaterEqual(hiq * hiq, xq, (x, got))

    @unittest.skipIf(mp is None, "mpmath is optional and used only by audit tests")
    def test_sqrt_and_log_extreme_finite_binary64_points(self):
        mp.mp.dps = 180
        values = [
            MIN_SUB,
            float(np.nextafter(sys.float_info.min, 0.0)),
            sys.float_info.min,
            1.0,
            2.0,
            float(np.nextafter(sys.float_info.max, 0.0)),
            sys.float_info.max,
        ]
        for x in values:
            x_mp = mp.mpf(x.as_integer_ratio()[0]) / mp.mpf(x.as_integer_ratio()[1])
            sqrt_i = sqrt_point_interval(x, self.backend)
            sqrt_exact = mp.sqrt(x_mp)
            self.assertLessEqual(mp.mpf(sqrt_i.lo.as_integer_ratio()[0]) / sqrt_i.lo.as_integer_ratio()[1], sqrt_exact)
            self.assertGreaterEqual(mp.mpf(sqrt_i.hi.as_integer_ratio()[0]) / sqrt_i.hi.as_integer_ratio()[1], sqrt_exact)
            log_i = log_point_interval(x, self.backend)
            log_exact = mp.log(x_mp)
            self.assertLessEqual(mp.mpf(log_i.lo.as_integer_ratio()[0]) / log_i.lo.as_integer_ratio()[1], log_exact)
            self.assertGreaterEqual(mp.mpf(log_i.hi.as_integer_ratio()[0]) / log_i.hi.as_integer_ratio()[1], log_exact)

    def test_near_singular_diagonal_semantic_solve_has_exact_analytic_containment(self):
        # delta and radius are exact powers of two, so the mathematical range of
        # the second component is exactly [1/(delta+r), 1/(delta-r)].
        for exponent in (20, 30, 40, 50):
            delta = 2.0 ** (-exponent)
            radius = 2.0 ** (-(exponent + 2))

            def model(p, delta=delta):
                x = p[0]
                A = np.array([[1.0, 0.0], [0.0, delta + x]], dtype=object)
                b = np.array([1.0, 1.0], dtype=object)
                return np.linalg.solve(A, b)

            result = autoia(model, [Interval(-radius, radius)], order=4)
            exact_lo, exact_hi = _exact_reciprocal_interval(delta - radius, delta + radius)
            self.assertTrue(lower_endpoint_le_exact(result.lower[1], exact_lo))
            self.assertTrue(upper_endpoint_ge_exact(result.upper[1], exact_hi))
            self.assertTrue(result.machine_rigorous)

    def test_near_singular_zero_crossing_solve_fails_closed(self):
        delta = 2.0 ** -30

        def model(p):
            x = p[0]
            A = np.array([[1.0, 0.0], [0.0, delta + x]], dtype=object)
            return np.linalg.solve(A, np.array([1.0, 1.0], dtype=object))

        with self.assertRaises(CertificationError):
            autoia(model, [Interval(-1.1 * delta, 0.1 * delta)], order=4)

    def test_semantic_and_generic_diagonal_solve_agree_on_adversarial_family(self):
        delta = 2.0 ** -30
        radius = 2.0 ** -32

        def semantic(p):
            x = p[0]
            A = np.array([[1.0, 0.0], [0.0, delta + x]], dtype=object)
            return np.linalg.solve(A, np.array([1.0, 1.0], dtype=object))

        def generic(p):
            x = p[0]
            return np.array([1.0, 1.0 / (delta + x)], dtype=object)

        box = [Interval(-radius, radius)]
        a = autoia(semantic, box, order=4)
        b = autoia(generic, box, order=4)
        # Different proof paths are allowed to reorder outward arithmetic, so
        # bit identity is deliberately *not* required.  Both certificates must
        # contain the same exact analytic range, and their reported intervals
        # must overlap.
        exact_lo, exact_hi = _exact_reciprocal_interval(delta - radius, delta + radius)
        for result in (a, b):
            self.assertTrue(lower_endpoint_le_exact(result.lower[1], exact_lo))
            self.assertTrue(upper_endpoint_ge_exact(result.upper[1], exact_hi))
        self.assertLessEqual(max(a.lower[1], b.lower[1]), min(a.upper[1], b.upper[1]))
        self.assertEqual(len(a.certificates), 1)
        self.assertEqual(len(b.certificates), 0)

    def test_branch_boundary_one_ulp_neighborhood_is_sound(self):
        left = float(np.nextafter(1.0, -math.inf))
        right = float(np.nextafter(1.0, math.inf))

        def model(p):
            x = p[0]
            if x <= 1.0:
                return x - 1.0
            return 2.0 * (x - 1.0)

        result = autoia(model, [Interval(left, right)], order=3)
        exact_left = exact_float(left) - Fraction(1, 1)
        exact_right = 2 * (exact_float(right) - Fraction(1, 1))
        self.assertTrue(lower_endpoint_le_exact(result.lower[0], exact_left))
        self.assertTrue(upper_endpoint_ge_exact(result.upper[0], exact_right))

    def test_domain_boundary_nextafter_neighborhoods(self):
        tiny = MIN_SUB

        def sqrt_model(p):
            return np.sqrt(p[0])

        sqrt_result = autoia(sqrt_model, [Interval(0.0, tiny)], order=4)
        self.assertTrue(sqrt_result.machine_rigorous)
        self.assertLessEqual(sqrt_result.lower[0], 0.0)
        self.assertGreaterEqual(sqrt_result.upper[0], 0.0)

        def log_model(p):
            return np.log(p[0])

        # Use the interval representation here so this test isolates strict
        # domain admissibility at the smallest positive binary64 value rather
        # than high-order Taylor coefficients of log near zero.
        positive = autoia(
            log_model, [Interval(tiny, sys.float_info.min)],
            order=4, representation="interval",
        )
        self.assertTrue(positive.machine_rigorous)
        with self.assertRaises(CertificationError):
            autoia(log_model, [Interval(0.0, tiny)], order=4)


if __name__ == "__main__":
    unittest.main()
