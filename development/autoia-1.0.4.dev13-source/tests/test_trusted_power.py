"""Independent high-precision tests for trusted constant-real power kernels."""

from __future__ import annotations

import random
import unittest

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    real_power_point_interval,
    real_power_interval,
    ElementaryDomainError,
)

try:
    import mpmath as mp
except ImportError:  # pragma: no cover
    mp = None


def mp_exact_float(x: float):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def contains_mp(I, value):
    return mp_exact_float(I.lo) <= value <= mp_exact_float(I.hi)


@unittest.skipIf(mp is None, "mpmath is optional and used only by audit tests")
class TestTrustedPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 140
        cls.backend = RigorousBinary64Backend()

    def test_point_powers_against_high_precision(self):
        cases = [
            (0.25, 0.5), (4.0, 0.5), (0.5, 1.25), (2.0, -0.25),
            (3.0, -2.0), (-2.0, 3.0), (-2.0, -3.0),
        ]
        for x, p in cases:
            with self.subTest(x=x, p=p):
                I = real_power_point_interval(x, p, self.backend)
                exact = mp.power(mp_exact_float(x), mp_exact_float(p))
                self.assertTrue(contains_mp(I, exact), (x, p, I, exact))

    def test_random_positive_point_powers(self):
        rng = random.Random(90210)
        for _ in range(200):
            x = 10.0 ** rng.uniform(-6.0, 6.0)
            p = rng.uniform(-4.0, 4.0)
            if float(p).is_integer():
                p += 0.125
            I = real_power_point_interval(x, p, self.backend)
            exact = mp.power(mp_exact_float(x), mp_exact_float(p))
            self.assertTrue(contains_mp(I, exact), (x, p, I, exact))

    def test_interval_powers_contain_samples(self):
        rng = random.Random(12345)
        for p in [-2.0, -0.5, 0.25, 0.5, 1.25, 2.0, 3.0]:
            X = Interval(0.4, 2.5)
            Y = real_power_interval(X, p, self.backend)
            for _ in range(100):
                x = rng.uniform(X.lo, X.hi)
                exact = mp.power(mp_exact_float(x), mp_exact_float(p))
                self.assertTrue(contains_mp(Y, exact), (p, X, Y, exact))

    def test_integer_negative_base_and_domain_failures(self):
        Y = real_power_interval(Interval(-2.0, -1.0), 2.0, self.backend)
        self.assertLessEqual(Y.lo, 1.0)
        self.assertGreaterEqual(Y.hi, 4.0)
        with self.assertRaises(ElementaryDomainError):
            real_power_interval(Interval(-2.0, -1.0), 0.5, self.backend)
        with self.assertRaises(ElementaryDomainError):
            real_power_interval(Interval(-1.0, 1.0), -1.0, self.backend)


if __name__ == "__main__":
    unittest.main()
