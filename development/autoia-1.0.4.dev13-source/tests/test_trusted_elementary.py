"""Independent high-precision audits for trusted elementary scalar kernels."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    sqrt_point_interval,
    sqrt_interval,
    exp_point_interval,
    exp_interval,
    log_point_interval,
    log_interval,
    log10_point_interval,
    log10_interval,
    ln10_interval,
    sin_interval,
    cos_interval,
    ln2_interval,
    pi_interval,
)

try:
    import mpmath as mp
except ImportError:  # pragma: no cover
    mp = None


def mp_exact_float(x: float):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def contains_mp(I: Interval, value) -> bool:
    return mp_exact_float(I.lo) <= value <= mp_exact_float(I.hi)


@unittest.skipIf(mp is None, "mpmath is optional and used only by the audit tests")
class TestTrustedElementary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 160
        cls.backend = RigorousBinary64Backend()

    def test_pi_and_ln2_constants_contain_high_precision_values(self):
        self.assertTrue(contains_mp(pi_interval(self.backend), mp.pi))
        self.assertTrue(contains_mp(ln2_interval(self.backend), mp.log(2)))
        self.assertTrue(contains_mp(ln10_interval(self.backend), mp.log(10)))

    def test_edge_point_values(self):
        cases = {
            "sqrt": (
                sqrt_point_interval,
                mp.sqrt,
                [0.0, np.nextafter(0.0, 1.0), 1e-320, 1e-300, 0.1, 1.0, 2.0, 1e300],
            ),
            "exp": (
                exp_point_interval,
                mp.exp,
                [-700.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 700.0],
            ),
            "log": (
                log_point_interval,
                mp.log,
                [np.nextafter(0.0, 1.0), 1e-300, 0.1, 1.0, 2.0, 10.0, 1e300],
            ),
            "log10": (
                log10_point_interval,
                mp.log10,
                [np.nextafter(0.0, 1.0), 1e-300, 0.1, 1.0, 2.0, 10.0, 1e300],
            ),
            "sin": (
                lambda x, b: sin_interval(Interval(x, x), b),
                mp.sin,
                [-1e6, -1000.0, -10.0, -1.0, 0.0, 0.1, 1.0, 10.0, 1000.0, 1e6],
            ),
            "cos": (
                lambda x, b: cos_interval(Interval(x, x), b),
                mp.cos,
                [-1e6, -1000.0, -10.0, -1.0, 0.0, 0.1, 1.0, 10.0, 1000.0, 1e6],
            ),
        }
        for name, (fn, exact_fn, xs) in cases.items():
            for x in xs:
                with self.subTest(function=name, x=x):
                    I = fn(float(x), self.backend)
                    exact = exact_fn(mp_exact_float(float(x)))
                    self.assertTrue(contains_mp(I, exact), (name, x, I, exact))

    def test_random_points_against_high_precision(self):
        rng = random.Random(20260829)
        for _ in range(200):
            # sqrt/log use a broad exponent distribution.
            exponent = rng.randint(-1000, 900)
            mantissa = rng.uniform(1.0, 2.0)
            x_pos = math.ldexp(mantissa, exponent)
            if math.isfinite(x_pos) and x_pos > 0.0:
                Is = sqrt_point_interval(x_pos, self.backend)
                self.assertTrue(contains_mp(Is, mp.sqrt(mp_exact_float(x_pos))))
                Il = log_point_interval(x_pos, self.backend)
                self.assertTrue(contains_mp(Il, mp.log(mp_exact_float(x_pos))))
                Il10 = log10_point_interval(x_pos, self.backend)
                self.assertTrue(contains_mp(Il10, mp.log10(mp_exact_float(x_pos))))

            x_exp = rng.uniform(-700.0, 700.0)
            Ie = exp_point_interval(x_exp, self.backend)
            self.assertTrue(contains_mp(Ie, mp.exp(mp_exact_float(x_exp))))

            x_trig = rng.uniform(-1e5, 1e5)
            Isin = sin_interval(Interval(x_trig, x_trig), self.backend)
            Icos = cos_interval(Interval(x_trig, x_trig), self.backend)
            xx = mp_exact_float(x_trig)
            self.assertTrue(contains_mp(Isin, mp.sin(xx)))
            self.assertTrue(contains_mp(Icos, mp.cos(xx)))

    def test_interval_extensions_contain_sampled_values(self):
        rng = random.Random(1234)
        specs = [
            (sqrt_interval, mp.sqrt, Interval(0.01, 7.0)),
            (exp_interval, mp.exp, Interval(-3.0, 2.0)),
            (log_interval, mp.log, Interval(0.02, 8.0)),
            (log10_interval, mp.log10, Interval(0.02, 8.0)),
            (sin_interval, mp.sin, Interval(-2.2, -1.8)),
            (cos_interval, mp.cos, Interval(0.1, 0.6)),
        ]
        for fn, exact_fn, X in specs:
            Y = fn(X, self.backend)
            for _ in range(100):
                x = rng.uniform(X.lo, X.hi)
                self.assertTrue(contains_mp(Y, exact_fn(mp_exact_float(x))))


if __name__ == "__main__":
    unittest.main()
