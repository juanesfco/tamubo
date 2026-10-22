import unittest
from fractions import Fraction

import mpmath as mp
import numpy as np

from autoia.trusted import (
    RigorousBinary64Backend,
    exp_point_interval,
    exp_point_interval_arrays,
    exp_interval_arrays,
    sqrt_point_interval_arrays,
    rigorous_interval_add_arrays,
    rigorous_interval_subtract_arrays,
    rigorous_interval_multiply_arrays,
    rigorous_interval_divide_point_arrays,
)


class TestTrustedArrayElementary(unittest.TestCase):
    def setUp(self):
        self.backend = RigorousBinary64Backend(reduction_method="vectorized")

    def test_elementwise_arithmetic_exact_rational_points(self):
        a = np.array([0.1, -2.0, 3.5, 2.0**-1022])
        b = np.array([0.2, 0.5, -1.25, 2.0])
        alo = ahi = a
        blo = bhi = b
        funcs = [
            (rigorous_interval_add_arrays, lambda x,y: x+y),
            (rigorous_interval_subtract_arrays, lambda x,y: x-y),
            (rigorous_interval_multiply_arrays, lambda x,y: x*y),
        ]
        for fn, exactop in funcs:
            lo, hi = fn(alo, ahi, blo, bhi, self.backend)
            for i in range(a.size):
                exact = exactop(Fraction.from_float(float(a[i])), Fraction.from_float(float(b[i])))
                self.assertLessEqual(Fraction.from_float(float(lo[i])), exact)
                self.assertGreaterEqual(Fraction.from_float(float(hi[i])), exact)
        lo, hi = rigorous_interval_divide_point_arrays(alo, ahi, b, self.backend)
        for i in range(a.size):
            exact = Fraction.from_float(float(a[i])) / Fraction.from_float(float(b[i]))
            self.assertLessEqual(Fraction.from_float(float(lo[i])), exact)
            self.assertGreaterEqual(Fraction.from_float(float(hi[i])), exact)

    def test_exp_point_arrays_high_precision(self):
        rng = np.random.default_rng(19231)
        x = np.concatenate((
            np.array([-100.0, -20.0, -1.0, -0.125, -0.0, 0.0, 0.125, 1.0, 20.0, 100.0]),
            rng.uniform(-40.0, 40.0, size=150),
        ))
        lo, hi = exp_point_interval_arrays(x, self.backend)
        mp.mp.dps = 100
        for xx, ll, hh in zip(x, lo, hi):
            exact = mp.exp(mp.mpf(float(xx)))
            self.assertLessEqual(mp.mpf(float(ll)), exact)
            self.assertGreaterEqual(mp.mpf(float(hh)), exact)

    def test_exp_interval_arrays_endpoint_containment(self):
        rng = np.random.default_rng(9917)
        centers = rng.uniform(-10.0, 10.0, size=80)
        widths = rng.uniform(0.0, 0.2, size=80)
        lo_in = centers - widths
        hi_in = centers + widths
        lo, hi = exp_interval_arrays(lo_in, hi_in, self.backend)
        mp.mp.dps = 100
        for a, b, ll, hh in zip(lo_in, hi_in, lo, hi):
            self.assertLessEqual(mp.mpf(float(ll)), mp.exp(mp.mpf(float(a))))
            self.assertGreaterEqual(mp.mpf(float(hh)), mp.exp(mp.mpf(float(b))))

    def test_sqrt_point_arrays_high_precision(self):
        vals = np.array([0.0, np.nextafter(0.0, 1.0), np.finfo(float).tiny, 0.1, 1.0, 2.0, 1e100, np.finfo(float).max])
        lo, hi = sqrt_point_interval_arrays(vals, self.backend)
        mp.mp.dps = 120
        for x, ll, hh in zip(vals, lo, hi):
            exact = mp.sqrt(mp.mpf(float(x)))
            self.assertLessEqual(mp.mpf(float(ll)), exact)
            self.assertGreaterEqual(mp.mpf(float(hh)), exact)


    def test_integer_cache_does_not_change_scalar_exp(self):
        # Regression intent: repeated calls on one backend reuse constants but
        # must remain bitwise stable.
        a = [exp_point_interval(x, self.backend) for x in (-3.0, -0.1, 0.1, 3.0)]
        cache_size = len(self.backend._integer_interval_cache)
        b = [exp_point_interval(x, self.backend) for x in (-3.0, -0.1, 0.1, 3.0)]
        self.assertGreater(cache_size, 0)
        self.assertEqual(a, b)
        self.assertEqual(cache_size, len(self.backend._integer_interval_cache))


if __name__ == "__main__":
    unittest.main()
