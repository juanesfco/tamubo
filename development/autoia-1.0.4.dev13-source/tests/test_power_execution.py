"""End-to-end tests for constant real powers."""

from __future__ import annotations

import math
import unittest
import numpy as np

from autoia import autoia, Interval, CertificationError, UnsupportedOperationError
from numpy import power as direct_numpy_power
from math import pow as direct_math_power


class TestPowerExecution(unittest.TestCase):
    def _grid(self, exponent, box, representation="polynomial", order=8):
        def model(p):
            return p[0] ** exponent
        R = autoia(model, [box], order=order, representation=representation)
        for x in np.linspace(box.lo, box.hi, 501):
            y = float(x) ** float(exponent)
            self.assertLessEqual(R.lower[0], y)
            self.assertGreaterEqual(R.upper[0], y)
        return R

    def test_negative_integer_powers(self):
        for exponent, box in [(-1, Interval(1.0, 2.0)), (-2, Interval(-2.0, -1.0)), (-3, Interval(1.0, 2.0))]:
            with self.subTest(exponent=exponent):
                self._grid(exponent, box)
                self._grid(exponent, box, representation="interval")

    def test_fractional_positive_and_negative_powers(self):
        for exponent in [0.1, 0.25, 0.5, 0.75, 1.25, 1.5, -0.1, -0.25, -0.5, -1.25]:
            with self.subTest(exponent=exponent):
                R = self._grid(exponent, Interval(0.5, 2.0))
                self.assertTrue(np.isfinite(R.width[0]))
                self._grid(exponent, Interval(0.5, 2.0), representation="interval")

    def test_half_power_reuses_sqrt_quality(self):
        R = self._grid(0.5, Interval(0.25, 4.0), order=8)
        self.assertLess(R.width[0], 1.7)

    def test_interval_even_power_preserves_same_base_dependency(self):
        def model(p):
            return p[0] ** 2
        R = autoia(model, [Interval(-1.0, 1.0)], representation="interval")
        self.assertLessEqual(R.lower[0], 0.0)
        self.assertGreaterEqual(R.upper[0], 1.0)
        self.assertGreaterEqual(R.lower[0], -1e-15)

    def test_numpy_power_and_alias(self):
        def model_module(p):
            return np.power(p[0], 1.25)
        def model_alias(p):
            return direct_numpy_power(p[0], -0.25)
        R1 = autoia(model_module, [Interval(0.5, 2.0)], order=8)
        R2 = autoia(model_alias, [Interval(0.5, 2.0)], order=8)
        for x in np.linspace(0.5, 2.0, 101):
            self.assertLessEqual(R1.lower[0], x**1.25)
            self.assertGreaterEqual(R1.upper[0], x**1.25)
            self.assertLessEqual(R2.lower[0], x**-0.25)
            self.assertGreaterEqual(R2.upper[0], x**-0.25)

    def test_math_pow_and_alias(self):
        def model_module(p):
            return math.pow(p[0], 1.25)
        def model_alias(p):
            return direct_math_power(p[0], -0.25)
        R1 = autoia(model_module, [Interval(0.5, 2.0)], order=8)
        R2 = autoia(model_alias, [Interval(0.5, 2.0)], order=8)
        for x in np.linspace(0.5, 2.0, 101):
            self.assertLessEqual(R1.lower[0], x**1.25)
            self.assertGreaterEqual(R1.upper[0], x**1.25)
            self.assertLessEqual(R2.lower[0], x**-0.25)
            self.assertGreaterEqual(R2.upper[0], x**-0.25)

    def test_domain_failures(self):
        def frac_negative_base(p):
            return p[0] ** 0.5
        def negative_power_zero(p):
            return p[0] ** -2
        with self.assertRaises(CertificationError):
            autoia(frac_negative_base, [Interval(-2.0, -1.0)])
        with self.assertRaises(CertificationError):
            autoia(negative_power_zero, [Interval(-1.0, 1.0)])

    def test_uncertain_exponent_unsupported(self):
        def model(p):
            return p[0] ** p[1]
        with self.assertRaises(UnsupportedOperationError):
            autoia(model, [Interval(1.0, 2.0), Interval(0.2, 0.4)])

    def test_power_dispatch_does_not_mutate_numpy_or_math(self):
        np_power = np.power
        math_power = math.pow
        def model(p):
            return np.power(p[0], 0.75) + math.pow(p[0], -0.25)
        autoia(model, [Interval(0.8, 1.2)], order=6)
        self.assertIs(np.power, np_power)
        self.assertIs(math.pow, math_power)


if __name__ == "__main__":
    unittest.main()
