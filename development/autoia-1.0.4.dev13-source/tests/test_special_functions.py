import math
import unittest

import numpy as np

from autoia import (
    Interval,
    autoia,
    erf,
    standard_normal_pdf,
    standard_normal_cdf,
    normal_pdf,
    normal_cdf,
)
from autoia.trusted import (
    RigorousBinary64Backend,
    erf_point_interval,
    erf_interval,
    standard_normal_pdf_point_interval,
    standard_normal_cdf_point_interval,
)


SQRT_2PI = math.sqrt(2.0 * math.pi)


def phi(x):
    return math.exp(-0.5 * x * x) / SQRT_2PI


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class TestTrustedSpecialFunctions(unittest.TestCase):
    def setUp(self):
        self.backend = RigorousBinary64Backend()

    def test_erf_points_contain_reference(self):
        for x in (-10.0, -5.0, -4.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 4.0, 5.0, 10.0):
            I = erf_point_interval(x, self.backend)
            self.assertLessEqual(I.lo, math.erf(x))
            self.assertGreaterEqual(I.hi, math.erf(x))
            self.assertGreaterEqual(I.lo, -1.000000000000001)
            self.assertLessEqual(I.hi, 1.000000000000001)

    def test_erf_interval_is_monotone_extension(self):
        I = erf_interval(Interval(-1.25, 0.8), self.backend)
        xs = np.linspace(-1.25, 0.8, 501)
        vals = [math.erf(float(x)) for x in xs]
        self.assertLessEqual(I.lo, min(vals))
        self.assertGreaterEqual(I.hi, max(vals))

    def test_normal_point_functions_contain_reference(self):
        for x in (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0):
            P = standard_normal_pdf_point_interval(x, self.backend)
            C = standard_normal_cdf_point_interval(x, self.backend)
            self.assertLessEqual(P.lo, phi(x))
            self.assertGreaterEqual(P.hi, phi(x))
            self.assertLessEqual(C.lo, Phi(x))
            self.assertGreaterEqual(C.hi, Phi(x))


class TestSpecialExecution(unittest.TestCase):
    def _check_dense(self, func, truth, box, order=8):
        def model(p):
            return np.array([func(p[0])], dtype=object)
        result = autoia(model, [Interval(*box)], order=order)
        xs = np.linspace(box[0], box[1], 1001)
        vals = [truth(float(x)) for x in xs]
        self.assertLessEqual(float(result.lower[0]), min(vals))
        self.assertGreaterEqual(float(result.upper[0]), max(vals))
        self.assertTrue(result.machine_rigorous)

    def test_erf_polynomial_dense_containment(self):
        self._check_dense(erf, math.erf, (-1.0, 1.0))

    def test_standard_normal_pdf_polynomial_dense_containment(self):
        self._check_dense(standard_normal_pdf, phi, (-2.0, 1.5))

    def test_standard_normal_cdf_polynomial_dense_containment(self):
        self._check_dense(standard_normal_cdf, Phi, (-2.0, 1.5))

    def test_math_erf_dispatch(self):
        original = math.erf
        def model(p):
            return np.array([math.erf(p[0])], dtype=object)
        result = autoia(model, [Interval(-0.75, 0.9)], order=8)
        self.assertIs(math.erf, original)
        xs = np.linspace(-0.75, 0.9, 501)
        vals = [math.erf(float(x)) for x in xs]
        self.assertLessEqual(float(result.lower[0]), min(vals))
        self.assertGreaterEqual(float(result.upper[0]), max(vals))

    def test_parameterized_normal_helpers(self):
        mu = 2.0
        sigma = 1.5
        def model(p):
            x = p[0]
            return np.array([normal_pdf(x, mu, sigma), normal_cdf(x, mu, sigma)], dtype=object)
        result = autoia(model, [Interval(-1.0, 1.0)], order=8)
        xs = np.linspace(-1.0, 1.0, 501)
        pdf_vals = [phi((float(x)-mu)/sigma)/sigma for x in xs]
        cdf_vals = [Phi((float(x)-mu)/sigma) for x in xs]
        self.assertLessEqual(float(result.lower[0]), min(pdf_vals))
        self.assertGreaterEqual(float(result.upper[0]), max(pdf_vals))
        self.assertLessEqual(float(result.lower[1]), min(cdf_vals))
        self.assertGreaterEqual(float(result.upper[1]), max(cdf_vals))

    def test_interval_representation(self):
        def model(p):
            x=p[0]
            return np.array([erf(x), standard_normal_pdf(x), standard_normal_cdf(x)], dtype=object)
        result = autoia(model, [Interval(-1.0, 1.0)], order=8, representation="interval")
        self.assertLessEqual(float(result.lower[0]), math.erf(-1.0))
        self.assertGreaterEqual(float(result.upper[0]), math.erf(1.0))
        self.assertLessEqual(float(result.lower[1]), phi(1.0))
        self.assertGreaterEqual(float(result.upper[1]), phi(0.0))
        self.assertLessEqual(float(result.lower[2]), Phi(-1.0))
        self.assertGreaterEqual(float(result.upper[2]), Phi(1.0))


if __name__ == "__main__":
    unittest.main()
