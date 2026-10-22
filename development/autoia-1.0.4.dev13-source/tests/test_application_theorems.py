import unittest
import math
import numpy as np

from autoia import Interval
from autoia.application import (
    ApplicationBox,
    CertificationContext,
    application_theorem,
    minimum_upper_bound,
    maximum_lower_bound,
)
from autoia.exceptions import CertificationError


class ApplicationTheoremPublicAPITests(unittest.TestCase):
    def test_certified_scalar_arithmetic_contains_exact_values(self):
        cert = CertificationContext()
        x = cert.interval(0.9, 1.1)
        y = cert.interval(1.9, 2.2)
        z = cert.exp((x * y + 2.0) / 3.0)
        rng = np.random.default_rng(123)
        for _ in range(100):
            xv = rng.uniform(0.9, 1.1)
            yv = rng.uniform(1.9, 2.2)
            exact = math.exp((xv * yv + 2.0) / 3.0)
            self.assertLessEqual(z.lower, exact)
            self.assertGreaterEqual(z.upper, exact)

    def test_context_enclose_hides_autoia_representation(self):
        cert = CertificationContext()
        box = ApplicationBox(np.array([0.9, 1.8]), np.array([1.1, 2.2]))
        result = cert.enclose(lambda p: p[0] * p[1] + np.exp(p[0]), box)
        self.assertLessEqual(result.lower, 0.9 * 1.8 + math.exp(0.9))
        self.assertGreaterEqual(result.upper, 1.1 * 2.2 + math.exp(1.1))
        self.assertFalse(hasattr(result, "polynomial"))
        self.assertFalse(hasattr(result, "generators"))

    def test_public_arithmetic_domain_failures_are_typed(self):
        cert = CertificationContext()
        with self.assertRaises(CertificationError):
            _ = cert.point(1.0) / cert.interval(-1.0, 1.0)
        with self.assertRaises(CertificationError):
            cert.sqrt(cert.interval(-2.0, -1.0))

    def test_checked_assumption_fails_closed(self):
        @application_theorem(name="positive scale theorem", target="f")
        def theorem(box, cert, scale):
            cert.require(scale > 0.0, "scale is positive")
            x = cert.input_coordinate(box, 0)
            return scale * x

        with self.assertRaises(CertificationError):
            theorem((np.array([0.0]), np.array([1.0])), scale=-1.0)

    def test_theorem_metadata_and_portfolio(self):
        @application_theorem(name="bound A", target="f")
        def bound_a(box, cert):
            x = cert.input_coordinate(box, 0)
            return x + 2.0

        @application_theorem(name="bound B", target="f")
        def bound_b(box, cert):
            return cert.point(10.0)

        box = ApplicationBox(np.array([-1.0]), np.array([3.0]))
        a = bound_a(box)
        b = bound_b(box)
        portfolio = minimum_upper_bound(a, b)
        self.assertEqual(a.theorem, "bound A")
        self.assertEqual(a.theorem_status, "user_asserted")
        self.assertEqual(portfolio.winner.theorem, "bound A")
        self.assertEqual(portfolio.value, a.value)
        self.assertTrue(portfolio.machine_rigorous)


    def test_certified_interval_intersection_is_general_and_tighter(self):
        cert = CertificationContext()
        a = cert.interval(-2.0, 4.0)
        b = cert.interval(1.0, 7.0)
        c = cert.intersect(a, b)
        self.assertEqual(c.lower, 1.0)
        self.assertEqual(c.upper, 4.0)
        with self.assertRaises(CertificationError):
            cert.intersect(cert.interval(-2.0, -1.0), cert.interval(1.0, 2.0))

    def test_lower_bound_portfolio_uses_strongest_valid_lower_bound(self):
        cert = CertificationContext()
        a = cert.lower_bound(cert.interval(1.0, 3.0), target="f")
        b = cert.lower_bound(cert.interval(2.0, 5.0), target="f")
        portfolio = maximum_lower_bound(a, b)
        self.assertEqual(portfolio.value, 2.0)
        self.assertIs(portfolio.winner, b)
        self.assertTrue(portfolio.machine_rigorous)

    def test_public_interval_box_input(self):
        @application_theorem(name="square loose theorem", target="f")
        def theorem(box, cert):
            x = cert.input_coordinate(box, 0)
            return cert.square(x)

        b = theorem([Interval(-2.0, 3.0)])
        self.assertGreaterEqual(b.value, 9.0)


if __name__ == '__main__':
    unittest.main()
