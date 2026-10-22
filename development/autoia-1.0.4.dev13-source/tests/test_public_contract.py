import unittest
import numpy as np

from autoia import Interval, autoia, RigorousBinary64Backend, CertificationError


class TestPublicContract(unittest.TestCase):
    def test_spring_contains_analytic_range(self):
        def spring(p):
            k1, k2, F = p
            A = np.array([[k1 + k2, -k2], [-k2, k2]])
            b = np.array([F, 0.0])
            return np.linalg.solve(A, b)

        R = autoia(
            spring,
            [Interval(90, 110), Interval(900, 1100), Interval(9.5, 10.5)],
            order=6,
            backend=RigorousBinary64Backend(),
        )
        exact_lo = 9.5 / 110.0
        exact_hi = 10.5 / 90.0
        self.assertTrue(R.machine_rigorous)
        for j in range(2):
            self.assertLessEqual(R.lower[j], exact_lo)
            self.assertGreaterEqual(R.upper[j], exact_hi)
        self.assertEqual(len(R.certificates), 1)

    def test_manual_2x2_needs_no_semantic_certificate(self):
        def manual_solve(A, b):
            a, bb = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            det = a*d - bb*c
            return np.array([
                (b[0]*d - bb*b[1]) / det,
                (a*b[1] - b[0]*c) / det,
            ], dtype=object)

        def model(p):
            p1, p2 = p
            A = np.array([[2+p1, 0.2*p2], [0.2*p2, 1.5-p1]])
            b = np.array([1+0.1*p2, 0.5+p1])
            return manual_solve(A, b)

        R = autoia(model, [Interval(-0.2, 0.2), Interval(-0.5, 0.5)], order=5)
        self.assertTrue(R.machine_rigorous)
        self.assertEqual(len(R.certificates), 0)

    def test_zero_crossing_denominator_fails_closed(self):
        def model(p):
            return 1.0 / p[0]

        with self.assertRaises(CertificationError):
            autoia(model, [Interval(-1.0, 1.0)], order=4)


if __name__ == "__main__":
    unittest.main()
