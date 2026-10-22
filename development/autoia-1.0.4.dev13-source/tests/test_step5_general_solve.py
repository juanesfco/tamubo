import unittest
import numpy as np

from autoia import autoia, Interval
from autoia.semantic.linear_solve import build_solution_polynomials_general
from autoia.polynomial import zero_alpha


class TestStep5GeneralPolynomialSolve(unittest.TestCase):
    def test_general_recurrence_quadratic_scalar_coefficients(self):
        # (1 + a q^2) x = 1 has formal series
        # x = 1 - a q^2 + a^2 q^4 + O(q^6).
        a = 0.25
        m = 1
        A = {
            (0,): np.array([[1.0]]),
            (2,): np.array([[a]]),
        }
        b = {(0,): np.array([1.0])}
        X = build_solution_polynomials_general(A, b, m, order=4)
        self.assertAlmostEqual(X[(0,)][0], 1.0)
        self.assertAlmostEqual(X[(1,)][0], 0.0)
        self.assertAlmostEqual(X[(2,)][0], -a)
        self.assertAlmostEqual(X[(3,)][0], 0.0)
        self.assertAlmostEqual(X[(4,)][0], a * a)

    def test_midpoint_preconditioner_rescues_borderline_quadratic_family(self):
        # Center preconditioning gives rho >= 1 for A(q)=1+q^2 on [-1,1].
        # The rigorous family-hull midpoint preconditioner gives rho <= 1/3
        # up to outward rounding and certifies the same represented family.
        def model(p):
            q = p[0]
            A = np.zeros((1, 1))
            b = np.zeros(1)
            A[0, 0] = 1.0 + q * q
            b[0] = 1.0
            return np.linalg.solve(A, b)

        R = autoia(model, [Interval(-1.0, 1.0)], order=4)
        self.assertEqual(len(R.certificates), 1)
        C = R.certificates[0]
        self.assertEqual(C["solve_mode"], "polynomial-general")
        self.assertEqual(C["selected_preconditioner"], "family-hull-midpoint")
        self.assertLess(C["rho_upper"], 0.34)

        by_name = {
            (c["predictor_name"], c["preconditioner_name"]): c
            for c in C["portfolio_candidates"]
        }
        center = by_name[("formal-A0-recurrence", "constant-coefficient")]
        self.assertFalse(center["success"])
        self.assertGreaterEqual(center["rho_upper"], 1.0)

        # Exact solution is 1/(1+q^2), hence lies in [1/2,1].
        self.assertLessEqual(R.lower[0], 0.5)
        self.assertGreaterEqual(R.upper[0], 1.0)
        for q in np.linspace(-1.0, 1.0, 201):
            truth = 1.0 / (1.0 + q*q)
            self.assertLessEqual(R.lower[0], truth)
            self.assertGreaterEqual(R.upper[0], truth)

    def test_quadratic_2x2_matrix_and_rhs(self):
        def model(p):
            x, y = p
            A = np.zeros((2, 2))
            b = np.zeros(2)
            A[0, 0] = 2.0 + 0.10*x*x + 0.05*x*y
            A[0, 1] = 0.20 + 0.03*y*y
            A[1, 0] = -0.10 + 0.02*x*y
            A[1, 1] = 1.50 + 0.08*y*y
            b[0] = 1.0 + 0.20*x + 0.05*x*x
            b[1] = 0.50 - 0.10*y + 0.04*x*y
            return np.linalg.solve(A, b)

        box = [Interval(-0.4, 0.4), Interval(-0.3, 0.3)]
        R = autoia(model, box, order=4)
        cert = R.certificates[0]
        self.assertEqual(cert["solve_mode"], "polynomial-general")
        self.assertEqual(cert["solve_input_polynomial_degree_A"], 2)
        self.assertEqual(cert["solve_input_polynomial_degree_b"], 2)
        self.assertLess(cert["rho_upper"], 1.0)

        for x in np.linspace(-0.4, 0.4, 17):
            for y in np.linspace(-0.3, 0.3, 13):
                A = np.array([
                    [2.0 + 0.10*x*x + 0.05*x*y, 0.20 + 0.03*y*y],
                    [-0.10 + 0.02*x*y, 1.50 + 0.08*y*y],
                ])
                b = np.array([
                    1.0 + 0.20*x + 0.05*x*x,
                    0.50 - 0.10*y + 0.04*x*y,
                ])
                truth = np.linalg.solve(A, b)
                self.assertTrue(np.all(R.lower <= truth))
                self.assertTrue(np.all(R.upper >= truth))

    def test_elementary_generated_matrix_polynomial_can_be_solved(self):
        def model(p):
            x = p[0]
            A = np.zeros((2, 2))
            b = np.zeros(2)
            A[0, 0] = 2.0 + 0.20*np.exp(x)
            A[0, 1] = 0.10 + 0.02*x*x
            A[1, 0] = -0.05*x
            A[1, 1] = 1.7 + 0.10*np.cos(x)
            b[0] = 1.0 + 0.1*x*x
            b[1] = 0.4 + 0.05*np.sin(x)
            return np.linalg.solve(A, b)

        R = autoia(model, [Interval(-0.2, 0.2)], order=4)
        cert = R.certificates[0]
        self.assertEqual(cert["solve_mode"], "polynomial-general")
        self.assertGreaterEqual(cert["solve_input_polynomial_degree_A"], 2)

        for x in np.linspace(-0.2, 0.2, 101):
            A = np.array([
                [2.0 + 0.20*np.exp(x), 0.10 + 0.02*x*x],
                [-0.05*x, 1.7 + 0.10*np.cos(x)],
            ])
            b = np.array([1.0 + 0.1*x*x, 0.4 + 0.05*np.sin(x)])
            truth = np.linalg.solve(A, b)
            self.assertTrue(np.all(R.lower <= truth))
            self.assertTrue(np.all(R.upper >= truth))

    def test_multiple_right_hand_sides_preserve_numpy_shape(self):
        def model(p):
            q = p[0]
            A = np.zeros((2, 2))
            B = np.zeros((2, 2))
            A[0, 0] = 2.0 + 0.10*q*q
            A[0, 1] = 0.15*q
            A[1, 0] = -0.05*q
            A[1, 1] = 1.5 + 0.08*q*q
            B[0, 0] = 1.0 + 0.1*q
            B[1, 0] = 0.4
            B[0, 1] = 0.2
            B[1, 1] = 1.0 - 0.1*q*q
            return np.linalg.solve(A, B)

        R = autoia(model, [Interval(-0.4, 0.4)], order=4)
        self.assertEqual(R.output_shape, (2, 2))
        self.assertEqual(R.lower_shaped.shape, (2, 2))
        self.assertEqual(len(R.certificates), 2)
        self.assertTrue(all(c["solve_mode"] == "polynomial-general" for c in R.certificates))

        for q in np.linspace(-0.4, 0.4, 101):
            A = np.array([
                [2.0 + 0.10*q*q, 0.15*q],
                [-0.05*q, 1.5 + 0.08*q*q],
            ])
            B = np.array([
                [1.0 + 0.1*q, 0.2],
                [0.4, 1.0 - 0.1*q*q],
            ])
            truth = np.linalg.solve(A, B)
            self.assertTrue(np.all(R.lower_shaped <= truth))
            self.assertTrue(np.all(R.upper_shaped >= truth))


if __name__ == "__main__":
    unittest.main()
