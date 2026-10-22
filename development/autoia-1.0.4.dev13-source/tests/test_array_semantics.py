import unittest
import numpy as np

from autoia import autoia, Interval, UnsupportedOperationError


# Direct aliases exercise the shadow-global alias replacement path.
from numpy import zeros as zeros_alias
from numpy import sum as sum_alias
from numpy import dot as dot_alias
from numpy.linalg import norm as norm_alias


class TestArrayConstructionAndMutation(unittest.TestCase):
    BOX = [Interval(1.0, 2.0), Interval(3.0, 4.0)]

    def test_zeros_matrix_assembly_without_dtype_object(self):
        def model(p):
            A = np.zeros((2, 2))
            A[0, 0] = p[0]
            A[0, 1] = 2.0
            A[1, 0] = p[1]
            A[1, 1] = p[0] + 1.0
            return A

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (2, 2))
        L = R.lower_shaped
        U = R.upper_shaped
        self.assertLessEqual(L[0, 0], 1.0); self.assertGreaterEqual(U[0, 0], 2.0)
        self.assertEqual(L[0, 1], 2.0); self.assertEqual(U[0, 1], 2.0)
        self.assertLessEqual(L[1, 0], 3.0); self.assertGreaterEqual(U[1, 0], 4.0)

    def test_array_constructor_and_slice_assignment(self):
        def model(p):
            a = np.array([0.0, 0.0, 0.0])
            a[:2] = p
            a[2] = p[0] + p[1]
            return a[1:]

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (2,))
        self.assertLessEqual(R.lower_shaped[0], 3.0)
        self.assertGreaterEqual(R.upper_shaped[1], 6.0)

    def test_zero_initialized_linear_system_assembly_reaches_semantic_solve(self):
        def model(p):
            k1, k2 = p
            A = np.zeros((2, 2))
            b = np.zeros(2)
            A[0, 0] = k1 + k2
            A[0, 1] = -k2
            A[1, 0] = -k2
            A[1, 1] = k2
            b[0] = 10.0
            return np.linalg.solve(A, b)

        R = autoia(model, [Interval(90.0, 110.0), Interval(900.0, 1100.0)], order=4)
        self.assertEqual(len(R.certificates), 1)
        exact_lo = 10.0 / 110.0
        exact_hi = 10.0 / 90.0
        self.assertLessEqual(R.lower[0], exact_lo)
        self.assertGreaterEqual(R.upper[0], exact_hi)

    def test_direct_zeros_alias_is_prepared(self):
        def model(p):
            a = zeros_alias(2)
            a[0] = p[0]
            a[1] = p[1]
            return a

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (2,))


class TestBroadcastAndStructuralArraySemantics(unittest.TestCase):
    BOX = [Interval(1.0, 2.0), Interval(3.0, 4.0)]

    def test_two_dimensional_broadcasting(self):
        def model(p):
            a = np.array([[p[0]], [p[1]]])        # (2,1)
            b = np.array([[1.0, 2.0, 3.0]])       # (1,3)
            return 2.0 * a + b

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (2, 3))
        self.assertLessEqual(R.lower_shaped[0, 0], 3.0)
        self.assertGreaterEqual(R.upper_shaped[1, 2], 11.0)

    def test_preexisting_numeric_array_times_validated_scalar(self):
        coeff = np.array([1.5, -2.0, 0.25])
        def model(p):
            return coeff * p[0] + np.array([p[1], 0.0, 1.0])
        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (3,))
        for x in (1.0, 2.0):
            for y in (3.0, 4.0):
                v = coeff*x + np.array([y, 0.0, 1.0])
                self.assertTrue(np.all(R.lower <= v))
                self.assertTrue(np.all(R.upper >= v))

    def test_stack_concatenate_reshape_transpose(self):
        def model(p):
            a = np.array([p[0], p[1]])
            b = np.stack([a, 2.0*a])
            c = np.concatenate([b.reshape(-1), np.array([p[0] + p[1]])])
            return c.reshape(5, 1).T

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (1, 5))


class TestReductionsAndLinearAlgebra(unittest.TestCase):
    BOX = [Interval(1.0, 2.0), Interval(3.0, 4.0)]

    def test_sum_mean_prod_axes_and_array_methods(self):
        def model(p):
            A = np.array([[p[0], p[1]], [2.0*p[0], 3.0]])
            return np.array([
                np.sum(A, axis=0)[0],
                np.mean(A, axis=1)[1],
                np.prod(A, axis=0)[0],
                A.sum(axis=1)[0],
                A.mean(axis=0)[1],
                A.prod(axis=1)[1],
            ])

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (6,))
        # Dense exact endpoint diagnostics for this simple monotone case.
        expected_points = []
        for x in (1.0, 2.0):
            for y in (3.0, 4.0):
                A = np.array([[x, y], [2*x, 3.0]])
                expected_points.append(np.array([
                    np.sum(A, axis=0)[0], np.mean(A, axis=1)[1],
                    np.prod(A, axis=0)[0], A.sum(axis=1)[0],
                    A.mean(axis=0)[1], A.prod(axis=1)[1],
                ]))
        E = np.vstack(expected_points)
        self.assertTrue(np.all(R.lower <= E.min(axis=0)))
        self.assertTrue(np.all(R.upper >= E.max(axis=0)))

    def test_direct_sum_alias(self):
        def model(p):
            return sum_alias(np.array([p[0], p[1]]))
        R = autoia(model, self.BOX, order=3)
        self.assertLessEqual(R.lower[0], 4.0)
        self.assertGreaterEqual(R.upper[0], 6.0)

    def test_dot_matmul_inner_outer_and_trace(self):
        def model(p):
            A = np.array([[p[0], 1.0], [2.0, p[1]]])
            x = np.array([p[1], p[0]])
            d = np.dot(A, x)
            m = np.matmul(A, x)
            op = A @ x
            inn = np.inner(x, np.array([2.0, 3.0]))
            out = np.outer(x, np.array([2.0, 3.0]))
            tr = np.trace(A)
            return np.concatenate([d, m, op, np.array([inn, tr]), out.reshape(-1)])

        R = autoia(model, self.BOX, order=3)
        self.assertEqual(R.output_shape, (12,))
        self.assertTrue(np.all(np.isfinite(R.lower)))
        self.assertTrue(np.all(np.isfinite(R.upper)))

    def test_direct_dot_alias(self):
        def model(p):
            return dot_alias(np.array([p[0], p[1]]), np.array([2.0, -1.0]))
        R = autoia(model, self.BOX, order=3)
        self.assertLessEqual(R.lower[0], -2.0)
        self.assertGreaterEqual(R.upper[0], 1.0)


class TestNormsAndResultShape(unittest.TestCase):
    def test_vector_norm_and_alias(self):
        box = [Interval(1.0, 2.0), Interval(3.0, 4.0)]
        def model(p):
            x = np.array([p[0], p[1]])
            return np.array([np.linalg.norm(x), norm_alias(x, ord=2)])
        R = autoia(model, box, order=5)
        self.assertEqual(R.output_shape, (2,))
        for x in np.linspace(1.0, 2.0, 11):
            for y in np.linspace(3.0, 4.0, 11):
                v = np.sqrt(x*x + y*y)
                self.assertLessEqual(R.lower[0], v)
                self.assertGreaterEqual(R.upper[0], v)
                self.assertLessEqual(R.lower[1], v)
                self.assertGreaterEqual(R.upper[1], v)

    def test_matrix_frobenius_norm(self):
        box = [Interval(1.0, 2.0)]
        def model(p):
            A = np.array([[p[0], 1.0], [-2.0, 3.0]])
            return np.array([np.linalg.norm(A), np.linalg.norm(A, ord='fro')])
        R = autoia(model, box, order=5)
        for x in np.linspace(1.0, 2.0, 31):
            v = np.sqrt(x*x + 14.0)
            self.assertTrue(np.all(R.lower <= v))
            self.assertTrue(np.all(R.upper >= v))

    def test_vector_norm_crossing_zero_uses_sound_semantic_fallback(self):
        box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
        def model(p):
            return np.linalg.norm(np.array([p[0], p[1]]))
        R = autoia(model, box, order=5)
        self.assertLessEqual(R.lower[0], 0.0)
        self.assertGreaterEqual(R.upper[0], np.sqrt(2.0))
        self.assertEqual(R.representation_names[0], "interval")

    def test_matrix_spectral_norm_is_explicitly_unsupported(self):
        box = [Interval(1.0, 2.0)]
        def model(p):
            A = np.array([[p[0], 1.0], [0.0, 2.0]])
            return np.linalg.norm(A, ord=2)
        with self.assertRaises(UnsupportedOperationError):
            autoia(model, box, order=3)

    def test_scalar_output_shape_metadata(self):
        box = [Interval(1.0, 2.0)]
        R = autoia(lambda p: p[0] + 1.0, box)
        self.assertEqual(R.output_shape, ())
        self.assertEqual(R.lower_shaped.shape, ())
        self.assertEqual(R.upper_shaped.shape, ())


if __name__ == '__main__':
    unittest.main()
