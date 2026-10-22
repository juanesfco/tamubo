import math
import unittest
import numpy as np

from autoia import (
    autoia, Interval, SharedGeneratorPolynomialEnclosure,
)


class TestSharedGeneratorRepresentation(unittest.TestCase):
    def test_public_representation_and_exact_fanout_cancellation(self):
        def model(p):
            x = p[0]
            y = x * x
            return np.array([y, y - y], dtype=object)

        poly = autoia(model, [Interval(-1.0, 1.0)], order=1, representation="polynomial")
        gen = autoia(model, [Interval(-1.0, 1.0)], order=1, representation="generator")

        self.assertGreater(poly.width[1], 1.0)
        self.assertEqual(gen.lower[1], 0.0)
        self.assertEqual(gen.upper[1], 0.0)
        self.assertEqual(gen.representation_names, ["generator-polynomial", "generator-polynomial"])
        self.assertTrue(isinstance(gen.objects[0].rep, SharedGeneratorPolynomialEnclosure))

    def test_generator_matrix_exposes_shared_provenance(self):
        def model(p):
            x = p[0]
            y = x * x
            return np.array([y, 2.0 * y], dtype=object)

        result = autoia(model, [Interval(-1.0, 1.0)], order=1, representation="generator")
        G = result.generator_matrix
        ids = result.generator_ids
        self.assertEqual(G.shape[0], 2)
        self.assertEqual(G.shape[1], len(ids))
        # At least one generator is visibly shared by the two outputs with the
        # expected factor-of-two coefficient relation.  Tiny rounding generators
        # are allowed in addition.
        shared = []
        for j in range(G.shape[1]):
            if G[0, j] != 0.0 and G[1, j] != 0.0:
                shared.append(j)
        self.assertTrue(shared)
        j = shared[0]
        self.assertAlmostEqual(G[1, j], 2.0 * G[0, j], places=12)

    def test_recomputed_expression_does_not_invent_shared_provenance(self):
        def model(p):
            x = p[0]
            y1 = x * x
            y2 = x * x
            return y1 - y2

        result = autoia(model, [Interval(-1.0, 1.0)], order=1, representation="generator")
        # The two truncation certificates were produced independently.  AutoIA
        # must not pretend they are the same auxiliary error variable.
        self.assertGreater(result.width[0], 0.5)

    def test_elementary_function_preserves_generator_representation(self):
        def model(p):
            x = p[0]
            y = x * x
            return np.exp(y)

        result = autoia(model, [Interval(-0.5, 0.5)], order=2, representation="generator")
        self.assertEqual(result.representation_names, ["generator-polynomial"])
        self.assertGreaterEqual(len(result.generator_ids), 1)
        xs = np.linspace(-0.5, 0.5, 101)
        vals = np.exp(xs * xs)
        self.assertLessEqual(result.lower[0], float(vals.min()))
        self.assertGreaterEqual(result.upper[0], float(vals.max()))

    def test_semantic_solve_returns_generator_outputs_without_fake_coupling(self):
        def model(p):
            k1, k2 = p
            A = np.zeros((2, 2))
            b = np.zeros(2)
            A[0, 0] = k1 + k2
            A[0, 1] = -k2
            A[1, 0] = -k2
            A[1, 1] = k2
            b[0] = 10.0
            u = np.linalg.solve(A, b)
            return u

        result = autoia(
            model,
            [Interval(90.0, 110.0), Interval(900.0, 1100.0)],
            order=4,
            representation="generator",
        )
        self.assertEqual(result.representation_names, ["generator-polynomial"] * 2)
        self.assertTrue(result.certificates)
        self.assertEqual(result.certificates[-1]["representation"], "generator-polynomial")
        self.assertTrue(result.certificates[-1]["generatorized_output_correction"])
        # The box correction theorem proves independent component radii only, so
        # the solve must not force both components to share one correction id.
        nonzero_cols = [set(np.flatnonzero(result.generator_matrix[i])) for i in range(2)]
        self.assertNotEqual(nonzero_cols[0], nonzero_cols[1])

    def test_path_aware_execution_accepts_generator_representation(self):
        def model(p):
            x = p[0]
            y = x * x
            if x >= 0.0:
                return y + x
            return y - x

        result = autoia(
            model, [Interval(-1.0, 1.0)], order=2, representation="generator"
        )
        self.assertTrue(hasattr(result, "branch_results"))
        for branch in result.branch_results:
            self.assertEqual(branch.representation_names, ["generator-polynomial"])

    def test_sampled_nonlinear_containment(self):
        def model(p):
            x, y = p
            a = x * x + 0.2 * y
            b = a * (1.0 + 0.3 * x) + np.sin(a)
            return np.array([a, b, b - b], dtype=object)

        result = autoia(
            model,
            [Interval(-0.7, 0.8), Interval(-0.4, 0.6)],
            order=3,
            representation="generator",
        )
        self.assertEqual(result.lower[2], 0.0)
        self.assertEqual(result.upper[2], 0.0)
        rng = np.random.default_rng(20260829)
        for _ in range(500):
            x = rng.uniform(-0.7, 0.8)
            y = rng.uniform(-0.4, 0.6)
            a = x*x + 0.2*y
            b = a*(1.0 + 0.3*x) + math.sin(a)
            truth = (a, b, 0.0)
            for j, v in enumerate(truth):
                self.assertLessEqual(result.lower[j], v)
                self.assertGreaterEqual(result.upper[j], v)


if __name__ == "__main__":
    unittest.main()

class TestGeneratorAlgebraDirect(unittest.TestCase):
    def test_manual_shared_generator_product_contains_sampled_auxiliary_values(self):
        from autoia import RigorousBinary64Backend
        backend = RigorousBinary64Backend()
        h = np.array([0.4])
        A = SharedGeneratorPolynomialEnclosure(
            {(0,): 1.2, (1,): 0.3},
            Interval(0.0, 0.0), h, 2, backend, "termwise",
            generators={900001: 0.2}, lift_remainder=False,
        )
        B = SharedGeneratorPolynomialEnclosure(
            {(0,): -0.8, (1,): 0.1},
            Interval(0.0, 0.0), h, 2, backend, "termwise",
            generators={900001: -0.15, 900002: 0.05}, lift_remainder=False,
        )
        C = A * B
        lohi, _ = C.range("termwise")
        rng = np.random.default_rng(7)
        for _ in range(1000):
            q = rng.uniform(-0.4, 0.4)
            e1 = rng.uniform(-1.0, 1.0)
            e2 = rng.uniform(-1.0, 1.0)
            a = 1.2 + 0.3*q + 0.2*e1
            b = -0.8 + 0.1*q - 0.15*e1 + 0.05*e2
            truth = a*b
            self.assertLessEqual(lohi.lo, truth)
            self.assertGreaterEqual(lohi.hi, truth)
