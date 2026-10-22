import unittest
import numpy as np

from autoia import autoia, Interval


class TestGeneratorAwareSemanticSolve(unittest.TestCase):
    def test_rhs_generator_survives_identity_solve_and_cancels(self):
        def model(p):
            x = p[0]
            y = x * x  # order=1 pushes x^2 into a shared auxiliary generator
            A = np.zeros((1, 1))
            b = np.zeros(1)
            A[0, 0] = 1.0
            b[0] = y
            u = np.linalg.solve(A, b)
            return np.array([y, u[0], u[0] - y], dtype=object)

        gen = autoia(
            model, [Interval(-1.0, 1.0)], order=1,
            representation="generator",
        )
        poly = autoia(
            model, [Interval(-1.0, 1.0)], order=1,
            representation="polynomial",
        )

        # The new solve carries the incoming generator through the identity
        # solve.  The final subtraction therefore cancels that shared error.
        self.assertLess(gen.width[2], 1.0e-12)
        self.assertGreater(poly.width[2], 1.0)

        cert = gen.certificates[-1]
        self.assertTrue(cert["generator_aware"])
        self.assertEqual(cert["solve_mode"], "generator-aware")
        self.assertTrue(cert["input_generator_ids"])
        self.assertTrue(cert["preserved_input_generator_ids"])
        self.assertEqual(
            set(cert["input_generator_ids"]),
            set(cert["preserved_input_generator_ids"]),
        )

    def test_shared_rhs_generator_maps_to_multiple_solution_components(self):
        def model(p):
            x = p[0]
            y = x * x
            A = np.array([[1.0, 0.0], [0.0, 2.0]])
            b = np.array([y, 2.0 * y], dtype=object)
            u = np.linalg.solve(A, b)
            return np.array([u[0], u[1], u[0] - u[1]], dtype=object)

        result = autoia(
            model, [Interval(-1.0, 1.0)], order=1,
            representation="generator",
        )
        self.assertLess(result.width[2], 1.0e-12)

        cert = result.certificates[-1]
        input_ids = set(cert["input_generator_ids"])
        preserved = set(cert["preserved_input_generator_ids"])
        self.assertTrue(input_ids & preserved)

        ids = result.generator_ids
        G = result.generator_matrix
        id_to_col = {gid: j for j, gid in enumerate(ids)}
        common = [gid for gid in input_ids if gid in id_to_col]
        self.assertTrue(common)
        # At least one incoming shared generator must have nonzero coefficients
        # in both solution components.
        self.assertTrue(any(
            G[0, id_to_col[gid]] != 0.0 and G[1, id_to_col[gid]] != 0.0
            for gid in common
        ))

    def test_generator_in_matrix_is_preserved_and_samples_are_contained(self):
        def model(p):
            x = p[0]
            y = x * x
            A = np.zeros((1, 1))
            b = np.ones(1)
            A[0, 0] = 2.0 + y
            u = np.linalg.solve(A, b)
            return np.array([y, u[0]], dtype=object)

        result = autoia(
            model, [Interval(-0.5, 0.5)], order=1,
            representation="generator",
        )
        cert = result.certificates[-1]
        self.assertTrue(cert["input_generator_ids"])
        self.assertTrue(cert["preserved_input_generator_ids"])
        self.assertEqual(
            set(cert["input_generator_ids"]),
            set(cert["preserved_input_generator_ids"]),
        )
        self.assertEqual(cert["selected_generator_predictor"],
                         "first-order-shared-generator")

        for x in np.linspace(-0.5, 0.5, 101):
            truth = 1.0 / (2.0 + x * x)
            self.assertLessEqual(result.lower[1], truth)
            self.assertGreaterEqual(result.upper[1], truth)

    def test_remaining_box_corrections_do_not_invent_shared_output_error(self):
        def model(p):
            x = p[0]
            y = x * x
            A = np.zeros((2, 2))
            b = np.zeros(2)
            A[0, 0] = 2.0 + 0.15 * y
            A[0, 1] = 0.1
            A[1, 0] = -0.05
            A[1, 1] = 1.6 + 0.08 * y
            b[0] = 1.0 + 0.2 * y
            b[1] = 0.4 - 0.1 * y
            return np.linalg.solve(A, b)

        result = autoia(
            model, [Interval(-0.8, 0.8)], order=1,
            representation="generator",
        )
        cert = result.certificates[-1]
        input_ids = set(cert["input_generator_ids"])
        correction_ids = set(cert["independent_correction_generator_ids"])
        self.assertTrue(correction_ids.isdisjoint(input_ids))

        # Fresh correction generators are per scalar component unless the proof
        # explicitly provides coupling.  There should be no newly invented
        # correction generator with nonzero coefficients in both output rows.
        ids = result.generator_ids
        G = result.generator_matrix
        id_to_col = {gid: j for j, gid in enumerate(ids)}
        for gid in correction_ids:
            if gid in id_to_col:
                col = G[:, id_to_col[gid]]
                self.assertLessEqual(np.count_nonzero(col), 1)

    def test_multiple_rhs_preserves_input_generator_provenance(self):
        def model(p):
            x = p[0]
            y = x * x
            A = np.array([[1.5, 0.1], [0.0, 2.0]])
            B = np.empty((2, 2), dtype=object)
            B[:, 0] = [y, 2.0 * y]
            B[:, 1] = [1.0 + y, 0.5 - y]
            return np.linalg.solve(A, B)

        result = autoia(
            model, [Interval(-0.5, 0.5)], order=1,
            representation="generator",
        )
        self.assertEqual(result.output_shape, (2, 2))
        self.assertEqual(len(result.certificates), 2)
        for cert in result.certificates:
            self.assertTrue(cert["generator_aware"])
            self.assertTrue(cert["input_generator_ids"])
            self.assertTrue(cert["preserved_input_generator_ids"])

    def test_direct_auxiliary_generator_domain_is_certified(self):
        from autoia.enclosures import SharedGeneratorPolynomialEnclosure
        from autoia.semantic.linear_solve import generator_semantic_solve
        from autoia.trusted import Interval, RigorousBinary64Backend

        backend = RigorousBinary64Backend()
        h = np.array([0.0])
        proto = SharedGeneratorPolynomialEnclosure.constant(
            0.0, h, 0, backend, "portfolio"
        )

        def rep(c, gens=None):
            return SharedGeneratorPolynomialEnclosure(
                {(0,): float(c)}, Interval(0.0, 0.0), h, 0, backend,
                "portfolio", generators=dict(gens or {}), lift_remainder=False,
            )

        A = np.array([
            [rep(2.0, {101: 0.10}), rep(0.0, {202: 0.05})],
            [rep(0.0, {101: 0.02}), rep(1.5, {202: -0.08})],
        ], dtype=object)
        b = np.array([
            rep(1.0, {101: 0.20}),
            rep(0.5, {202: 0.10}),
        ], dtype=object)

        out, cert = generator_semantic_solve(A, b, proto, np.linalg.solve)
        self.assertEqual(set(cert["input_generator_ids"]), {101, 202})
        self.assertEqual(set(cert["preserved_input_generator_ids"]), {101, 202})
        ranges = [v.range()[0] for v in out]

        for e1 in np.linspace(-1.0, 1.0, 31):
            for e2 in np.linspace(-1.0, 1.0, 31):
                Atrue = np.array([
                    [2.0 + 0.10 * e1, 0.05 * e2],
                    [0.02 * e1, 1.5 - 0.08 * e2],
                ])
                btrue = np.array([1.0 + 0.20 * e1, 0.5 + 0.10 * e2])
                truth = np.linalg.solve(Atrue, btrue)
                for j in range(2):
                    self.assertLessEqual(ranges[j].lo, truth[j])
                    self.assertGreaterEqual(ranges[j].hi, truth[j])


if __name__ == "__main__":
    unittest.main()
