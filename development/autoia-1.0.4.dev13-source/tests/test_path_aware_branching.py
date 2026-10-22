import math
import unittest
import numpy as np

from autoia import (
    autoia, autoia_subdivide, autoia_adaptive_subdivide,
    Interval, BranchPolicy, AutoIAResult, AutoIAPathwiseResult,
    ResourceLimitError, UnsupportedOperationError, ModelExecutionError,
)


def assert_contains(testcase, result, value, index=0, tol=0.0):
    testcase.assertLessEqual(float(result.lower[index]) - tol, float(value))
    testcase.assertGreaterEqual(float(result.upper[index]) + tol, float(value))


class TestPathAwareBranching(unittest.TestCase):
    def test_branchless_model_keeps_ordinary_result_type(self):
        def model(p):
            return p[0] * p[0] + 1.0

        r = autoia(model, [Interval(-1.0, 1.0)], order=4)
        self.assertIsInstance(r, AutoIAResult)
        self.assertNotIsInstance(r, AutoIAPathwiseResult)

    def test_exact_affine_if_else_is_partitioned_at_root(self):
        def model(p):
            x = p[0]
            if x >= 0.0:
                return x
            return -x

        r = autoia(model, [Interval(-1.0, 1.0)], order=4)
        self.assertIsInstance(r, AutoIAPathwiseResult)
        self.assertEqual(len(r.branch_results), 2)
        self.assertTrue(any(e["action"] == "partition-exact-affine-predicate"
                            for e in r.path_events))
        self.assertLessEqual(r.lower[0], 0.0)
        self.assertGreaterEqual(r.upper[0], 1.0)
        # Only outward-rounding noise may extend below zero.
        self.assertGreater(r.lower[0], -1.0e-12)

    def test_strict_inequality_boundary_semantics(self):
        def model(p):
            x = p[0]
            if x > 0.0:
                return 2.0 + x
            return -3.0 + x

        r = autoia(model, [Interval(-1.0, 1.0)], order=4)
        # x=0 follows the false branch and must therefore include -3 exactly.
        assert_contains(self, r, -3.0)
        assert_contains(self, r, -4.0)
        assert_contains(self, r, 3.0)

    def test_equality_and_inequality_create_boundary_path(self):
        def eq_model(p):
            x = p[0]
            if x == 0.0:
                return 10.0
            return x

        def ne_model(p):
            x = p[0]
            if x != 0.0:
                return x
            return -10.0

        eq = autoia(eq_model, [Interval(-1.0, 1.0)], order=3)
        ne = autoia(ne_model, [Interval(-1.0, 1.0)], order=3)
        self.assertGreaterEqual(eq.upper[0], 10.0)
        self.assertLessEqual(ne.lower[0], -10.0)
        self.assertGreaterEqual(len(eq.branch_results), 3)
        self.assertGreaterEqual(len(ne.branch_results), 3)

    def test_nested_if_elif_and_downstream_arithmetic(self):
        def model(p):
            x = p[0]
            if x < -0.25:
                y = -1.0 + x * x
            elif x <= 0.5:
                y = x * x
            else:
                y = 1.0 + x
            return y * y + 0.1 * x

        r = autoia(model, [Interval(-1.0, 1.0)], order=5)
        self.assertIsInstance(r, AutoIAPathwiseResult)
        for x in np.linspace(-1.0, 1.0, 401):
            if x < -0.25:
                y = -1.0 + x*x
            elif x <= 0.5:
                y = x*x
            else:
                y = 1.0 + x
            assert_contains(self, r, y*y + 0.1*x)

    def test_short_circuit_and_or_and_truth_testing(self):
        def model(p):
            x, y = p
            if x and (y > 0.0):
                a = x + y
            elif (x <= 0.0) or (y <= 0.0):
                a = x - y
            else:
                a = 0.0
            return a

        r = autoia(
            model,
            [Interval(-0.5, 0.5), Interval(-0.5, 0.5)],
            order=3,
            branch_policy=BranchPolicy(max_branches=64),
        )
        self.assertIsInstance(r, AutoIAPathwiseResult)
        for x in np.linspace(-0.5, 0.5, 17):
            for y in np.linspace(-0.5, 0.5, 17):
                if x and (y > 0.0):
                    a = x + y
                elif (x <= 0.0) or (y <= 0.0):
                    a = x - y
                else:
                    a = 0.0
                assert_contains(self, r, a)

    def test_nonlinear_multivariate_predicate_is_sound(self):
        def model(p):
            x, y = p
            z = x*x + y
            if z > 0.2:
                return np.exp(x) + y
            return np.cos(y) - x

        r = autoia(
            model,
            [Interval(-0.5, 0.5), Interval(-0.5, 0.5)],
            order=5,
            branch_policy=BranchPolicy(
                max_branches=64,
                max_split_depth_per_decision=2,
                max_total_splits=16,
            ),
        )
        self.assertIsInstance(r, AutoIAPathwiseResult)
        self.assertTrue(any(e["action"] == "enumerate-predicate-paths"
                            for e in r.path_events))
        for x in np.linspace(-0.5, 0.5, 25):
            for y in np.linspace(-0.5, 0.5, 25):
                z = x*x + y
                truth = math.exp(x) + y if z > 0.2 else math.cos(y) - x
                assert_contains(self, r, truth)

    def test_branch_local_semantic_solve(self):
        def model(p):
            x = p[0]
            A = np.zeros((2, 2))
            b = np.zeros(2)
            if x >= 0.0:
                A[0, 0] = 2.0 + 0.2*x
                A[1, 1] = 1.5
                b[:] = [1.0 + x, 0.5]
            else:
                A[0, 0] = 1.8 - 0.1*x
                A[1, 1] = 1.4 - 0.1*x
                b[:] = [0.8, 0.4 - x]
            A[0, 1] = 0.1
            A[1, 0] = -0.05
            u = np.linalg.solve(A, b)
            return u[0] + u[1]

        r = autoia(model, [Interval(-0.4, 0.4)], order=4)
        self.assertIsInstance(r, AutoIAPathwiseResult)
        self.assertGreaterEqual(len(r.certificates), 2)
        for x in np.linspace(-0.4, 0.4, 101):
            if x >= 0:
                A = np.array([[2.0+0.2*x, 0.1],[-0.05,1.5]])
                b = np.array([1.0+x,0.5])
            else:
                A = np.array([[1.8-0.1*x,0.1],[-0.05,1.4-0.1*x]])
                b = np.array([0.8,0.4-x])
            truth = np.linalg.solve(A,b).sum()
            assert_contains(self, r, truth)

    def test_piecewise_objects_retain_branch_metadata(self):
        def model(p):
            return p[0] if p[0] >= 0.0 else -p[0]

        r = autoia(model, [Interval(-1.0, 1.0)], order=3)
        piece = r.objects[0]
        self.assertEqual(piece.representation_name, "piecewise")
        self.assertEqual(len(piece.branches), len(r.branch_results))
        self.assertEqual(len(piece.branch_metadata), len(r.branch_results))
        self.assertTrue(all(m is not None for m in piece.branch_metadata))

    def test_fixed_subdivision_composes_with_path_aware_execution(self):
        def model(p):
            return p[0] if p[0] >= 0.0 else -p[0]

        r = autoia_subdivide(model, [Interval(-1.0, 1.0)], 2, order=3)
        self.assertLessEqual(r.lower[0], 0.0)
        self.assertGreaterEqual(r.upper[0], 1.0)
        self.assertTrue(any(isinstance(br, AutoIAPathwiseResult)
                            for br in r.branch_results))

    def test_adaptive_structural_subdivision_with_paths_fails_typed(self):
        def model(p):
            return p[0] if p[0] >= 0.0 else -p[0]

        with self.assertRaises(UnsupportedOperationError):
            autoia_adaptive_subdivide(model, [Interval(-1.0, 1.0)], order=3)

    def test_branch_budget_fails_closed(self):
        def model(p):
            x = p[0]
            return x if x >= 0.0 else -x

        with self.assertRaises(ResourceLimitError):
            autoia(
                model, [Interval(-1.0, 1.0)], order=3,
                branch_policy=BranchPolicy(max_branches=1),
            )

    def test_path_output_shape_mismatch_is_model_execution_failure(self):
        def model(p):
            if p[0] >= 0.0:
                return np.array([p[0], 1.0])
            return p[0]

        with self.assertRaises(ModelExecutionError):
            autoia(model, [Interval(-1.0, 1.0)], order=3)


if __name__ == "__main__":
    unittest.main()
