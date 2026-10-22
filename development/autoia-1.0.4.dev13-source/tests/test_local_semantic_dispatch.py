import inspect
import unittest

import numpy as np
from numpy.linalg import solve as imported_solve_alias
from numpy.linalg import eigvals as imported_eigvals_alias

from autoia import Interval, autoia, UnsupportedOperationError
from autoia.semantic import GenericSolveContext, execute_model_with_dispatch
import autoia.semantic.linear_solve as linear_solve
import autoia.semantic.dispatch as dispatch


_REAL_SOLVE = np.linalg.solve
_REAL_NUMPY = np
_OBSERVED_REAL_SOLVE_UNCHANGED = []


def _spring_system_from_p(p):
    k1, k2, F = p
    A = np.array([[k1 + k2, -k2], [-k2, k2]], dtype=object)
    b = np.array([F, 0.0], dtype=object)
    return A, b


def helper_uses_numpy_solve(A, b):
    return np.linalg.solve(A, b)


def model_through_same_module_helper(p):
    A, b = _spring_system_from_p(p)
    return helper_uses_numpy_solve(A, b)


def model_through_imported_alias(p):
    A, b = _spring_system_from_p(p)
    return imported_solve_alias(A, b)


def model_with_local_numpy_import(p):
    # dev7 deliberately does not rewrite IMPORT_NAME bytecode.  This model must
    # fail closed rather than relying on process-global monkeypatching.
    import numpy as local_np
    A, b = _spring_system_from_p(p)
    return local_np.linalg.solve(A, b)


def model_observes_real_numpy_while_using_proxy(
    p,
    real_numpy=_REAL_NUMPY,
    real_solve=_REAL_SOLVE,
):
    # Defaults are intentionally ordinary Python objects captured before AutoIA
    # prepares the shadow namespace.  They let the model observe the real NumPy
    # module while its global ``np`` name is locally proxied.
    _OBSERVED_REAL_SOLVE_UNCHANGED.append(real_numpy.linalg.solve is real_solve)
    A, b = _spring_system_from_p(p)
    return np.linalg.solve(A, b)


class TestLocalSemanticDispatch(unittest.TestCase):
    def setUp(self):
        _OBSERVED_REAL_SOLVE_UNCHANGED.clear()

    def _run_spring(self, model):
        return autoia(
            model,
            [Interval(90.0, 110.0), Interval(900.0, 1100.0), Interval(9.5, 10.5)],
            order=4,
        )

    def test_public_execution_never_replaces_real_numpy_solve(self):
        before = np.linalg.solve
        R = self._run_spring(model_observes_real_numpy_while_using_proxy)
        after = np.linalg.solve

        self.assertIs(before, _REAL_SOLVE)
        self.assertIs(after, _REAL_SOLVE)
        self.assertEqual(_OBSERVED_REAL_SOLVE_UNCHANGED, [True])
        self.assertTrue(R.machine_rigorous)
        self.assertEqual(len(R.certificates), 1)

    def test_same_module_helper_is_prepared_locally(self):
        R = self._run_spring(model_through_same_module_helper)
        self.assertTrue(R.machine_rigorous)
        self.assertEqual(len(R.certificates), 1)

    def test_direct_imported_solve_alias_is_prepared_locally(self):
        R = self._run_spring(model_through_imported_alias)
        self.assertTrue(R.machine_rigorous)
        self.assertEqual(len(R.certificates), 1)

    def test_context_entry_itself_does_not_mutate_numpy(self):
        # A tiny validated prototype is easiest to obtain from an ordinary run.
        def no_solve(p):
            return p[0]

        probe = autoia(no_solve, [Interval(-1.0, 1.0)], order=2)
        prototype = probe.objects[0]

        before = np.linalg.solve
        with GenericSolveContext(prototype) as ctx:
            self.assertIs(np.linalg.solve, before)
        self.assertIs(np.linalg.solve, before)
        self.assertEqual(ctx.certificates, [])

    def test_local_numpy_import_fails_closed_under_current_dispatch_boundary(self):
        with self.assertRaises(UnsupportedOperationError):
            self._run_spring(model_with_local_numpy_import)
        self.assertIs(np.linalg.solve, _REAL_SOLVE)

    def test_active_code_contains_no_global_numpy_solve_assignment(self):
        sources = (
            inspect.getsource(linear_solve),
            inspect.getsource(dispatch),
        )
        for source in sources:
            self.assertNotIn("np.linalg.solve =", source)


def model_with_eigvals(p):
    x = p[0]
    A = np.array([[2.0 + x, -1.0], [-1.0, 2.0 - x]])
    return np.linalg.eigvals(A)


def model_with_imported_eigvals_alias(p):
    x = p[0]
    A = np.array([[2.0 + x, -1.0], [-1.0, 2.0 - x]])
    return imported_eigvals_alias(A)


class TestDeferredLinalgDiagnostics(unittest.TestCase):
    def test_eigvals_on_validated_data_fails_with_explicit_operation_name(self):
        with self.assertRaises(UnsupportedOperationError) as cm:
            autoia(model_with_eigvals, [Interval(-0.1, 0.1)], order=2)
        text = str(cm.exception)
        self.assertIn("np.linalg.eigvals", text)
        self.assertIn("deferred", text.lower())

    def test_imported_eigvals_alias_gets_same_explicit_diagnostic(self):
        with self.assertRaises(UnsupportedOperationError) as cm:
            autoia(model_with_imported_eigvals_alias, [Interval(-0.1, 0.1)], order=2)
        self.assertIn("np.linalg.eigvals", str(cm.exception))

    def test_eigvals_proxy_preserves_ordinary_numeric_numpy_behavior(self):
        A = np.array([[2.0, -1.0], [-1.0, 2.0]])
        expected = np.linalg.eigvals(A)
        # Calling the model normally must remain ordinary NumPy execution.
        actual = model_with_eigvals(np.array([0.0]))
        self.assertTrue(np.allclose(np.sort(actual), np.sort(expected)))


if __name__ == "__main__":
    unittest.main()
