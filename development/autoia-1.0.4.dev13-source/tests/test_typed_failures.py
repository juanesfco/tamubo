import unittest
import numpy as np

from autoia import (
    Interval,
    RangePolicy,
    autoia,
    AutoIAError,
    InvalidInputError,
    UnsupportedOperationError,
    CertificationError,
    ModelExecutionError,
)


class TestTypedFailures(unittest.TestCase):
    def test_public_interval_validation_is_typed(self):
        with self.assertRaises(InvalidInputError):
            Interval(2.0, 1.0)

    def test_policy_validation_is_typed(self):
        with self.assertRaises(InvalidInputError):
            RangePolicy(internal_method="not-a-method")

    def test_empty_input_box_is_invalid_input(self):
        with self.assertRaises(InvalidInputError):
            autoia(lambda p: 1.0, [])

    def test_bad_representation_is_invalid_input(self):
        with self.assertRaises(InvalidInputError):
            autoia(lambda p: p[0], [Interval(0.0, 1.0)], representation="mystery")

    def test_zero_crossing_division_is_certification_failure(self):
        def model(p):
            return 1.0 / p[0]

        with self.assertRaises(CertificationError) as cm:
            autoia(model, [Interval(-1.0, 1.0)], order=4)
        self.assertEqual(cm.exception.code, "certification-failed")
        self.assertIn("zero", str(cm.exception).lower())
        self.assertTrue(cm.exception.suggestions)
        self.assertIsInstance(cm.exception.__cause__, ZeroDivisionError)

    def test_unsupported_transcendental_is_typed(self):
        def model(p):
            return np.tan(p[0])

        with self.assertRaises(UnsupportedOperationError) as cm:
            autoia(model, [Interval(-0.1, 0.1)])
        self.assertEqual(cm.exception.code, "unsupported-operation")

    def test_negative_power_zero_crossing_is_certification_failure(self):
        def model(p):
            return p[0] ** -1

        with self.assertRaises(CertificationError):
            autoia(model, [Interval(-1.0, 1.0)])

    def test_uncertain_exponent_is_unsupported(self):
        def model(p):
            return p[0] ** p[1]

        with self.assertRaises(UnsupportedOperationError):
            autoia(model, [Interval(1.0, 2.0), Interval(0.2, 0.4)])

    def test_user_model_failure_is_not_mislabeled_as_autoia_internal(self):
        def model(p):
            raise RuntimeError("user model exploded")

        with self.assertRaises(ModelExecutionError) as cm:
            autoia(model, [Interval(0.0, 1.0)])
        self.assertIn("user model exploded", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    def test_interval_semantic_solve_contraction_failure_is_certification_error(self):
        def spring(p):
            k1, k2, F = p
            A = np.array([[k1 + k2, -k2], [-k2, k2]])
            b = np.array([F, 0.0])
            return np.linalg.solve(A, b)

        with self.assertRaises(CertificationError) as cm:
            autoia(
                spring,
                [Interval(90, 110), Interval(900, 1100), Interval(9.5, 10.5)],
                representation="interval",
            )
        self.assertIn("contraction", str(cm.exception).lower())

    def test_all_public_execution_failures_share_base_class(self):
        def model(p):
            return np.tan(p[0])

        try:
            autoia(model, [Interval(-0.1, 0.1)])
        except Exception as exc:
            self.assertIsInstance(exc, AutoIAError)
        else:
            self.fail("unsupported operation unexpectedly succeeded")

    def test_public_backend_environment_failure_is_typed(self):
        from unittest import mock
        from autoia import RigorousBinary64Backend, NumericalLimitError
        from autoia.trusted.environment import Binary64EnvironmentError
        with mock.patch(
            "autoia.trusted.binary64.require_binary64_environment",
            side_effect=Binary64EnvironmentError("synthetic environment failure"),
        ):
            with self.assertRaises(NumericalLimitError):
                RigorousBinary64Backend()


if __name__ == "__main__":
    unittest.main()
