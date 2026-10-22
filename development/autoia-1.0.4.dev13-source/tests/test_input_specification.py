"""V1 input-specification boundary tests for exact-real physical intervals."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import math
import pickle
import sys
import unittest

import numpy as np

from autoia import Interval, InvalidInputError, autoia, prepare


def qfloat(x: float) -> Fraction:
    return Fraction.from_float(float(x))


class TestExactRealInputConstruction(unittest.TestCase):
    def test_direct_constructor_keeps_binary64_exact_semantics(self):
        I = Interval(0.1, 0.3)
        self.assertEqual(I.input_semantics, "binary64-exact")
        self.assertEqual(qfloat(I.lo), qfloat(0.1))
        self.assertEqual(qfloat(I.hi), qfloat(0.3))
        self.assertEqual(I.input_specification["semantics"], "binary64-exact")

    def test_exact_decimal_interval_outwardly_contains_intended_rationals(self):
        I = Interval.enclosing("0.1", "0.3")
        qlo = Fraction(1, 10)
        qhi = Fraction(3, 10)
        self.assertLessEqual(qfloat(I.lo), qlo)
        self.assertGreaterEqual(qfloat(I.hi), qhi)
        # These two decimal values round inward on opposite sides in ordinary
        # binary64, so this specifically checks the motivating failure mode.
        self.assertGreater(qfloat(0.1), qlo)
        self.assertLess(qfloat(0.3), qhi)
        self.assertEqual(I.input_semantics, "exact-real-enclosing")
        self.assertTrue(I.input_specification["containment_verified"])

    def test_negative_nonrepresentable_decimal_endpoints_use_correct_directions(self):
        I = Interval.enclosing("-0.3", "-0.1")
        self.assertLessEqual(qfloat(I.lo), Fraction(-3, 10))
        self.assertGreaterEqual(qfloat(I.hi), Fraction(-1, 10))
        self.assertTrue(np.isfinite(I.lo) and np.isfinite(I.hi))

    def test_fraction_and_decimal_inputs_are_exact_specs(self):
        A = Interval.enclosing(Fraction(1, 10), Fraction(3, 10))
        B = Interval.enclosing(Decimal("0.1"), Decimal("0.3"))
        self.assertEqual((A.lo, A.hi), (B.lo, B.hi))

    def test_binary64_representable_exact_endpoints_are_not_widened(self):
        I = Interval.enclosing(Fraction(1, 2), Fraction(3, 4))
        self.assertEqual(I.lo, 0.5)
        self.assertEqual(I.hi, 0.75)

    def test_nonrepresentable_exact_point_becomes_two_float_bracket(self):
        I = Interval.enclosing("0.1", "0.1")
        q = Fraction(1, 10)
        self.assertLessEqual(qfloat(I.lo), q)
        self.assertGreaterEqual(qfloat(I.hi), q)
        self.assertLess(I.lo, I.hi)
        self.assertEqual(I.hi, np.nextafter(I.lo, math.inf))

    def test_subnormal_neighborhood_is_enclosed_without_assuming_nonzero_float_conversion(self):
        min_sub = float.fromhex("0x0.0000000000001p-1022")
        I = Interval.enclosing("1e-324", "1e-324")
        self.assertEqual(I.lo, 0.0)
        self.assertEqual(I.hi, min_sub)
        q = Fraction(1, 10**324)
        self.assertLessEqual(qfloat(I.lo), q)
        self.assertGreaterEqual(qfloat(I.hi), q)

        J = Interval.enclosing("-1e-324", "-1e-324")
        self.assertEqual(J.lo, -min_sub)
        self.assertEqual(J.hi, -0.0)
        qn = -q
        self.assertLessEqual(qfloat(J.lo), qn)
        self.assertGreaterEqual(qfloat(J.hi), qn)

    def test_scientific_decimal_string_is_exact(self):
        I = Interval.enclosing("1.25e-2", "2.5e-2")
        self.assertLessEqual(qfloat(I.lo), Fraction(1, 80))
        self.assertGreaterEqual(qfloat(I.hi), Fraction(1, 40))

    def test_float_arguments_are_rejected_to_avoid_false_exact_decimal_claim(self):
        with self.assertRaises(InvalidInputError):
            Interval.enclosing(0.1, 0.3)

    def test_nonfinite_and_out_of_finite_binary64_range_fail_closed(self):
        for text in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(text=text):
                with self.assertRaises(InvalidInputError):
                    Interval.enclosing(text, text)
        with self.assertRaises(InvalidInputError):
            Interval.enclosing("1e400", "1e400")
        with self.assertRaises(InvalidInputError):
            Interval.enclosing("-1e400", "-1e400")

    def test_exact_endpoint_order_is_checked_before_machine_conversion(self):
        with self.assertRaises(InvalidInputError):
            Interval.enclosing("0.30000000000000000000000001", "0.3")

    def test_interval_metadata_survives_pickle_for_process_workflows(self):
        I = Interval.enclosing("0.1", "0.3")
        J = pickle.loads(pickle.dumps(I))
        self.assertEqual(J.input_semantics, "exact-real-enclosing")
        self.assertEqual(J.input_specification, I.input_specification)


class TestOutwardPadding(unittest.TestCase):
    def test_one_ulp_outward_padding(self):
        I = Interval.outward(0.1, 0.3)
        self.assertEqual(I.lo, np.nextafter(0.1, -math.inf))
        self.assertEqual(I.hi, np.nextafter(0.3, math.inf))
        self.assertEqual(I.input_semantics, "binary64-one-ulp-outward")

    def test_padding_point_zero_expands_both_sides(self):
        min_sub = float.fromhex("0x0.0000000000001p-1022")
        I = Interval.outward(0.0, 0.0)
        self.assertEqual(I.lo, -min_sub)
        self.assertEqual(I.hi, min_sub)

    def test_padding_max_finite_fails_closed_because_v1_requires_finite_endpoints(self):
        with self.assertRaises(InvalidInputError):
            Interval.outward(0.0, sys.float_info.max)


class TestResultInputMetadata(unittest.TestCase):
    @staticmethod
    def model(p):
        return p[0] * p[0] + 2.0

    def test_primary_result_records_exact_real_input_semantics(self):
        I = Interval.enclosing("0.1", "0.3")
        result = autoia(self.model, [I], order=4)
        self.assertEqual(result.input_semantics, ("exact-real-enclosing",))
        self.assertEqual(result.input_specification[0]["requested_lower_fraction"], "1/10")
        self.assertEqual(result.input_specification[0]["requested_upper_fraction"], "3/10")
        self.assertEqual(result.physical_input_box, ((I.lo, I.hi),))
        self.assertTrue(result.machine_rigorous)

    def test_prepared_cache_metadata_follows_current_request_semantics(self):
        session = prepare(self.model, order=4)
        exact = Interval.enclosing("0.1", "0.3")
        r1 = session.certify([exact])
        self.assertEqual(r1.input_semantics, ("exact-real-enclosing",))

        # Same stored machine interval, intentionally reconstructed with direct
        # binary64-exact semantics.  The numerical certificate may be reused,
        # but per-request provenance must not be stale.
        direct = Interval(exact.lo, exact.hi)
        r2 = session.certify([direct])
        self.assertEqual(r2.input_semantics, ("binary64-exact",))
        self.assertEqual(r2.cache_info["kind"], "exact-certificate")


if __name__ == "__main__":
    unittest.main()
