import math
import unittest
import numpy as np

from tests.support import oracle_v11 as oracle
from autoia.enclosures import (
    ParameterizedEnclosure,
    PolynomialEnclosure,
    IntervalEnclosure,
    PiecewiseEnclosure,
    ValidatedValue,
)
from autoia.trusted import Interval, RigorousBinary64Backend
from autoia.contracts import ScalarRepresentation


NEW_BACKEND = RigorousBinary64Backend()
OLD_BACKEND = oracle.RigorousBinary64Backend()


def bits(x):
    return np.float64(x).view(np.uint64).item()


def assert_interval_bits(testcase, a, b):
    testcase.assertEqual(bits(a.lo), bits(b.lo))
    testcase.assertEqual(bits(a.hi), bits(b.hi))


def assert_poly_bits(testcase, new, old):
    testcase.assertEqual(set(new.coeff), set(old.coeff))
    for alpha in new.coeff:
        testcase.assertEqual(bits(new.coeff[alpha]), bits(old.coeff[alpha]))
    assert_interval_bits(testcase, new.remainder, old.remainder)
    testcase.assertEqual(new.order, old.order)
    testcase.assertTrue(np.array_equal(new.h, old.h))


class TestEnclosureRegression(unittest.TestCase):
    def setUp(self):
        self.h = np.array([0.35, 0.2])
        self.order = 3

    def new_parameter(self, center, index):
        return PolynomialEnclosure.parameter(
            center, index, self.h, self.order, NEW_BACKEND, "termwise"
        )

    def old_parameter(self, center, index):
        return oracle.PolynomialEnclosure.parameter(
            center, index, self.h, self.order, OLD_BACKEND, "termwise"
        )

    def test_parameter_and_generic_arithmetic_match_oracle_bitwise(self):
        xn = self.new_parameter(1.2, 0)
        yn = self.new_parameter(-0.4, 1)
        xo = self.old_parameter(1.2, 0)
        yo = self.old_parameter(-0.4, 1)

        expressions = [
            (xn + 2.5 * yn - 0.1, xo + 2.5 * yo - 0.1),
            ((xn * yn + xn**2) * (1.0 - yn),
             (xo * yo + xo**2) * (1.0 - yo)),
        ]
        for new, old in expressions:
            assert_poly_bits(self, new, old)
            rn, _ = new.range("portfolio")
            ro, _ = old.range("portfolio")
            assert_interval_bits(self, rn, ro)

    def test_reciprocal_and_division_match_oracle_bitwise(self):
        xn = self.new_parameter(2.0, 0) + 0.1 * self.new_parameter(0.0, 1)
        xo = self.old_parameter(2.0, 0) + 0.1 * self.old_parameter(0.0, 1)

        invn = xn.reciprocal()
        invo = xo.reciprocal()
        assert_poly_bits(self, invn, invo)

        qn = (1.0 + self.new_parameter(0.0, 1)) / xn
        qo = (1.0 + self.old_parameter(0.0, 1)) / xo
        assert_poly_bits(self, qn, qo)

    def test_interval_enclosure_matches_oracle(self):
        In = Interval(-0.25, 0.75)
        Io = oracle.Interval(-0.25, 0.75)
        new = IntervalEnclosure(In, self.h, self.order, NEW_BACKEND, "termwise")
        old = oracle.IntervalEnclosure(Io, self.h, self.order, OLD_BACKEND, "termwise")

        rn, mn = new.range()
        ro, mo = old.range()
        assert_interval_bits(self, rn, ro)
        self.assertEqual(mn, mo)

    def test_validated_value_polynomial_path_matches_oracle(self):
        xn = ValidatedValue(self.new_parameter(1.2, 0))
        yn = ValidatedValue(self.new_parameter(-0.4, 1))
        xo = oracle.ValidatedValue(self.old_parameter(1.2, 0))
        yo = oracle.ValidatedValue(self.old_parameter(-0.4, 1))

        new = (xn * yn + 0.25) / (2.0 + xn)
        old = (xo * yo + 0.25) / (2.0 + xo)
        self.assertEqual(new.representation_name, old.representation_name)
        assert_poly_bits(self, new.rep, old.rep)

    def test_validated_value_interval_path_matches_oracle(self):
        new_a = ValidatedValue(IntervalEnclosure(
            Interval(0.8, 1.2), self.h, 2, NEW_BACKEND, "termwise"
        ))
        new_b = ValidatedValue(IntervalEnclosure(
            Interval(1.7, 2.1), self.h, 3, NEW_BACKEND, "termwise"
        ))
        old_a = oracle.ValidatedValue(oracle.IntervalEnclosure(
            oracle.Interval(0.8, 1.2), self.h, 2, OLD_BACKEND, "termwise"
        ))
        old_b = oracle.ValidatedValue(oracle.IntervalEnclosure(
            oracle.Interval(1.7, 2.1), self.h, 3, OLD_BACKEND, "termwise"
        ))

        new = (new_a + new_b) * new_b / new_a
        old = (old_a + old_b) * old_b / old_a
        self.assertEqual(new.representation_name, "interval")
        assert_interval_bits(self, new.rep.interval, old.rep.interval)
        self.assertEqual(new.rep.order, old.rep.order)

    def test_mixed_interval_polynomial_promotion_matches_oracle(self):
        pn = ValidatedValue(self.new_parameter(1.0, 0))
        po = oracle.ValidatedValue(self.old_parameter(1.0, 0))
        inew = ValidatedValue(IntervalEnclosure(
            Interval(0.4, 0.6), self.h, self.order,
            NEW_BACKEND, "termwise"
        ))
        iold = oracle.ValidatedValue(oracle.IntervalEnclosure(
            oracle.Interval(0.4, 0.6), self.h, self.order,
            OLD_BACKEND, "termwise"
        ))

        new = (pn + inew) * (2.0 - inew)
        old = (po + iold) * (2.0 - iold)
        self.assertEqual(new.representation_name, "polynomial")
        assert_poly_bits(self, new.rep, old.rep)

    def test_piecewise_hull_matches_oracle(self):
        b1n = IntervalEnclosure(
            Interval(-1.0, 0.3), self.h, 2, NEW_BACKEND, "termwise"
        )
        b2n = IntervalEnclosure(
            Interval(0.1, 2.0), self.h, 2, NEW_BACKEND, "termwise"
        )
        b1o = oracle.IntervalEnclosure(
            oracle.Interval(-1.0, 0.3), self.h, 2, OLD_BACKEND, "termwise"
        )
        b2o = oracle.IntervalEnclosure(
            oracle.Interval(0.1, 2.0), self.h, 2, OLD_BACKEND, "termwise"
        )
        new = PiecewiseEnclosure([b1n, b2n], NEW_BACKEND)
        old = oracle.PiecewiseEnclosure([b1o, b2o], OLD_BACKEND)
        rn, mn = new.range()
        ro, mo = old.range()
        assert_interval_bits(self, rn, ro)
        self.assertEqual(mn["num_branches"], mo["num_branches"])
        for a, b in zip(mn["branch_ranges"], mo["branch_ranges"]):
            assert_interval_bits(self, a, b)

    def test_parameterized_alias_is_exact_same_class(self):
        self.assertIs(ParameterizedEnclosure, PolynomialEnclosure)


class ToyRepresentation:
    """Minimal third representation used to test structural dispatch."""

    representation_name = "toy"
    promotion_priority = 10
    supports_value_arithmetic = True

    def __init__(self, interval, h, order, backend, range_method="termwise"):
        self.interval = interval
        self.h = np.asarray(h, dtype=float)
        self.m = len(self.h)
        self.order = int(order)
        self.backend = backend
        self.range_method = range_method

    def copy(self):
        return ToyRepresentation(
            self.interval, self.h.copy(), self.order,
            self.backend, self.range_method,
        )

    def range(self, method=None):
        return self.interval, {"method": "toy"}

    def constant_like(self, value):
        x = Interval(float(value), float(value))
        return ToyRepresentation(x, self.h, self.order, self.backend, self.range_method)

    def negated(self):
        return ToyRepresentation(
            Interval(-self.interval.hi, -self.interval.lo),
            self.h, self.order, self.backend, self.range_method,
        )

    def binary_same(self, other, op):
        if op == "add":
            I = self.backend.add(self.interval, other.interval)
        elif op == "mul":
            I = self.backend.mul(self.interval, other.interval)
        elif op == "div":
            I = self.backend.div(self.interval, other.interval)
        else:
            raise ValueError(op)
        return ToyRepresentation(
            I, self.h, min(self.order, other.order),
            self.backend, self.range_method,
        )

    def convert_from(self, other):
        r, _ = other.range(getattr(other, "range_method", None))
        return ToyRepresentation(
            r, other.h, other.order, other.backend, other.range_method
        )


class TestRepresentationContract(unittest.TestCase):
    def test_validated_value_accepts_third_structural_representation(self):
        h = np.array([0.1])
        raw = ToyRepresentation(Interval(1.0, 1.1), h, 2, NEW_BACKEND)
        self.assertIsInstance(raw, ScalarRepresentation)
        a = ValidatedValue(raw)
        b = ValidatedValue(ToyRepresentation(
            Interval(2.0, 2.2), h, 2, NEW_BACKEND
        ))
        c = a * b + 1.0
        self.assertEqual(c.representation_name, "toy")
        r, _ = c.range()
        self.assertLessEqual(r.lo, 3.0)
        self.assertGreaterEqual(r.hi, 3.42)

    def test_mixed_representation_uses_promotion_contract_not_concrete_type(self):
        h = np.array([0.1])
        toy = ValidatedValue(ToyRepresentation(
            Interval(1.0, 1.1), h, 2, NEW_BACKEND
        ))
        interval = ValidatedValue(IntervalEnclosure(
            Interval(2.0, 2.2), h, 2, NEW_BACKEND, "termwise"
        ))
        result = toy + interval
        self.assertEqual(result.representation_name, "toy")
        r, _ = result.range()
        self.assertLessEqual(r.lo, 3.0)
        self.assertGreaterEqual(r.hi, 3.3)


if __name__ == "__main__":
    unittest.main()
