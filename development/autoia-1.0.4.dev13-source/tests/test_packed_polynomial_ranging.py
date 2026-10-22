import itertools
import unittest
from fractions import Fraction

import numpy as np

from autoia.trusted import (
    Interval, RigorousBinary64Backend, rigorous_interval_subtract_point_arrays,
)
from autoia.polynomial import (
    compile_polynomial_range_plan,
    pack_point_polynomial_family,
    pack_interval_polynomial_family,
    range_packed_point_termwise,
    range_packed_interval_termwise,
    range_interval_polynomial_family,
)
from autoia.semantic.linear_solve import range_interval_coefficient_poly


def F(x):
    return Fraction.from_float(float(x))


def exact_interval_product(a_lo, a_hi, b_lo, b_hi):
    vals = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
    return min(vals), max(vals)


def exact_monomial_range(alpha, h):
    H = Fraction(1, 1)
    for ai, hi in zip(alpha, h):
        H *= F(hi) ** int(ai)
    if sum(int(v) for v in alpha) == 0:
        return Fraction(1, 1), Fraction(1, 1)
    if all(int(v) % 2 == 0 for v in alpha):
        return Fraction(0, 1), H
    return -H, H


def exact_termwise_point(coeff, support, h):
    lo = Fraction(0, 1)
    hi = Fraction(0, 1)
    for c, alpha in zip(coeff, support):
        ml, mh = exact_monomial_range(alpha, h)
        pl, ph = exact_interval_product(F(c), F(c), ml, mh)
        lo += pl
        hi += ph
    return lo, hi


def exact_termwise_interval(coeff_lo, coeff_hi, support, h):
    lo = Fraction(0, 1)
    hi = Fraction(0, 1)
    for cl, ch, alpha in zip(coeff_lo, coeff_hi, support):
        ml, mh = exact_monomial_range(alpha, h)
        pl, ph = exact_interval_product(F(cl), F(ch), ml, mh)
        lo += pl
        hi += ph
    return lo, hi


def assert_encloses_exact(testcase, got_lo, got_hi, exact_lo, exact_hi):
    testcase.assertLessEqual(F(got_lo), exact_lo)
    testcase.assertGreaterEqual(F(got_hi), exact_hi)


class TestPackedPolynomialRanging(unittest.TestCase):
    def setUp(self):
        self.backend = RigorousBinary64Backend(
            reduction_method="vectorized", vectorized_reduction_min_terms=1
        )

    def test_plan_is_canonical_cached_and_physical_only(self):
        support_a = [(1, 0, 2), (0, 0, 0), (0, 2, 0), (1, 0, 2)]
        support_b = list(reversed(support_a))
        p1 = compile_polynomial_range_plan(support_a, 3)
        p2 = compile_polynomial_range_plan(support_b, 3)
        self.assertIs(p1, p2)
        self.assertEqual(p1.support, ((0, 0, 0), (0, 2, 0), (1, 0, 2)))
        self.assertEqual(p1.degree_vector, (1, 2, 2))
        self.assertFalse(p1.exponents.flags.writeable)
        self.assertFalse(p1.even_monomial.flags.writeable)
        # There is deliberately no auxiliary-generator dimension in the plan.
        self.assertEqual(p1.exponents.shape[1], 3)

    def test_point_family_termwise_contains_exact_rational_natural_range(self):
        rng = np.random.default_rng(2026082901)
        m = 3
        support = [
            (0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 1, 0),
            (0, 0, 3), (2, 0, 1), (1, 2, 0), (0, 1, 2),
            (2, 1, 1), (3, 0, 0), (0, 4, 0), (1, 0, 4),
            (2, 2, 1), (3, 1, 1), (1, 3, 1), (2, 1, 2),
        ]
        h = np.array([0.17, 0.29, 0.41])
        coeff_arrays = {a: rng.normal(size=(2, 4)) for a in support}
        packed = pack_point_polynomial_family(coeff_arrays, m)
        lo, hi = range_packed_point_termwise(packed, h, self.backend)
        self.assertEqual(lo.shape, (2, 4))
        for idx in np.ndindex((2, 4)):
            coeff = [coeff_arrays[a][idx] for a in packed.plan.support]
            elo, ehi = exact_termwise_point(coeff, packed.plan.support, h)
            assert_encloses_exact(self, lo[idx], hi[idx], elo, ehi)

    def test_interval_minus_representative_contains_exact_rational_difference(self):
        rng = np.random.default_rng(2026082906)
        lo = rng.normal(size=(7, 9))
        width = rng.uniform(0.0, 1.0e-7, size=(7, 9))
        hi = lo + width
        rep = 0.5 * lo + 0.5 * hi
        rep = np.minimum(np.maximum(rep, lo), hi)
        elo, ehi = rigorous_interval_subtract_point_arrays(
            lo, hi, rep, self.backend
        )
        for idx in np.ndindex(lo.shape):
            exact_lo = F(lo[idx]) - F(rep[idx])
            exact_hi = F(hi[idx]) - F(rep[idx])
            assert_encloses_exact(self, elo[idx], ehi[idx], exact_lo, exact_hi)

    def test_interval_family_termwise_contains_exact_rational_natural_range(self):
        rng = np.random.default_rng(2026082902)
        m = 2
        support = [
            (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
            (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (2, 2),
            (0, 4), (3, 2), (2, 3), (4, 1), (1, 4),
        ]
        h = np.array([0.23, 0.37])
        coeff_arrays = {}
        for alpha in support:
            center = rng.normal(size=(3, 2))
            rad = rng.uniform(0.0, 1.0e-10, size=(3, 2))
            arr = np.empty((3, 2), dtype=object)
            for idx in np.ndindex(arr.shape):
                arr[idx] = Interval(center[idx] - rad[idx], center[idx] + rad[idx])
            coeff_arrays[alpha] = arr

        packed = pack_interval_polynomial_family(coeff_arrays, m)
        lo, hi = range_packed_interval_termwise(packed, h, self.backend)
        for idx in np.ndindex((3, 2)):
            clo = [packed.lower[idx + (t,)] for t in range(packed.plan.term_count)]
            chi = [packed.upper[idx + (t,)] for t in range(packed.plan.term_count)]
            elo, ehi = exact_termwise_interval(clo, chi, packed.plan.support, h)
            assert_encloses_exact(self, lo[idx], hi[idx], elo, ehi)

    def test_interval_family_matches_scalar_contract_for_termwise_and_portfolio(self):
        rng = np.random.default_rng(2026082903)
        h = np.array([0.31, 0.22])
        support = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        coeff_arrays = {}
        for alpha in support:
            arr = np.empty(5, dtype=object)
            for j in range(5):
                center = float(rng.normal())
                radius = float(rng.uniform(0.0, 1.0e-8))
                arr[j] = Interval(center - radius, center + radius)
            coeff_arrays[alpha] = arr

        for method in ("termwise", "portfolio"):
            fast, meta = range_interval_polynomial_family(
                coeff_arrays, h, self.backend, method
            )
            for j in range(5):
                scalar_poly = {alpha: coeff_arrays[alpha][j] for alpha in support}
                ref, _ = range_interval_coefficient_poly(
                    scalar_poly, h, self.backend, method
                )
                # Different rigorous reduction orders need not be bit-identical,
                # but two sound enclosures of the same set cannot be disjoint.
                self.assertLessEqual(max(fast[j].lo, ref.lo), min(fast[j].hi, ref.hi))
                self.assertEqual(meta[j]["lowered_execution"], "packed-family")
                self.assertEqual(meta[j]["term_count"], len(support))

    def test_family_axis_can_include_many_box_dimension(self):
        # The packed kernel is deliberately agnostic about leading dimensions.
        # Here axis 0 acts as a future many-box dimension N and axis 1 as output O.
        coeff_arrays = {
            (0,): np.arange(12, dtype=float).reshape(3, 4),
            (1,): np.full((3, 4), 0.25),
            (2,): np.full((3, 4), -0.1),
        }
        packed = pack_point_polynomial_family(coeff_arrays, 1)
        lo, hi = range_packed_point_termwise(packed, [0.5], self.backend)
        self.assertEqual(lo.shape, (3, 4))
        self.assertEqual(hi.shape, (3, 4))
        for n, o in itertools.product(range(3), range(4)):
            c = [coeff_arrays[a][n, o] for a in packed.plan.support]
            elo, ehi = exact_termwise_point(c, packed.plan.support, [0.5])
            assert_encloses_exact(self, lo[n, o], hi[n, o], elo, ehi)


if __name__ == "__main__":
    unittest.main()
