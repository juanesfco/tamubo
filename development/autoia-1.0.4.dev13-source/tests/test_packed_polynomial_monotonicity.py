import unittest
from fractions import Fraction

import numpy as np

from autoia.trusted import Interval, RigorousBinary64Backend
from autoia.polynomial import (
    compile_polynomial_range_plan,
    pack_point_polynomial_family,
    range_packed_point_derivatives,
    range_packed_point_monotonicity,
    range_point_polynomial_family,
)
from autoia.polynomial.ranging import range_monotonicity_scalar
from autoia.semantic.linear_solve import range_point_polynomial_array


def F(x):
    return Fraction.from_float(float(x))


def exact_interval_product(a_lo, a_hi, b_lo, b_hi):
    vals = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
    return min(vals), max(vals)


def exact_variable_power_range(power, h, sign, lower_problem):
    power = int(power)
    if power == 0:
        return Fraction(1), Fraction(1)
    H = F(h) ** power
    sign = int(sign)
    if sign != 0:
        if sign > 0:
            x = -F(h) if lower_problem else F(h)
        else:
            x = F(h) if lower_problem else -F(h)
        value = x ** power
        return value, value
    if power % 2 == 0:
        return Fraction(0), H
    return -H, H


def exact_monomial_range_with_sign(alpha, h, signs, lower_problem):
    lo = Fraction(1)
    hi = Fraction(1)
    for power, hj, sign in zip(alpha, h, signs):
        vlo, vhi = exact_variable_power_range(power, hj, sign, lower_problem)
        lo, hi = exact_interval_product(lo, hi, vlo, vhi)
    return lo, hi


def exact_derivative_natural_range(coeff, support, h, variable):
    lo = Fraction(0)
    hi = Fraction(0)
    for c, alpha in zip(coeff, support):
        k = int(alpha[variable])
        if k == 0:
            continue
        beta = list(alpha)
        beta[variable] -= 1
        # Symmetric monomial natural range.
        H = Fraction(1)
        for power, hj in zip(beta, h):
            H *= F(hj) ** int(power)
        if sum(beta) == 0:
            ml, mh = Fraction(1), Fraction(1)
        elif all(int(v) % 2 == 0 for v in beta):
            ml, mh = Fraction(0), H
        else:
            ml, mh = -H, H
        scale = F(c) * k
        pl, ph = exact_interval_product(scale, scale, ml, mh)
        lo += pl
        hi += ph
    return lo, hi


def exact_fixed_termwise_range(coeff, support, h, signs, lower_problem):
    lo = Fraction(0)
    hi = Fraction(0)
    for c, alpha in zip(coeff, support):
        ml, mh = exact_monomial_range_with_sign(alpha, h, signs, lower_problem)
        pl, ph = exact_interval_product(F(c), F(c), ml, mh)
        lo += pl
        hi += ph
    return lo, hi


def assert_encloses_exact(testcase, got_lo, got_hi, exact_lo, exact_hi):
    testcase.assertLessEqual(F(got_lo), exact_lo)
    testcase.assertGreaterEqual(F(got_hi), exact_hi)


class TestPackedPolynomialMonotonicity(unittest.TestCase):
    def setUp(self):
        self.backend = RigorousBinary64Backend(
            reduction_method="vectorized", vectorized_reduction_min_terms=1
        )

    def test_derivative_structure_is_compiled_and_immutable(self):
        support = [(0, 0, 0), (2, 0, 1), (0, 3, 0), (1, 1, 0)]
        plan = compile_polynomial_range_plan(support, 3)
        self.assertEqual(plan.derivative_term_index.shape, (3, len(plan.support)))
        self.assertEqual(plan.derivative_multipliers.shape, (3, len(plan.support)))
        self.assertFalse(plan.derivative_term_index.flags.writeable)
        self.assertFalse(plan.derivative_multipliers.flags.writeable)
        # d/dq0 of q0^2 q2 is 2 q0 q2, and d/dq2 is q0^2.
        t = plan.support.index((2, 0, 1))
        j0 = int(plan.derivative_term_index[0, t])
        j2 = int(plan.derivative_term_index[2, t])
        self.assertEqual(plan.derivative_support[j0], (1, 0, 1))
        self.assertEqual(plan.derivative_support[j2], (2, 0, 0))
        self.assertEqual(int(plan.derivative_multipliers[0, t]), 2)
        self.assertEqual(int(plan.derivative_multipliers[2, t]), 1)
        # Auxiliary shared generators do not appear in the derivative plan.
        self.assertEqual(plan.derivative_term_index.shape[0], 3)

    def test_batched_derivatives_enclose_exact_rational_natural_ranges(self):
        rng = np.random.default_rng(2026083001)
        m = 4
        support = [
            (0, 0, 0, 0), (1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 3, 0),
            (0, 0, 0, 2), (1, 1, 0, 0), (2, 0, 1, 0), (1, 0, 0, 2),
            (0, 2, 1, 0), (1, 1, 1, 0), (0, 1, 0, 3), (2, 1, 0, 1),
            (1, 0, 2, 1), (0, 3, 0, 1), (2, 0, 2, 0),
        ]
        h = np.array([0.13, 0.21, 0.31, 0.27])
        coeff_arrays = {alpha: rng.normal(size=(3, 5)) for alpha in support}
        packed = pack_point_polynomial_family(coeff_arrays, m)
        lo, hi = range_packed_point_derivatives(packed, h, self.backend)
        self.assertEqual(lo.shape, (3, 5, m))
        for idx in np.ndindex((3, 5)):
            coeff = [coeff_arrays[a][idx] for a in packed.plan.support]
            for i in range(m):
                elo, ehi = exact_derivative_natural_range(
                    coeff, packed.plan.support, h, i
                )
                assert_encloses_exact(self, lo[idx + (i,)], hi[idx + (i,)], elo, ehi)

    def test_monotonicity_signs_are_proof_valid_and_fixed_ranges_are_rigorous(self):
        rng = np.random.default_rng(2026083002)
        m = 3
        support = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0),
            (1, 0, 1), (0, 1, 1), (2, 1, 0), (1, 2, 0),
        ]
        h = np.array([0.22, 0.34, 0.19])
        coeff_arrays = {alpha: rng.normal(size=(4, 3)) for alpha in support}
        packed = pack_point_polynomial_family(coeff_arrays, m)
        lo, hi, meta = range_packed_point_monotonicity(packed, h, self.backend)
        self.assertEqual(meta["signs"].shape, (4, 3, m))

        for idx in np.ndindex((4, 3)):
            coeff = [coeff_arrays[a][idx] for a in packed.plan.support]
            signs = [int(v) for v in meta["signs"][idx]]
            for i, sign in enumerate(signs):
                elo, ehi = exact_derivative_natural_range(
                    coeff, packed.plan.support, h, i
                )
                if sign > 0:
                    self.assertGreaterEqual(elo, 0)
                elif sign < 0:
                    self.assertLessEqual(ehi, 0)

            exact_lo_lo, exact_lo_hi = exact_fixed_termwise_range(
                coeff, packed.plan.support, h, signs, True
            )
            exact_hi_lo, exact_hi_hi = exact_fixed_termwise_range(
                coeff, packed.plan.support, h, signs, False
            )
            assert_encloses_exact(
                self,
                meta["lower_fixed_range_lower"][idx],
                meta["lower_fixed_range_upper"][idx],
                exact_lo_lo,
                exact_lo_hi,
            )
            assert_encloses_exact(
                self,
                meta["upper_fixed_range_lower"][idx],
                meta["upper_fixed_range_upper"][idx],
                exact_hi_lo,
                exact_hi_hi,
            )
            self.assertLessEqual(F(lo[idx]), exact_lo_lo)
            self.assertGreaterEqual(F(hi[idx]), exact_hi_hi)

    def test_hand_constructed_monotone_family_matches_expected_signs(self):
        h = np.array([0.2, 0.3, 0.4])
        coeff_arrays = {
            (0, 0, 0): np.array([0.0, 1.0, -1.0]),
            (1, 0, 0): np.array([2.0, 1.0, 0.5]),
            (0, 1, 0): np.array([-3.0, 0.8, -0.5]),
            (0, 0, 1): np.array([0.7, -0.6, 0.9]),
            (2, 0, 0): np.array([0.05, 0.02, 0.01]),
        }
        packed = pack_point_polynomial_family(coeff_arrays, 3)
        _, _, meta = range_packed_point_monotonicity(packed, h, self.backend)
        expected = np.array([[1, -1, 1], [1, 1, -1], [1, -1, 1]], dtype=np.int8)
        np.testing.assert_array_equal(meta["signs"], expected)

    def test_packed_family_monotonicity_overlaps_scalar_oracle(self):
        rng = np.random.default_rng(2026083003)
        h = np.array([0.25, 0.18, 0.33])
        support = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0),
            (1, 0, 1), (0, 1, 1), (2, 1, 0), (0, 2, 1),
        ]
        coeff_arrays = {alpha: rng.normal(size=9) for alpha in support}
        ranges, meta = range_point_polynomial_family(
            coeff_arrays, h, self.backend, "monotonicity"
        )
        for j in range(9):
            p = {alpha: float(coeff_arrays[alpha][j]) for alpha in support}
            ref, ref_meta = range_monotonicity_scalar(p, h, self.backend)
            self.assertLessEqual(max(ranges[j].lo, ref.lo), min(ranges[j].hi, ref.hi))
            self.assertEqual(meta[j]["method"], "monotonicity")
            self.assertEqual(meta[j]["lowered_execution"], "packed-family")
            # A different rigorous reduction order may certify fewer monotone
            # directions, but any nonzero packed sign must be proof-valid.
            self.assertEqual(len(meta[j]["signs"]), len(ref_meta["signs"]))

    def test_semantic_array_portfolio_uses_phase2_lowering(self):
        h = np.array([0.2, 0.15])
        coeff_arrays = {
            (0, 0): np.array([[1.0, 0.8], [1.2, 0.9]]),
            (1, 0): np.array([[0.4, -0.2], [0.1, 0.3]]),
            (0, 1): np.array([[0.1, 0.2], [-0.3, 0.4]]),
            (2, 0): np.array([[0.05, 0.03], [0.02, -0.04]]),
        }
        rem = np.empty((2, 2), dtype=object)
        for idx in np.ndindex(rem.shape):
            rem[idx] = Interval(-1.0e-10, 1.0e-10)
        ranges, meta = range_point_polynomial_array(
            coeff_arrays, rem, h, self.backend, "portfolio"
        )
        self.assertEqual(ranges.shape, (2, 2))
        for idx in np.ndindex(ranges.shape):
            self.assertEqual(
                meta[idx]["lowered_execution"],
                "packed-cheap-scalar-bernstein",
            )

    def test_portfolio_uses_packed_cheap_components_and_scalar_bernstein(self):
        h = np.array([0.35, 0.27])
        coeff_arrays = {
            (0, 0): np.array([0.2, -0.1]),
            (1, 0): np.array([1.0, 0.7]),
            (0, 1): np.array([-0.4, 0.8]),
            (2, 0): np.array([0.3, -0.2]),
            (1, 1): np.array([0.1, 0.25]),
        }
        ranges, meta = range_point_polynomial_family(
            coeff_arrays, h, self.backend, "portfolio"
        )
        self.assertEqual(ranges.shape, (2,))
        for j in range(2):
            self.assertEqual(meta[j]["lowered_execution"], "packed-cheap-scalar-bernstein")
            self.assertIn("monotonicity_meta", meta[j])
            self.assertIn("bernstein_meta", meta[j])


if __name__ == "__main__":
    unittest.main()
