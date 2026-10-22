import itertools
import unittest
from fractions import Fraction
from math import comb

import numpy as np

from autoia.trusted import RigorousBinary64Backend
from autoia.polynomial import (
    compile_polynomial_range_plan,
    pack_point_polynomial_family,
    bernstein_axis_transform_endpoints,
    packed_bernstein_coefficients,
    range_packed_point_bernstein,
    range_packed_point_reduced_bernstein,
    range_point_polynomial_family,
    range_packed_point_monotonicity,
)
from autoia.polynomial.ranging import (
    range_bernstein_scalar,
    range_monotonicity_scalar,
    range_monotonicity_reduced_bernstein_scalar,
)


def F(x):
    return Fraction.from_float(float(x))


def exact_axis_weight(degree, h, gamma, alpha):
    degree = int(degree)
    gamma = int(gamma)
    alpha = int(alpha)
    H = F(h)
    total = Fraction(0)
    for beta in range(min(alpha, gamma) + 1):
        total += (
            Fraction(comb(alpha, beta))
            * ((-H) ** (alpha - beta))
            * ((2 * H) ** beta)
            * Fraction(comb(gamma, beta), comb(degree, beta))
        )
    return total


def exact_bernstein_coefficient(coeff, support, degrees, h, gamma, fixed=None, unresolved=None):
    fixed = {} if fixed is None else dict(fixed)
    if unresolved is None:
        unresolved = tuple(i for i in range(len(degrees)) if i not in fixed)
    unresolved = tuple(unresolved)
    total = Fraction(0)
    for c, alpha in zip(coeff, support):
        term = F(c)
        for i, value in fixed.items():
            term *= F(value) ** int(alpha[i])
        for ui, i in enumerate(unresolved):
            term *= exact_axis_weight(
                int(degrees[i]), h[i], gamma[ui], int(alpha[i])
            )
        total += term
    return total


def assert_contains_fraction(testcase, lo, hi, value):
    testcase.assertLessEqual(F(lo), value)
    testcase.assertGreaterEqual(F(hi), value)


class TestPackedPolynomialBernstein(unittest.TestCase):
    def setUp(self):
        self.backend = RigorousBinary64Backend(
            reduction_method="vectorized", vectorized_reduction_min_terms=1
        )

    def test_dense_structure_is_compiled_and_immutable(self):
        plan = compile_polynomial_range_plan(
            [(0, 0, 0), (2, 0, 1), (0, 3, 0), (1, 1, 0)], 3
        )
        self.assertEqual(plan.dense_shape, (3, 4, 2))
        self.assertEqual(plan.dense_term_index.shape, (len(plan.support),))
        self.assertFalse(plan.dense_term_index.flags.writeable)
        for t, alpha in enumerate(plan.support):
            self.assertEqual(
                int(plan.dense_term_index[t]),
                int(np.ravel_multi_index(alpha, plan.dense_shape)),
            )

    def test_one_dimensional_transform_entries_enclose_exact_rationals(self):
        for degree in range(6):
            h = 0.37
            lo, hi = bernstein_axis_transform_endpoints(degree, h, self.backend)
            self.assertEqual(lo.shape, (degree + 1, degree + 1))
            for gamma in range(degree + 1):
                for alpha in range(degree + 1):
                    exact = exact_axis_weight(degree, h, gamma, alpha)
                    assert_contains_fraction(self, lo[gamma, alpha], hi[gamma, alpha], exact)

    def test_full_coefficients_enclose_exact_multivariate_bernstein_coefficients(self):
        rng = np.random.default_rng(2026083004)
        support = [
            (0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3),
            (1, 1, 0), (2, 0, 1), (1, 2, 1), (2, 1, 2),
        ]
        h = np.array([0.17, 0.29, 0.41])
        coeff_arrays = {alpha: rng.normal(size=(2, 3)) for alpha in support}
        packed = pack_point_polynomial_family(coeff_arrays, 3)
        lo, hi, meta = packed_bernstein_coefficients(packed, h, self.backend)
        self.assertEqual(lo.shape, (2, 3) + packed.plan.dense_shape)
        self.assertEqual(meta["unresolved_variables"], (0, 1, 2))
        for idx in np.ndindex((2, 3)):
            coeff = [coeff_arrays[a][idx] for a in packed.plan.support]
            for gamma in itertools.product(
                *[range(d + 1) for d in packed.plan.degree_vector]
            ):
                exact = exact_bernstein_coefficient(
                    coeff, packed.plan.support, packed.plan.degree_vector, h, gamma
                )
                assert_contains_fraction(self, lo[idx + gamma], hi[idx + gamma], exact)

    def test_fixed_dimension_coefficients_enclose_exact_reduced_coefficients(self):
        rng = np.random.default_rng(2026083005)
        support = [
            (0, 0, 0, 0), (1, 0, 0, 0), (0, 2, 0, 0),
            (0, 0, 3, 0), (0, 0, 0, 2), (1, 1, 0, 0),
            (2, 0, 1, 0), (1, 0, 0, 2), (0, 2, 1, 1),
        ]
        h = np.array([0.13, 0.21, 0.31, 0.27])
        coeff_arrays = {alpha: rng.normal(size=4) for alpha in support}
        packed = pack_point_polynomial_family(coeff_arrays, 4)
        fixed = {0: -float(h[0]), 2: +float(h[2])}
        unresolved = (1, 3)
        lo, hi, meta = packed_bernstein_coefficients(
            packed, h, self.backend, fixed=fixed, unresolved=unresolved
        )
        expected_shape = (4, packed.plan.degree_vector[1] + 1, packed.plan.degree_vector[3] + 1)
        self.assertEqual(lo.shape, expected_shape)
        self.assertEqual(meta["unresolved_variables"], unresolved)
        for j in range(4):
            coeff = [coeff_arrays[a][j] for a in packed.plan.support]
            for gamma in itertools.product(
                range(packed.plan.degree_vector[1] + 1),
                range(packed.plan.degree_vector[3] + 1),
            ):
                exact = exact_bernstein_coefficient(
                    coeff, packed.plan.support, packed.plan.degree_vector, h,
                    gamma, fixed=fixed, unresolved=unresolved,
                )
                assert_contains_fraction(self, lo[(j,) + gamma], hi[(j,) + gamma], exact)

    def test_full_family_range_overlaps_scalar_bernstein_oracle(self):
        rng = np.random.default_rng(2026083006)
        support = [
            (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
            (3, 0), (2, 1), (1, 2), (0, 3),
        ]
        h = np.array([0.23, 0.37])
        coeff_arrays = {alpha: rng.normal(size=8) for alpha in support}
        packed = pack_point_polynomial_family(coeff_arrays, 2)
        lo, hi, meta = range_packed_point_bernstein(packed, h, self.backend)
        self.assertEqual(meta["method"], "bernstein")
        for j in range(8):
            p = {a: float(coeff_arrays[a][j]) for a in support}
            ref, _ = range_bernstein_scalar(p, h, self.backend)
            self.assertLessEqual(max(lo[j], ref.lo), min(hi[j], ref.hi))

    def test_reduced_family_range_overlaps_scalar_reduced_oracle(self):
        # Dominant linear terms deliberately create several different rigorous
        # monotonicity patterns across one family.
        h = np.array([0.2, 0.25, 0.3])
        coeff_arrays = {
            (0, 0, 0): np.array([0.0, 0.1, -0.2, 0.3]),
            (1, 0, 0): np.array([2.0, -2.0, 2.0, -2.0]),
            (0, 1, 0): np.array([1.5, 1.5, -1.5, -1.5]),
            (0, 0, 1): np.array([0.0, 0.0, 1.2, -1.2]),
            (2, 0, 0): np.array([0.05, 0.05, 0.05, 0.05]),
            (0, 2, 0): np.array([-0.03, -0.03, -0.03, -0.03]),
            (0, 0, 2): np.array([0.4, 0.4, 0.02, 0.02]),
            (1, 1, 0): np.array([0.01, -0.01, 0.01, -0.01]),
        }
        packed = pack_point_polynomial_family(coeff_arrays, 3)
        _, _, mm = range_packed_point_monotonicity(packed, h, self.backend)
        lo, hi, meta = range_packed_point_reduced_bernstein(
            packed, h, self.backend, monotonicity_meta=mm
        )
        for j in range(4):
            p = {a: float(coeff_arrays[a][j]) for a in coeff_arrays}
            _, scalar_mm = range_monotonicity_scalar(p, h, self.backend)
            ref, _ = range_monotonicity_reduced_bernstein_scalar(
                p, h, self.backend, monotonicity_meta=scalar_mm
            )
            self.assertLessEqual(max(lo[j], ref.lo), min(hi[j], ref.hi))
            self.assertEqual(meta[j]["method"], "monotonicity-reduced-bernstein")
            self.assertEqual(
                meta[j]["lowered_execution"],
                "packed-separable-reduced-bernstein",
            )

    def test_family_bernstein_supports_leading_many_box_axis(self):
        coeff_arrays = {
            (0, 0): np.arange(12, dtype=float).reshape(3, 4),
            (1, 0): np.full((3, 4), 0.25),
            (0, 1): np.full((3, 4), -0.15),
            (1, 1): np.full((3, 4), 0.05),
        }
        ranges, meta = range_point_polynomial_family(
            coeff_arrays, [0.4, 0.3], self.backend, "bernstein"
        )
        self.assertEqual(ranges.shape, (3, 4))
        for idx in np.ndindex((3, 4)):
            self.assertEqual(meta[idx]["lowered_execution"], "packed-separable-bernstein")

    def test_portfolio_preserves_phase2_compatibility_tag_and_records_phase3_kernel(self):
        coeff_arrays = {
            (0, 0): np.array([1.0, 2.0]),
            (1, 0): np.array([0.4, -0.3]),
            (0, 1): np.array([0.2, 0.1]),
            (2, 0): np.array([0.05, 0.04]),
        }
        _, meta = range_point_polynomial_family(
            coeff_arrays, [0.2, 0.15], self.backend, "portfolio"
        )
        for j in range(2):
            self.assertEqual(meta[j]["lowered_execution"], "packed-cheap-scalar-bernstein")
            self.assertEqual(
                meta[j]["bernstein_execution"],
                "packed-separable-reduced-bernstein",
            )


if __name__ == "__main__":
    unittest.main()
