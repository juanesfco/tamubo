import unittest
import numpy as np

import autoia as migrated
from tests.support import oracle_v11 as oracle
from tests.support.models import spring_model, generic_coupled_model
from autoia.semantic import linear_solve as sem
from autoia.trusted import RigorousBinary64Backend


def assert_float_bits_equal(testcase, a, b, msg=None):
    testcase.assertEqual(float(a).hex(), float(b).hex(), msg)


def assert_array_bits_equal(testcase, a, b, msg=None):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    testcase.assertEqual(aa.shape, bb.shape, msg)
    for idx in np.ndindex(aa.shape):
        assert_float_bits_equal(testcase, aa[idx], bb[idx], f"{msg or ''} index={idx}")


def assert_interval_equal(testcase, a, b, msg=None):
    assert_float_bits_equal(testcase, a.lo, b.lo, msg)
    assert_float_bits_equal(testcase, a.hi, b.hi, msg)


class TestSemanticSolveRegression(unittest.TestCase):
    def setUp(self):
        self.backend = RigorousBinary64Backend()
        self.oracle_backend = oracle.RigorousBinary64Backend()
        self.A0 = np.array([[4.0, -0.75], [-0.75, 3.25]], dtype=float)
        self.A_i = [
            np.array([[0.2, 0.05], [0.05, -0.1]], dtype=float),
            np.array([[-0.1, 0.0], [0.0, 0.15]], dtype=float),
        ]
        self.b0 = np.array([1.2, -0.4], dtype=float)
        self.b_i = [
            np.array([0.1, -0.03], dtype=float),
            np.array([-0.02, 0.08], dtype=float),
        ]
        self.h = np.array([0.25, 0.4], dtype=float)

    def test_predictor_coefficients_match_oracle_bitwise(self):
        new = sem.build_solution_polynomials(
            self.A0, self.A_i, self.b0, self.b_i, order=4
        )
        old = oracle.build_solution_polynomials(
            self.A0, self.A_i, self.b0, self.b_i, order=4
        )
        self.assertEqual(set(new), set(old))
        for alpha in sorted(new):
            assert_array_bits_equal(self, new[alpha], old[alpha], f"alpha={alpha}")

    def test_residual_interval_coefficients_match_oracle_bitwise(self):
        Xnew = sem.build_solution_polynomials(
            self.A0, self.A_i, self.b0, self.b_i, order=3
        )
        Xold = oracle.build_solution_polynomials(
            self.A0, self.A_i, self.b0, self.b_i, order=3
        )
        new = sem.residual_coefficient_intervals(
            self.A0, self.A_i, self.b0, self.b_i, Xnew, self.backend
        )
        old = oracle.residual_coefficient_intervals(
            self.A0, self.A_i, self.b0, self.b_i, Xold, self.oracle_backend
        )
        self.assertEqual(set(new), set(old))
        for alpha in sorted(new):
            for j in range(len(new[alpha])):
                assert_interval_equal(self, new[alpha][j], old[alpha][j], f"alpha={alpha}, j={j}")

        gnew, rnew = sem.abs_bound_interval_vector_poly(new, self.h, self.backend)
        gold, rold = oracle.abs_bound_interval_vector_poly(old, self.h, self.oracle_backend)
        assert_array_bits_equal(self, gnew, gold)
        for a, b in zip(rnew, rold):
            assert_interval_equal(self, a, b)

    def test_componentwise_correction_matches_oracle_bitwise(self):
        c = np.array([1.2e-5, 4.7e-6, 7.5e-6], dtype=float)
        B = np.array([
            [0.08, 0.02, 0.01],
            [0.03, 0.12, 0.02],
            [0.01, 0.04, 0.09],
        ], dtype=float)
        d0 = 2.1e-5
        new, hnew = sem.refine_componentwise_correction(c, B, d0, self.backend)
        old, hold = oracle.refine_componentwise_correction(c, B, d0, self.oracle_backend)
        assert_array_bits_equal(self, new, old)
        self.assertEqual(len(hnew), len(hold))
        for a, b in zip(hnew, hold):
            assert_array_bits_equal(self, a, b)

    def test_polynomial_semantic_spring_matches_oracle_bitwise(self):
        boxes_new = [
            migrated.Interval(90.0, 110.0),
            migrated.Interval(900.0, 1100.0),
            migrated.Interval(9.5, 10.5),
        ]
        boxes_old = [
            oracle.Interval(90.0, 110.0),
            oracle.Interval(900.0, 1100.0),
            oracle.Interval(9.5, 10.5),
        ]
        new = migrated.autoia(
            spring_model, boxes_new, order=4,
            representation="polynomial", range_method="monotonicity"
        )
        old = oracle.autoia(
            oracle.spring_model, boxes_old, order=4,
            representation="polynomial", range_method="monotonicity"
        )
        assert_array_bits_equal(self, new.lower, old.lower)
        assert_array_bits_equal(self, new.upper, old.upper)
        self.assertEqual(len(new.certificates), len(old.certificates))
        cn, co = new.certificates[0], old.certificates[0]
        for key in ("g", "s", "rmax", "c", "c_legacy", "c_direct", "B", "d_componentwise"):
            assert_array_bits_equal(self, cn[key], co[key], key)
        for key in ("rho_upper", "eta_upper", "delta_lower", "d_upper"):
            assert_float_bits_equal(self, cn[key], co[key], key)

    def test_interval_semantic_coupled_matches_oracle_bitwise(self):
        boxes_new = [migrated.Interval(0.85, 1.15), migrated.Interval(0.75, 1.25)]
        boxes_old = [oracle.Interval(0.85, 1.15), oracle.Interval(0.75, 1.25)]
        new = migrated.autoia(
            generic_coupled_model, boxes_new, order=3,
            representation="interval", range_method="termwise"
        )
        old = oracle.autoia(
            oracle.generic_coupled_model, boxes_old, order=3,
            representation="interval", range_method="termwise"
        )
        assert_array_bits_equal(self, new.lower, old.lower)
        assert_array_bits_equal(self, new.upper, old.upper)
        self.assertEqual(len(new.certificates), len(old.certificates))
        cn, co = new.certificates[0], old.certificates[0]
        for key in ("c", "B", "d_componentwise"):
            assert_array_bits_equal(self, cn[key], co[key], key)
        for key in ("rho_upper", "d_scalar"):
            assert_float_bits_equal(self, cn[key], co[key], key)


if __name__ == "__main__":
    unittest.main()
