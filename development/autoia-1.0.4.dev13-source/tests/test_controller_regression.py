import unittest
import numpy as np

import autoia as migrated
from tests.support import oracle_v11 as oracle
from tests.support.models import generic_coupled_model, generic_ratio_model, hidden_monotone_model
from autoia.controller import RangePolicy, SubdivisionPolicy


def assert_float_bits_equal(testcase, a, b, msg=None):
    testcase.assertEqual(float(a).hex(), float(b).hex(), msg)


def assert_array_bits_equal(testcase, a, b, msg=None):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    testcase.assertEqual(aa.shape, bb.shape, msg)
    for idx in np.ndindex(aa.shape):
        assert_float_bits_equal(testcase, aa[idx], bb[idx], f"{msg or ''} index={idx}")



def canonical(obj):
    if hasattr(obj, "lo") and hasattr(obj, "hi"):
        return ("Interval", float(obj.lo).hex(), float(obj.hi).hex())
    if isinstance(obj, np.ndarray):
        return ("ndarray", obj.shape, tuple(float(x).hex() for x in np.asarray(obj, dtype=float).flat))
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(canonical(v) for v in obj)
    if isinstance(obj, (np.floating, float)):
        return ("float", float(obj).hex())
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


class TestControllerRegression(unittest.TestCase):
    def test_default_coupled_controller_matches_oracle_bitwise(self):
        new_box = [migrated.Interval(-0.2, 0.2), migrated.Interval(-0.5, 0.5)]
        old_box = [oracle.Interval(-0.2, 0.2), oracle.Interval(-0.5, 0.5)]
        new = migrated.autoia(generic_coupled_model, new_box, order=5)
        old = oracle.autoia(oracle.generic_coupled_model, old_box, order=5)
        assert_array_bits_equal(self, new.lower, old.lower)
        assert_array_bits_equal(self, new.upper, old.upper)
        self.assertEqual(canonical(new.final_range_diagnostics), canonical(old.final_range_diagnostics))
        self.assertEqual(canonical(new.range_policy_events), canonical(old.range_policy_events))

    def test_explicit_policy_matches_oracle_bitwise(self):
        new_policy = RangePolicy(
            internal_method="termwise", final_method="adaptive",
            bernstein_max_coefficients=16,
            proof_bernstein_max_coefficients=64,
        )
        old_policy = oracle.RangePolicy(
            internal_method="termwise", final_method="adaptive",
            bernstein_max_coefficients=16,
            proof_bernstein_max_coefficients=64,
        )
        new_box = [migrated.Interval(-0.4, 0.4), migrated.Interval(-0.5, 0.5)]
        old_box = [oracle.Interval(-0.4, 0.4), oracle.Interval(-0.5, 0.5)]
        new = migrated.autoia(generic_ratio_model, new_box, order=5, range_policy=new_policy)
        old = oracle.autoia(oracle.generic_ratio_model, old_box, order=5, range_policy=old_policy)
        assert_array_bits_equal(self, new.lower, old.lower)
        assert_array_bits_equal(self, new.upper, old.upper)
        self.assertEqual(canonical(new.final_range_diagnostics), canonical(old.final_range_diagnostics))
        self.assertEqual(canonical(new.range_policy_events), canonical(old.range_policy_events))

    def test_fixed_subdivision_matches_oracle_bitwise(self):
        new_box = [migrated.Interval(-0.2, 0.2), migrated.Interval(-0.5, 0.5)]
        old_box = [oracle.Interval(-0.2, 0.2), oracle.Interval(-0.5, 0.5)]
        new = migrated.autoia_subdivide(
            generic_coupled_model, new_box, [2, 2], order=4,
        )
        old = oracle.autoia_subdivide(
            oracle.generic_coupled_model, old_box, [2, 2], order=4,
        )
        assert_array_bits_equal(self, new.lower, old.lower)
        assert_array_bits_equal(self, new.upper, old.upper)
        self.assertEqual(len(new.branch_results), len(old.branch_results))
        for nr, orr in zip(new.branch_results, old.branch_results):
            assert_array_bits_equal(self, nr.lower, orr.lower)
            assert_array_bits_equal(self, nr.upper, orr.upper)

    def test_adaptive_subdivision_matches_oracle_bitwise(self):
        new_box = [migrated.Interval(-0.6, 0.6) for _ in range(4)]
        old_box = [oracle.Interval(-0.6, 0.6) for _ in range(4)]
        new_rp = RangePolicy(bernstein_max_coefficients=16)
        old_rp = oracle.RangePolicy(bernstein_max_coefficients=16)
        new_sp = SubdivisionPolicy(max_branches=4, max_depth=3, max_probe_directions=2, split_method="lookahead")
        old_sp = oracle.SubdivisionPolicy(max_branches=4, max_depth=3, max_probe_directions=2)
        new = migrated.autoia_adaptive_subdivide(
            hidden_monotone_model, new_box, order=3,
            range_policy=new_rp, subdivision_policy=new_sp,
        )
        old = oracle.autoia_adaptive_subdivide(
            hidden_monotone_model, old_box, order=3,
            range_policy=old_rp, subdivision_policy=old_sp,
        )
        assert_array_bits_equal(self, new.lower, old.lower)
        assert_array_bits_equal(self, new.upper, old.upper)
        self.assertEqual(new.branch_depths, old.branch_depths)
        self.assertEqual(new.subdivision_events, old.subdivision_events)
        self.assertEqual(len(new.branch_results), len(old.branch_results))


if __name__ == "__main__":
    unittest.main()
