"""Step-10a tests for general physical-domain predicate certification."""

from __future__ import annotations

import math
import unittest
import numpy as np

from autoia import autoia, prepare, Interval, CertificationError
from autoia.policies import BranchPolicy


class TestDomainProofs(unittest.TestCase):
    def test_sqrt_exact_physical_boundary_without_clipping(self):
        def model(p):
            return np.sqrt(p[0])

        R = autoia(model, [Interval(0.0, 4.0)], order=6)
        self.assertLessEqual(R.lower[0], 0.0)
        self.assertGreaterEqual(R.upper[0], 2.0)
        events = [e for e in R.range_policy_events if e.get("kind") == "domain-proof"]
        self.assertEqual(events[0]["final_range"], Interval(0.0, 4.0))
        self.assertEqual(events[0]["methods_tried"], ("natural-physical-box",))
        self.assertIsNotNone(R.objects[0].natural_interval)

    def test_structural_proof_can_avoid_natural_replay(self):
        def model(p):
            z = p[0] - 0.5
            return np.sqrt(z*z + 0.01)

        R = autoia(model, [Interval(0.0, 1.0)], order=6)
        events = [e for e in R.range_policy_events if e.get("kind") == "domain-proof"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "proved")
        self.assertEqual(events[0]["methods_tried"], ("termwise",))
        for x in np.linspace(0.0, 1.0, 501):
            y = math.sqrt((x - 0.5)**2 + 0.01)
            self.assertLessEqual(R.lower[0], y)
            self.assertGreaterEqual(R.upper[0], y)

    def test_domain_refinement_certifies_valid_smooth_model(self):
        # Whole-box interval, termwise, monotonicity, and Bernstein proofs all
        # remain too wide here.  Re-executing on smaller physical boxes resolves
        # the universal domain predicate without path forking.
        def model(p):
            z = p[0] - 0.3
            return np.sqrt(z*z + 0.001)

        R = autoia(model, [Interval(0.0, 1.0)], order=6)
        self.assertTrue(hasattr(R, "path_events"))
        splits = [e for e in R.path_events if e.get("action") == "split-unresolved-domain"]
        self.assertGreaterEqual(len(splits), 1)
        self.assertTrue(all(e["operation"] == "sqrt" for e in splits))
        for x in np.linspace(0.0, 1.0, 1001):
            y = math.sqrt((x - 0.3)**2 + 0.001)
            self.assertLessEqual(R.lower[0], y)
            self.assertGreaterEqual(R.upper[0], y)

    def test_domain_refinement_budget_is_fail_closed(self):
        def model(p):
            z = p[0] - 0.3
            return np.sqrt(z*z + 0.001)

        policy = BranchPolicy(max_domain_split_depth_per_obligation=0)
        with self.assertRaises(CertificationError):
            autoia(model, [Interval(0.0, 1.0)], order=6, branch_policy=policy)

    def test_log_remains_strict_at_zero(self):
        def model(p):
            return np.log(p[0])

        with self.assertRaises(CertificationError):
            autoia(model, [Interval(0.0, 1.0)], order=6)

    def test_positive_noninteger_power_accepts_zero_boundary(self):
        for exponent in (0.25, 0.5, 1.25):
            with self.subTest(exponent=exponent):
                def model(p, exponent=exponent):
                    return p[0] ** exponent
                R = autoia(model, [Interval(0.0, 4.0)], order=6)
                for x in np.linspace(0.0, 4.0, 301):
                    y = x ** exponent
                    self.assertLessEqual(R.lower[0], y)
                    self.assertGreaterEqual(R.upper[0], y)

    def test_negative_noninteger_power_rejects_zero_boundary(self):
        def model(p):
            return p[0] ** -0.25
        with self.assertRaises(CertificationError):
            autoia(model, [Interval(0.0, 4.0)], order=6)

    def test_invalid_sqrt_box_still_fails_closed(self):
        def model(p):
            return np.sqrt(p[0])
        with self.assertRaises(CertificationError):
            autoia(model, [Interval(-0.1, 1.0)], order=6)

    def test_generator_representation_uses_same_domain_architecture(self):
        def model(p):
            return np.sqrt(p[0])
        R = autoia(model, [Interval(0.0, 4.0)], order=6, representation="generator")
        self.assertLessEqual(R.lower[0], 0.0)
        self.assertGreaterEqual(R.upper[0], 2.0)
        events = [e for e in R.range_policy_events if e.get("kind") == "domain-proof"]
        self.assertEqual(events[0]["methods_tried"], ("natural-physical-box",))

    def test_prepared_manybox_reuses_domain_architecture(self):
        def model(p):
            return np.sqrt(p[0])

        session = prepare(model, order=6)
        results = session.certify_many([
            [Interval(0.0, 4.0)],
            [Interval(0.0, 1.0)],
            [Interval(1.0, 9.0)],
        ])
        exact = [2.0, 1.0, 3.0]
        for R, hi in zip(results, exact):
            self.assertLessEqual(R.lower[0], 0.0 if hi < 3.0 else 1.0)
            self.assertGreaterEqual(R.upper[0], hi)


if __name__ == "__main__":
    unittest.main()
