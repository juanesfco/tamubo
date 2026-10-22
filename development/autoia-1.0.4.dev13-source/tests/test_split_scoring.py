import unittest
from unittest.mock import patch
import numpy as np

import autoia
from autoia import Interval, RangePolicy, SubdivisionPolicy
from autoia.controller.execution import autoia_single
import autoia.controller.subdivision as subdivision
from autoia.controller.split_scoring import score_split_direction, cheap_score_key


def hidden_monotone_model(p):
    z = 0.0
    for x in p:
        z = z + x + x*x + (x**3)/3.0
    return z


def stationary_model(p):
    x = p[0]
    return x*x + 0.5*(x**3)


class TestCheapSplitScore(unittest.TestCase):
    def test_default_subdivision_policy_uses_cheap_score(self):
        self.assertEqual(SubdivisionPolicy().split_method, "cheap")

    def test_invalid_split_method_is_typed_public_failure(self):
        with self.assertRaises(autoia.InvalidInputError):
            SubdivisionPolicy(split_method="mystery")

    def test_hidden_monotone_score_predicts_resolution(self):
        box = [Interval(-0.6, 0.6) for _ in range(5)]
        rp = RangePolicy(bernstein_max_coefficients=256)
        root = autoia_single(hidden_monotone_model, box, order=3, range_policy=rp)
        M = subdivision._branch_controller_metrics(root)
        self.assertTrue(M["needs_subdivision"])
        ambiguity = subdivision._direction_ambiguity_scores(root, box)
        scores = [
            score_split_direction(root, box, i, float(ambiguity[i]))
            for i in M["unresolved_variables"]
        ]
        best = max(scores, key=cheap_score_key)
        self.assertGreater(best["predicted_unresolved_gain"], 0)
        self.assertGreater(best["predicted_structural_gain"], 0.0)

    def test_stationary_score_predicts_local_progress_and_width_gain(self):
        box = [Interval(-1.0, 1.0)]
        rp = RangePolicy(bernstein_max_coefficients=2)
        root = autoia_single(stationary_model, box, order=3, range_policy=rp)
        score = score_split_direction(root, box, 0, 1.0)
        # The union still contains one unresolved descendant, but recentering
        # resolves one child and predicts the known first-step width improvement.
        self.assertEqual(score["predicted_unresolved_gain"], 0)
        self.assertGreater(score["predicted_local_resolution_gain"], 0)
        self.assertGreater(score["predicted_relative_width_gain"], 0.0)

    def test_cheap_controller_matches_reference_on_stationary_study(self):
        box = [Interval(-1.0, 1.0)]
        rp = RangePolicy(bernstein_max_coefficients=2)
        kwargs = dict(
            max_branches=8,
            max_depth=8,
            max_probe_directions=1,
            min_structural_gain=0.05,
            min_relative_width_gain=1.0e-6,
        )
        cheap = autoia.autoia_adaptive_subdivide(
            stationary_model, box, order=3, range_policy=rp,
            subdivision_policy=SubdivisionPolicy(**kwargs, split_method="cheap"),
        )
        ref = autoia.autoia_adaptive_subdivide(
            stationary_model, box, order=3, range_policy=rp,
            subdivision_policy=SubdivisionPolicy(**kwargs, split_method="lookahead"),
        )
        np.testing.assert_array_equal(cheap.lower, ref.lower)
        np.testing.assert_array_equal(cheap.upper, ref.upper)
        self.assertEqual(len(cheap.branch_results), len(ref.branch_results))

    def test_cheap_controller_certifies_only_selected_candidate(self):
        box = [Interval(-0.6, 0.6) for _ in range(5)]
        rp = RangePolicy(bernstein_max_coefficients=256)
        kwargs = dict(
            max_branches=4,
            max_depth=6,
            max_probe_directions=2,
            min_structural_gain=0.01,
            min_relative_width_gain=1.0e-4,
        )
        original = subdivision._evaluate_subdivision_candidate

        cheap_calls = 0
        def cheap_wrapped(*args, **kw):
            nonlocal cheap_calls
            cheap_calls += 1
            return original(*args, **kw)
        with patch.object(subdivision, "_evaluate_subdivision_candidate", cheap_wrapped):
            cheap = autoia.autoia_adaptive_subdivide(
                hidden_monotone_model, box, order=3, range_policy=rp,
                subdivision_policy=SubdivisionPolicy(**kwargs, split_method="cheap"),
            )

        ref_calls = 0
        def ref_wrapped(*args, **kw):
            nonlocal ref_calls
            ref_calls += 1
            return original(*args, **kw)
        with patch.object(subdivision, "_evaluate_subdivision_candidate", ref_wrapped):
            ref = autoia.autoia_adaptive_subdivide(
                hidden_monotone_model, box, order=3, range_policy=rp,
                subdivision_policy=SubdivisionPolicy(**kwargs, split_method="lookahead"),
            )

        self.assertEqual(len(cheap.branch_results), 4)
        self.assertEqual(len(ref.branch_results), 4)
        self.assertEqual(cheap_calls, 3)  # one selected direction per split
        self.assertEqual(ref_calls, 6)    # two fully certified probes per split
        self.assertLess(cheap_calls, ref_calls)
        self.assertLessEqual(cheap.lower[0], -1.56)
        self.assertGreaterEqual(cheap.upper[0], 5.16)

    def test_cheap_controller_supports_generator_representation(self):
        box = [Interval(-0.6, 0.6) for _ in range(5)]
        rp = RangePolicy(bernstein_max_coefficients=256)
        r = autoia.autoia_adaptive_subdivide(
            hidden_monotone_model, box, order=3, representation="generator",
            range_policy=rp,
            subdivision_policy=SubdivisionPolicy(
                max_branches=4, max_depth=6, max_probe_directions=2,
                split_method="cheap",
            ),
        )
        self.assertEqual(len(r.branch_results), 4)
        self.assertLessEqual(r.lower[0], -1.56)
        self.assertGreaterEqual(r.upper[0], 5.16)



if __name__ == "__main__":
    unittest.main()
