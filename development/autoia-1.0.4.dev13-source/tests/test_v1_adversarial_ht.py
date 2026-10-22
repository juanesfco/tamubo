import math
import unittest

import numpy as np

from autoia import Interval, autoia


def _endpoint_branch_model(p):
    x = p[0]
    if x > 1.0:
        raise ValueError("x must not exceed 1")
    if x < 1.0:
        return x
    elif x == 1.0:
        return 10.0
    raise AssertionError("unreachable")


def _effectiveness_shell_and_tube_two_shells(p):
    # Public ht.effectiveness_from_NTU S&T algebra, with n_shell_tube=2.
    NTU, Cr = p
    if Cr > 1:
        raise ValueError("Heat capacity rate must be less than 1 by definition.")
    shells = 2
    NTU = NTU/shells
    x0 = math.sqrt(1. + Cr*Cr)
    x1 = math.exp(-NTU*x0)
    top = 1. + x1
    bottom = 1. - x1
    effectiveness = 2./(1. + Cr + x0*top/bottom)
    term = ((1. - effectiveness*Cr)/(1. - effectiveness))**shells
    effectiveness = (term - 1.)/(term - Cr)
    return effectiveness


class TestV1AdversarialHT(unittest.TestCase):
    def test_strict_closed_endpoint_prunes_impossible_exception_path(self):
        result = autoia(_endpoint_branch_model, [Interval(0.95, 1.0)])
        self.assertTrue(result.machine_rigorous)
        self.assertLessEqual(float(result.lower[0]), 0.95)
        self.assertGreaterEqual(float(result.upper[0]), 10.0)
        selected = getattr(result, "selected_result", result)
        self.assertEqual(len(selected.branch_results), 2)
        self.assertTrue(any(
            e.get("action") == "partition-exact-affine-predicate"
            and e.get("root") == 1.0
            for e in selected.path_events
        ))

    def test_reciprocal_proof_failure_triggers_physical_subdivision(self):
        box = [Interval(1.0, 3.0), Interval(0.3, 0.8)]
        result = autoia(_effectiveness_shell_and_tube_two_shells, box)
        self.assertTrue(result.machine_rigorous)
        selected = getattr(result, "selected_result", result)
        self.assertTrue(any(
            e.get("action") == "split-unresolved-domain"
            and e.get("operation") == "reciprocal"
            for e in selected.path_events
        ))

        # Sampling is diagnostic, not part of the proof; it guards against a
        # regression in this frozen external engineering acceptance case.
        for NTU in np.linspace(1.0, 3.0, 13):
            for Cr in np.linspace(0.3, 0.8, 11):
                y = _effectiveness_shell_and_tube_two_shells([NTU, Cr])
                self.assertLessEqual(float(result.lower[0]), y)
                self.assertGreaterEqual(float(result.upper[0]), y)


if __name__ == "__main__":
    unittest.main()
