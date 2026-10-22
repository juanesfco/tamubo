"""End-to-end validated elementary-function execution tests."""

from __future__ import annotations

import math
import unittest
import numpy as np

from autoia import autoia, Interval, CertificationError

from numpy import sin as direct_sin_alias
from math import cos as direct_math_cos_alias
from math import log10 as direct_math_log10_alias


class TestElementaryExecution(unittest.TestCase):
    def _assert_grid_contained(self, name, box, order=6, representation="polynomial"):
        def model(p):
            return getattr(np, name)(p[0])

        R = autoia(model, [box], order=order, representation=representation)
        fn = getattr(math, name)
        for x in np.linspace(box.lo, box.hi, 1001):
            y = fn(float(x))
            self.assertLessEqual(R.lower[0], y)
            self.assertGreaterEqual(R.upper[0], y)
        return R

    def test_all_five_polynomial_functions_contain_dense_grid(self):
        cases = {
            "sqrt": Interval(0.25, 4.0),
            "exp": Interval(-0.5, 0.5),
            "log": Interval(0.5, 2.0),
            "log10": Interval(0.5, 2.0),
            "sin": Interval(-0.5, 0.5),
            "cos": Interval(-0.5, 0.5),
        }
        results = {
            name: self._assert_grid_contained(name, box, order=8)
            for name, box in cases.items()
        }
        # Broad but meaningful quality guards: these catch accidental fallback
        # to enormous remainders without pretending to specify optimal widths.
        self.assertLess(results["sqrt"].width[0], 1.7)
        self.assertLess(results["log"].width[0], 1.5)
        self.assertLess(results["exp"].width[0], 1.2)
        self.assertLess(results["sin"].width[0], 1.1)
        self.assertLess(results["cos"].width[0], 0.2)

    def test_interval_representation_supports_all_five(self):
        for name, box in [
            ("sqrt", Interval(0.25, 4.0)),
            ("exp", Interval(-0.5, 0.5)),
            ("log", Interval(0.5, 2.0)),
            ("sin", Interval(-0.5, 0.5)),
            ("cos", Interval(-0.5, 0.5)),
        ]:
            with self.subTest(function=name):
                self._assert_grid_contained(name, box, order=4, representation="interval")

    def test_sqrt_touching_zero_is_sound_in_interval_representation(self):
        def model(p):
            return np.sqrt(p[0])
        R = autoia(
            model, [Interval(0.0, 4.0)], order=6, representation="interval"
        )
        self.assertLessEqual(R.lower[0], 0.0)
        self.assertGreaterEqual(R.upper[0], 2.0)
        self.assertLess(R.width[0], 2.0000000001)

    def test_polynomial_boundary_domain_is_proved_from_physical_box(self):
        # The retained centered polynomial range may extend a few ulps below
        # zero, but the independent natural interval over the actual physical
        # input box is exactly [0,4].  Step 10a uses that independent enclosure
        # as a domain proof rather than clipping the polynomial range.
        def model(p):
            return np.sqrt(p[0])
        R = autoia(model, [Interval(0.0, 4.0)], order=6)
        self.assertLessEqual(R.lower[0], 0.0)
        self.assertGreaterEqual(R.upper[0], 2.0)
        events = [e for e in R.range_policy_events if e.get("kind") == "domain-proof"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["methods_tried"], ("natural-physical-box",))
        self.assertEqual(events[0]["status"], "proved")
        self.assertEqual(events[0]["final_range"], Interval(0.0, 4.0))

    def test_composed_exp_log_contains_identity(self):
        def model(p):
            return np.exp(np.log(p[0]))
        box = Interval(0.8, 1.2)
        R = autoia(model, [box], order=6)
        for x in np.linspace(box.lo, box.hi, 501):
            self.assertLessEqual(R.lower[0], x)
            self.assertGreaterEqual(R.upper[0], x)

    def test_direct_numpy_alias_is_dispatched_locally(self):
        def model(p):
            return direct_sin_alias(p[0])
        R = autoia(model, [Interval(-0.2, 0.2)], order=6)
        self.assertLessEqual(R.lower[0], math.sin(-0.2))
        self.assertGreaterEqual(R.upper[0], math.sin(0.2))

    def test_elementary_dispatch_does_not_mutate_numpy_or_math(self):
        np_originals = {name: getattr(np, name) for name in ("sqrt", "exp", "log", "log10", "sin", "cos")}
        math_originals = {name: getattr(math, name) for name in ("sqrt", "exp", "log", "log10", "sin", "cos")}

        def model(p):
            return np.sin(p[0]) + math.cos(p[0])

        autoia(model, [Interval(-0.1, 0.1)], order=4)
        for name, fn in np_originals.items():
            self.assertIs(getattr(np, name), fn)
        for name, fn in math_originals.items():
            self.assertIs(getattr(math, name), fn)

    def test_math_module_and_direct_alias_dispatch(self):
        def model_module(p):
            return math.cos(p[0]) + math.exp(p[0])
        def model_alias(p):
            return direct_math_cos_alias(p[0])
        R1 = autoia(model_module, [Interval(-0.2, 0.2)], order=6)
        R2 = autoia(model_alias, [Interval(-0.2, 0.2)], order=6)
        for x in np.linspace(-0.2, 0.2, 101):
            y1 = math.cos(x) + math.exp(x)
            y2 = math.cos(x)
            self.assertLessEqual(R1.lower[0], y1)
            self.assertGreaterEqual(R1.upper[0], y1)
            self.assertLessEqual(R2.lower[0], y2)
            self.assertGreaterEqual(R2.upper[0], y2)

    def test_log10_module_numpy_and_direct_alias_dispatch(self):
        def model_np(p):
            return np.log10(p[0])
        def model_math(p):
            return math.log10(p[0])
        def model_alias(p):
            return direct_math_log10_alias(p[0])
        box = Interval(0.5, 2.0)
        for model in (model_np, model_math, model_alias):
            R = autoia(model, [box], order=6)
            for x in np.linspace(box.lo, box.hi, 201):
                y = math.log10(float(x))
                self.assertLessEqual(R.lower[0], y)
                self.assertGreaterEqual(R.upper[0], y)

    def test_log10_domain_failure_is_fail_closed(self):
        def model(p):
            return math.log10(p[0])
        with self.assertRaises(CertificationError):
            autoia(model, [Interval(-0.1, 1.0)], order=5)

    def test_object_array_ufunc_dispatch(self):
        def model(p):
            x = np.array([p[0], p[0] + 0.1], dtype=object)
            return np.sin(x)
        R = autoia(model, [Interval(-0.2, 0.2)], order=5)
        self.assertEqual(len(R.lower), 2)
        self.assertLessEqual(R.lower[0], math.sin(-0.2))
        self.assertGreaterEqual(R.upper[1], math.sin(0.3))

    def test_elementary_postprocessing_after_semantic_solve(self):
        def spring_post(p):
            k1, k2, F = p
            A = np.array([[k1 + k2, -k2], [-k2, k2]], dtype=object)
            b = np.array([F, 0.0], dtype=object)
            u = np.linalg.solve(A, b)
            return np.array([np.exp(u[0]), np.sin(u[1]), np.sqrt(1.0 + u[0])], dtype=object)

        boxes = [Interval(90, 110), Interval(900, 1100), Interval(9.5, 10.5)]
        R = autoia(spring_post, boxes, order=5)
        self.assertEqual(len(R.certificates), 1)
        for k1 in np.linspace(90.0, 110.0, 21):
            for F in np.linspace(9.5, 10.5, 23):
                u = F / k1
                vals = [math.exp(u), math.sin(u), math.sqrt(1.0 + u)]
                for j, val in enumerate(vals):
                    self.assertLessEqual(R.lower[j], val)
                    self.assertGreaterEqual(R.upper[j], val)

    def test_multivariate_nonlinear_arguments_remain_contained(self):
        def model(p):
            z = 0.4 + 0.15*p[0] - 0.10*p[1] + 0.08*p[0]*p[1]
            return np.array([
                np.exp(z),
                np.sin(z),
                np.cos(z),
                np.log(1.5 + z),
                np.sqrt(2.0 + z),
            ], dtype=object)

        boxes = [Interval(-0.5, 0.5), Interval(-0.4, 0.6)]
        R = autoia(model, boxes, order=6)
        for x in np.linspace(boxes[0].lo, boxes[0].hi, 41):
            for y in np.linspace(boxes[1].lo, boxes[1].hi, 43):
                z = 0.4 + 0.15*x - 0.10*y + 0.08*x*y
                exact = [math.exp(z), math.sin(z), math.cos(z), math.log(1.5+z), math.sqrt(2.0+z)]
                for j, val in enumerate(exact):
                    self.assertLessEqual(R.lower[j], val)
                    self.assertGreaterEqual(R.upper[j], val)

    def test_real_domain_failures_are_certification_failures(self):
        def bad_log(p):
            return np.log(p[0])
        def bad_sqrt(p):
            return np.sqrt(p[0])
        with self.assertRaises(CertificationError):
            autoia(bad_log, [Interval(-0.1, 1.0)])
        with self.assertRaises(CertificationError):
            autoia(bad_sqrt, [Interval(-0.1, 1.0)])


if __name__ == "__main__":
    unittest.main()
