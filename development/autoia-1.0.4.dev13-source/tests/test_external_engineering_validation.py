"""Regression tests for the pre-existing external engineering corpus."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from autoia import Interval, autoia
from validation.external_engineering.cases import CASES, CASE_BY_NAME
from validation.external_engineering.external_sources import fluids_extract, ht_extract, grheat_extract


class TestExternalEngineeringValidation(unittest.TestCase):
    def test_external_case_names_are_unique(self):
        names = [case.name for case in CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_vendored_external_source_modules_do_not_import_autoia(self):
        for module in (fluids_extract, ht_extract, grheat_extract):
            source = inspect.getsource(module)
            self.assertNotIn("import autoia", source)
            self.assertNotIn("from autoia", source)

    def test_every_external_case_runs_numerically(self):
        for case in CASES:
            midpoint = np.asarray([(lo + hi)*0.5 for lo, hi in case.bounds], dtype=float)
            with self.subTest(case=case.name):
                y = np.asarray(case.model(midpoint), dtype=float)
                self.assertGreater(y.size, 0)
                self.assertTrue(np.all(np.isfinite(y)))

    def test_representative_external_cases_certify(self):
        for name in (
            "fluids_morsi_alexander_branch",
            "fluids_swamee_jain_log10",
            "ht_cylindrical_heat_transfer",
            "grheat_plane_instantaneous",
        ):
            case = CASE_BY_NAME[name]
            with self.subTest(case=name):
                result = autoia(
                    case.model,
                    [Interval(lo, hi) for lo, hi in case.bounds],
                    order=case.order,
                )
                self.assertTrue(result.machine_rigorous)
                self.assertTrue(np.all(np.isfinite(result.lower)))
                self.assertTrue(np.all(np.isfinite(result.upper)))

    def test_external_piecewise_drag_really_crosses_branch_boundary(self):
        case = CASE_BY_NAME["fluids_morsi_alexander_branch"]
        self.assertLess(case.bounds[0][0], 1000.0)
        self.assertGreater(case.bounds[0][1], 1000.0)
        result = autoia(case.model, [Interval(*case.bounds[0])], order=case.order)
        self.assertGreaterEqual(len(result.branch_results), 2)


if __name__ == "__main__":
    unittest.main()
