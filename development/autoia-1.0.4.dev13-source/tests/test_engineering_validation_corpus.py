import inspect
import unittest
import warnings

import numpy as np

from autoia import Interval, autoia
from validation.engineering_corpus.cases import CASES, CASE_BY_NAME


class TestEngineeringValidationCorpus(unittest.TestCase):
    def test_case_names_are_unique(self):
        names = [case.name for case in CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_model_sources_are_autoia_independent(self):
        seen = set()
        for case in CASES:
            module = inspect.getmodule(case.model)
            if module in seen:
                continue
            seen.add(module)
            source = inspect.getsource(module)
            self.assertNotIn("import autoia", source)
            self.assertNotIn("from autoia", source)

    def test_every_model_runs_numerically_at_its_midpoint(self):
        for case in CASES:
            midpoint = np.asarray([(lo + hi) * 0.5 for lo, hi in case.bounds], dtype=float)
            with self.subTest(case=case.name):
                y = np.asarray(case.model(midpoint))
                self.assertGreater(y.size, 0)
                self.assertTrue(np.all(np.isfinite(y)))

    def test_lmtd_invalid_witness_is_an_actual_source_level_singularity(self):
        case = CASE_BY_NAME["heat_exchanger_lmtd_singular_box"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                y = np.asarray(case.model(np.asarray(case.invalid_witness, dtype=float)), dtype=float)
        self.assertTrue(np.any(~np.isfinite(y)))

    def test_generator_pressure_vessel_representative_case_certifies(self):
        case = CASE_BY_NAME["pressure_vessel_generator"]
        result = autoia(
            case.model,
            [Interval(lo, hi) for lo, hi in case.bounds],
            order=case.order,
            representation=case.representation,
        )
        self.assertTrue(result.machine_rigorous)
        self.assertEqual(result.lower.shape, (3,))
        self.assertTrue(all(name == "generator-polynomial" for name in result.representation_names))


if __name__ == "__main__":
    unittest.main()
