import inspect
import unittest

import autoia.enclosures.interval as interval_mod
import autoia.enclosures.piecewise as piecewise_mod
import autoia.enclosures.polynomial as polynomial_mod
import autoia.enclosures.runtime as runtime_mod
import autoia.enclosures.value as value_mod
import autoia.representations as public_reps


class TestEnclosureBoundary(unittest.TestCase):
    def test_enclosure_modules_do_not_import_migration_or_api_layers(self):
        forbidden = (
            "_reference_v11",
            "from ..api",
            "from autoia.api",
            "from ..results",
            "from ..policies",
        )
        for module in (
            interval_mod, piecewise_mod, polynomial_mod, runtime_mod, value_mod
        ):
            source = inspect.getsource(module)
            for marker in forbidden:
                self.assertNotIn(marker, source, (module.__name__, marker))

    def test_public_representations_facade_uses_extracted_enclosures(self):
        source = inspect.getsource(public_reps)
        self.assertNotIn("_reference_v11", source)
        self.assertIn("from .enclosures import", source)

    def test_validated_value_has_no_concrete_polynomial_interval_dispatch(self):
        source = inspect.getsource(value_mod)
        self.assertNotIn("isinstance(rep, IntervalEnclosure)", source)
        self.assertNotIn("isinstance(rep, PolynomialEnclosure)", source)


if __name__ == "__main__":
    unittest.main()
