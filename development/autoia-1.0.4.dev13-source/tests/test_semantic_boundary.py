import inspect
import unittest

import autoia.semantic.linear_solve as ls


class TestSemanticBoundary(unittest.TestCase):
    def test_semantic_solve_module_does_not_import_migration_api_or_controller(self):
        source = inspect.getsource(ls)
        forbidden = (
            "_reference_v11",
            "from ..api",
            "from autoia.api",
            "from ..policies",
            "from autoia.policies",
            "adaptive_final_range",
            "SubdivisionPolicy",
        )
        for marker in forbidden:
            self.assertNotIn(marker, source)

    def test_semantic_module_owns_solve_contexts(self):
        self.assertEqual(ls.GenericSolveContext.__module__, "autoia.semantic.linear_solve")
        self.assertEqual(ls.SolveContext.__module__, "autoia.semantic.linear_solve")


if __name__ == "__main__":
    unittest.main()
