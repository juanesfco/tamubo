import ast
from pathlib import Path
import unittest

import autoia.controller as controller
import autoia.api as api


class TestControllerBoundary(unittest.TestCase):
    def test_controller_modules_do_not_import_api_or_migration_artifacts(self):
        root = Path(controller.__file__).resolve().parent
        forbidden = (
            "autoia.api", "autoia._reference_v11", "autoia._reference_v11_oracle"
        )
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(any(alias.name.startswith(p) for p in forbidden))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level == 0:
                        self.assertFalse(any(module.startswith(p) for p in forbidden))

    def test_public_api_imports_controller_not_migration_core(self):
        source = Path(api.__file__).read_text()
        self.assertIn("from .controller import", source)
        self.assertNotIn("_reference_v11", source)

    def test_runtime_no_longer_contains_process_global_policy_stacks(self):
        import autoia.enclosures.runtime as runtime
        source = Path(runtime.__file__).read_text()
        self.assertNotIn("_ACTIVE_RANGE_POLICY_STACK", source)
        self.assertNotIn("_ACTIVE_RANGE_EVENT_STACK", source)
        self.assertIn("ContextVar", source)


if __name__ == "__main__":
    unittest.main()
