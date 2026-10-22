from __future__ import annotations

import ast
from pathlib import Path
import unittest

import autoia


class TestMigrationRetirement(unittest.TestCase):
    def setUp(self):
        self.package_root = Path(autoia.__file__).resolve().parent

    def test_migration_modules_are_absent_from_production_package(self):
        self.assertFalse((self.package_root / "_reference_v11.py").exists())
        self.assertFalse((self.package_root / "_reference_v11_oracle.py").exists())

    def test_production_modules_do_not_import_migration_artifacts(self):
        offenders = []
        for path in self.package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
                for name in names:
                    if "_reference_v11" in name:
                        offenders.append((path.relative_to(self.package_root).as_posix(), name))
        self.assertEqual(offenders, [])

    def test_historical_oracle_lives_only_in_test_support(self):
        oracle_path = Path(__file__).resolve().parent / "support" / "oracle_v11.py"
        self.assertTrue(oracle_path.exists())
        self.assertGreater(oracle_path.stat().st_size, 100_000)


if __name__ == "__main__":
    unittest.main()
