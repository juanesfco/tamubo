import ast
from pathlib import Path
import unittest

import autoia.polynomial.algebra as algebra
import autoia.polynomial.ranging as ranging


class TestPolynomialBoundary(unittest.TestCase):
    def test_polynomial_modules_do_not_import_higher_layers(self):
        forbidden = {
            "autoia._reference_v11",
            "autoia._reference_v11_oracle",
            "autoia.representations",
            "autoia.api",
            "autoia.policies",
            "autoia.results",
            "autoia.controller",
            "autoia.semantic",
            "autoia.enclosures",
        }
        for module in (algebra, ranging):
            path = Path(module.__file__)
            tree = ast.parse(path.read_text())
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
            for name in imports:
                self.assertFalse(
                    any(name == f or name.startswith(f + ".") for f in forbidden),
                    f"{path.name} imports forbidden higher layer {name}",
                )

    def test_ranging_depends_on_algebra_and_trusted_not_migration_core(self):
        source = Path(ranging.__file__).read_text()
        self.assertIn("from .algebra import", source)
        self.assertIn("from ..trusted import", source)
        self.assertNotIn("_reference_v11", source)


if __name__ == "__main__":
    unittest.main()
