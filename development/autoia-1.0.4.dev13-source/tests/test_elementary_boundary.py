"""Architecture-boundary tests for the elementary-function layer."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import autoia.elementary as elementary


class TestElementaryBoundary(unittest.TestCase):
    def test_elementary_core_depends_only_on_trusted_layer(self):
        root = Path(elementary.__file__).resolve().parent
        forbidden = (
            "autoia.api",
            "autoia.controller",
            "autoia.semantic",
            "autoia.enclosures",
            "autoia.polynomial",
        )
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level == 0:
                        self.assertFalse(any(module.startswith(x) for x in forbidden))
                    elif node.level > 0:
                        top = module.split(".", 1)[0]
                        self.assertIn(top, {"trusted", "taylor"})
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(any(alias.name.startswith(x) for x in forbidden))

    def test_enclosure_elementary_does_not_depend_on_semantic_or_controller(self):
        import autoia.enclosures.elementary as ee
        path = Path(ee.__file__).resolve()
        text = path.read_text()
        self.assertNotIn("autoia.controller", text)
        self.assertNotIn("autoia.semantic", text)
        self.assertNotIn("autoia.api", text)


if __name__ == "__main__":
    unittest.main()
