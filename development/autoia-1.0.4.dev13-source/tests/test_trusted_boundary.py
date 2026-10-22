"""Architectural tests for the extracted trusted-kernel dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import autoia
import autoia.trusted as trusted


class TestTrustedBoundary(unittest.TestCase):
    def test_trusted_modules_do_not_import_higher_autoia_layers(self):
        root = Path(trusted.__file__).resolve().parent
        allowed_relative = {"environment", "interval", "binary64", "reductions", "elementary", "array_ops"}
        forbidden_prefixes = {
            "autoia.api",
            "autoia.representations",
            "autoia.policies",
            "autoia.results",
            "autoia.controller",
            "autoia.semantic",
            "autoia.enclosures",
            "autoia.polynomial",
            "autoia._reference_v11",
        }

        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            any(alias.name.startswith(p) for p in forbidden_prefixes),
                            f"trusted module {path.name} imports higher layer {alias.name}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level > 0:
                        top = module.split(".", 1)[0]
                        self.assertIn(
                            top,
                            allowed_relative,
                            f"trusted module {path.name} imports .{module}",
                        )
                    else:
                        self.assertFalse(
                            any(module.startswith(p) for p in forbidden_prefixes),
                            f"trusted module {path.name} imports higher layer {module}",
                        )

    def test_proof_critical_nextafter_is_confined_to_trusted_layer(self):
        package_root = Path(autoia.__file__).resolve().parent
        offenders = []
        for path in package_root.rglob("*.py"):
            if "trusted" in path.parts:
                continue
            if "np.nextafter(" in path.read_text():
                offenders.append(path.relative_to(package_root).as_posix())
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
