import ast
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestScipyCompatibilityBoundary(unittest.TestCase):
    def test_scipy_dispatch_has_no_hard_scipy_import(self):
        path = ROOT / "autoia" / "semantic" / "scipy_dispatch.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(
                    any(alias.name == "scipy" or alias.name.startswith("scipy.") for alias in node.names)
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                self.assertFalse(module == "scipy" or module.startswith("scipy."))

    def test_scipy_is_not_an_autoia_install_dependency(self):
        text = (ROOT / "pyproject.toml").read_text().lower()
        dependency_line = next(line for line in text.splitlines() if line.startswith("dependencies"))
        self.assertNotIn("scipy", dependency_line)

    def test_top_level_scipy_only_import_is_supported_lazily(self):
        code = r"""
import sys
import scipy
assert 'scipy.stats' not in sys.modules
assert 'scipy.special' not in sys.modules
import numpy as np
from autoia import Interval, autoia
def model(p):
    x = p[0]
    return np.array([scipy.stats.norm.cdf(x), scipy.special.ndtr(x)])
r = autoia(model, [Interval(-0.5, 0.5)], order=6)
assert r.machine_rigorous
assert r.lower[0] <= 0.3085375387 <= r.upper[0]
assert r.lower[1] <= 0.3085375387 <= r.upper[1]
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
