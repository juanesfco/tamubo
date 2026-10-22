import pathlib
import unittest


class ApplicationPackageBoundaryTests(unittest.TestCase):
    def test_autoia_has_no_optimization_application_package(self):
        root = pathlib.Path(__file__).parents[1] / "autoia"
        self.assertFalse((root / "applications").exists())
        forbidden = ("expected improvement", "acquisition function", "branch-and-bound optimizer")
        for path in root.rglob("*.py"):
            text = path.read_text().lower()
            for marker in forbidden:
                self.assertNotIn(marker, text, f"optimization semantic {marker!r} leaked into {path}")

    def test_gp_integration_is_optional_and_dependency_free(self):
        text = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text().lower()
        self.assertIn("autoia.integrations", text)
        # Optional live integrations must not make Torch/GP libraries install requirements.
        dependency_block = text.split("dependencies =", 1)[1].split("\n", 1)[0]
        self.assertNotIn("botorch", dependency_block)
        self.assertNotIn("gpytorch", dependency_block)
        self.assertNotIn("torch", dependency_block)


if __name__ == "__main__":
    unittest.main()
