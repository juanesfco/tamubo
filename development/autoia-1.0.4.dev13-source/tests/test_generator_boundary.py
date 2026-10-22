from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestGeneratorBoundary(unittest.TestCase):
    def test_generator_layer_does_not_import_semantic_controller_or_api(self):
        text = (ROOT / "autoia" / "enclosures" / "generator.py").read_text()
        self.assertNotIn("autoia.semantic", text)
        self.assertNotIn("autoia.controller", text)
        self.assertNotIn("autoia.api", text)
        self.assertNotIn("..semantic", text)
        self.assertNotIn("..controller", text)

    def test_generator_representation_is_public(self):
        from autoia import SharedGeneratorPolynomialEnclosure
        from autoia.representations import SharedGeneratorPolynomialEnclosure as R
        self.assertIs(SharedGeneratorPolynomialEnclosure, R)


if __name__ == "__main__":
    unittest.main()
