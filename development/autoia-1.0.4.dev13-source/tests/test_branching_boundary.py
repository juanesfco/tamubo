import threading
import unittest
from pathlib import Path

from autoia.enclosures.branching import (
    BranchRuntime, BranchExecutionContext, active_branch_runtime,
)

ROOT = Path(__file__).resolve().parents[1]


class TestBranchingBoundary(unittest.TestCase):
    def test_enclosure_branching_layer_does_not_import_controller(self):
        text = (ROOT / "autoia" / "enclosures" / "branching.py").read_text()
        self.assertNotIn("autoia.controller", text)
        self.assertNotIn("..controller", text)

    def test_branch_runtime_is_context_local_across_threads(self):
        barrier = threading.Barrier(2)
        observed = {}

        def worker(name, decision):
            runtime = BranchRuntime(forced_decisions={0: decision})
            with BranchExecutionContext(runtime):
                barrier.wait()
                observed[name] = active_branch_runtime().forced_decisions[0]
                barrier.wait()

        a = threading.Thread(target=worker, args=("a", True))
        b = threading.Thread(target=worker, args=("b", False))
        a.start(); b.start(); a.join(); b.join()
        self.assertEqual(observed, {"a": True, "b": False})

    def test_control_signals_are_not_ordinary_exceptions(self):
        from autoia.enclosures.branching import BranchDecisionRequired, InfeasiblePath
        self.assertFalse(issubclass(BranchDecisionRequired, Exception))
        self.assertFalse(issubclass(InfeasiblePath, Exception))
        self.assertTrue(issubclass(BranchDecisionRequired, BaseException))
        self.assertTrue(issubclass(InfeasiblePath, BaseException))


if __name__ == "__main__":
    unittest.main()
