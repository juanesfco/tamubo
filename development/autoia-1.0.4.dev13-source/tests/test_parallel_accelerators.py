import unittest
import numpy as np

from autoia import (
    Interval, prepare, enclose_many, available_accelerators,
    proposal_map, resolve_proposal_device, UnsupportedOperationError,
)


def module_model(p):
    return p[0]*p[0] + 0.5*p[1]


class TestParallelAndAccelerators(unittest.TestCase):
    def test_workers_auto_and_parallel_auto_preserve_results(self):
        boxes = [
            [Interval(-0.2, -0.1), Interval(-0.1, 0.1)],
            [Interval(-0.1, 0.0), Interval(-0.1, 0.1)],
            [Interval(0.0, 0.1), Interval(-0.1, 0.1)],
            [Interval(0.1, 0.2), Interval(-0.1, 0.1)],
        ]
        session = prepare(module_model, order=3, cache_max_entries=0)
        serial = session.enclose_many(boxes, workers=1, parallel="serial", use_cache=False)
        auto = session.enclose_many(boxes, workers="auto", parallel="auto", use_cache=False)
        self.assertEqual(len(serial), len(auto))
        for a, b in zip(serial, auto):
            np.testing.assert_array_equal(a.lower, b.lower)
            np.testing.assert_array_equal(a.upper, b.upper)


    def test_safe_auto_small_batch_stays_serial(self):
        boxes = [
            [Interval(-0.2, -0.1), Interval(0.0, 0.1)],
            [Interval(-0.1, 0.0), Interval(0.0, 0.1)],
            [Interval(0.0, 0.1), Interval(0.0, 0.1)],
            [Interval(0.1, 0.2), Interval(0.0, 0.1)],
        ]
        session = prepare(module_model, order=3, cache_max_entries=0)
        results, info = session.enclose_many(
            boxes, workers="auto", parallel="auto", return_info=True, use_cache=False
        )
        self.assertEqual(len(results), 4)
        self.assertEqual(info.resolved_workers, 1)
        self.assertEqual(info.resolved_parallel, "serial")
        self.assertTrue(str(info.scheduling_note).startswith("auto-small-batch"))
        self.assertTrue(info.machine_rigorous)

    def test_auto_proposal_device_is_conservative_cpu(self):
        self.assertEqual(resolve_proposal_device("auto"), "cpu")

    def test_top_level_enclose_many(self):
        boxes = [[Interval(-0.1, 0.0), Interval(0.0, 0.1)],
                 [Interval(0.0, 0.1), Interval(0.0, 0.1)]]
        out = enclose_many(module_model, boxes, workers=1, parallel="serial", order=3)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r.machine_rigorous for r in out))

    def test_accelerator_report_never_calls_gpu_trusted(self):
        info = available_accelerators()
        self.assertTrue(info["cpu"].available)
        self.assertTrue(info["cpu"].trusted_for_proof)
        self.assertFalse(info["mps"].trusted_for_proof)
        self.assertFalse(info["cuda"].trusted_for_proof)

    def test_cpu_torch_proposal_map_when_torch_available(self):
        try:
            import torch
        except Exception:
            self.skipTest("optional torch not installed")

        def f(x):
            return torch.sum(x*x, dim=1)

        result = proposal_map(f, [[1.0, 2.0], [3.0, 4.0]], device="cpu")
        np.testing.assert_allclose(result.values, [5.0, 25.0])
        self.assertFalse(result.machine_rigorous)
        self.assertEqual(result.device, "cpu")

    def test_explicit_unavailable_mps_fails_clearly(self):
        info = available_accelerators()
        if info["mps"].available:
            self.skipTest("MPS is available; platform-specific execution is tested on Apple hardware")
        try:
            import torch
        except Exception:
            with self.assertRaises(UnsupportedOperationError):
                proposal_map(lambda x: x, [[1.0]], device="mps")
            return
        with self.assertRaises(UnsupportedOperationError):
            proposal_map(lambda x: x, [[1.0]], device="mps")

    def test_manybox_execution_metadata_is_explicit_and_nonproof(self):
        boxes = [
            [Interval(-0.2, -0.1), Interval(0.0, 0.1)],
            [Interval(0.0, 0.1), Interval(0.0, 0.1)],
        ]
        session = prepare(module_model, order=3, cache_max_entries=0)
        results, info = session.enclose_many(
            boxes, workers=1, parallel="serial", return_info=True, use_cache=False
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(info.n_boxes, 2)
        self.assertEqual(info.resolved_workers, 1)
        self.assertEqual(info.resolved_parallel, "serial")
        self.assertTrue(info.machine_rigorous)
        self.assertEqual(info.proof_backend, "cpu-binary64")
        self.assertIs(session.last_manybox_execution, info)

    def test_top_level_manybox_return_info(self):
        boxes = [[Interval(-0.1, 0.0), Interval(0.0, 0.1)]]
        results, info = enclose_many(
            module_model, boxes, workers=1, parallel="serial", return_info=True, order=3
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(info.n_boxes, 1)
        self.assertTrue(info.machine_rigorous)

    def test_process_parallel_local_closure_with_cloudpickle_when_available(self):
        try:
            import cloudpickle  # noqa: F401
        except Exception:
            self.skipTest("optional cloudpickle not installed")

        shift = 0.125
        def local_model(p):
            return (p[0] + shift)*(p[0] + shift)

        boxes = [[Interval(-0.2, -0.1)], [Interval(0.0, 0.1)]]
        session = prepare(local_model, order=3, cache_max_entries=0)
        results, info = session.enclose_many(
            boxes, workers=2, parallel="process", return_info=True, use_cache=False
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(info.resolved_parallel, "process")
        self.assertTrue(info.used_cloudpickle)
        self.assertTrue(info.machine_rigorous)


if __name__ == "__main__":
    unittest.main()
