import threading
import unittest

from autoia.controller import RangePolicy
from autoia.enclosures.runtime import (
    RangePolicyContext,
    active_range_policy,
    record_range_policy_event,
)


class TestControllerContext(unittest.TestCase):
    def test_nested_context_restores_parent_exactly(self):
        outer = RangePolicy(proof_bernstein_max_coefficients=17)
        inner = RangePolicy(proof_bernstein_max_coefficients=93)
        with RangePolicyContext(outer) as outer_ctx:
            self.assertIs(active_range_policy(), outer)
            record_range_policy_event({"where": "outer-before"})
            with RangePolicyContext(inner) as inner_ctx:
                self.assertIs(active_range_policy(), inner)
                record_range_policy_event({"where": "inner"})
            self.assertIs(active_range_policy(), outer)
            record_range_policy_event({"where": "outer-after"})

        self.assertEqual(
            outer_ctx.events,
            [{"where": "outer-before"}, {"where": "outer-after"}],
        )
        self.assertEqual(inner_ctx.events, [{"where": "inner"}])
        self.assertEqual(active_range_policy().proof_bernstein_max_coefficients, 256)

    def test_concurrent_threads_do_not_share_policy_or_events(self):
        barrier = threading.Barrier(2)
        results = {}
        errors = []

        def worker(name, cap):
            try:
                policy = RangePolicy(proof_bernstein_max_coefficients=cap)
                with RangePolicyContext(policy) as ctx:
                    record_range_policy_event({"worker": name, "stage": "before"})
                    barrier.wait(timeout=5)
                    observed = active_range_policy().proof_bernstein_max_coefficients
                    record_range_policy_event({"worker": name, "stage": "after"})
                    results[name] = (observed, list(ctx.events))
            except BaseException as exc:  # preserve thread failures for assertion
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("a", 11))
        t2 = threading.Thread(target=worker, args=("b", 97))
        t1.start(); t2.start(); t1.join(10); t2.join(10)
        self.assertFalse(t1.is_alive() or t2.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results["a"][0], 11)
        self.assertEqual(results["b"][0], 97)
        self.assertEqual(
            results["a"][1],
            [{"worker": "a", "stage": "before"}, {"worker": "a", "stage": "after"}],
        )
        self.assertEqual(
            results["b"][1],
            [{"worker": "b", "stage": "before"}, {"worker": "b", "stage": "after"}],
        )


if __name__ == "__main__":
    unittest.main()
