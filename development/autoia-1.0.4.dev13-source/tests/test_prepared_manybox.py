"""Step-9c prepared-session and many-box execution tests."""

from __future__ import annotations

import unittest
import numpy as np

from autoia import (
    autoia, prepare, PreparedAutoIA, Interval,
    InvalidInputError, AutoIAResult,
)


def process_test_model(p):
    x, y = p
    return np.exp(0.1*x) + x*y + 0.25*y*y


class TestPreparedSession(unittest.TestCase):
    def test_fresh_prepared_execution_matches_one_shot(self):
        def model(p):
            x, y = p
            A = np.zeros((2, 2))
            b = np.zeros(2)
            A[0, 0] = 2.0 + 0.1*x*x
            A[0, 1] = 0.05
            A[1, 0] = -0.02
            A[1, 1] = 1.5 + 0.05*y
            b[0] = 1.0 + x
            b[1] = 0.4 + y*y
            u = np.linalg.solve(A, b)
            return np.array([u[0], np.exp(u[1])])

        box = [Interval(-0.2, 0.2), Interval(-0.1, 0.1)]
        direct = autoia(model, box, order=4)
        session = prepare(model, order=4, cache_max_entries=0)
        prepared = session.certify(box, use_cache=False)
        np.testing.assert_array_equal(prepared.lower, direct.lower)
        np.testing.assert_array_equal(prepared.upper, direct.upper)
        self.assertEqual(session.stats.model_preparations, 1)
        self.assertEqual(session.stats.engineering_model_executions, 1)

    def test_exact_cache_avoids_model_reexecution(self):
        calls = [0]

        def model(p):
            calls[0] += 1
            return p[0]*p[0] + 1.0

        session = prepare(model, order=3)
        box = [Interval(-0.2, 0.3)]
        first = session.certify(box)
        second = session.certify(box)
        self.assertIsNot(first, second)
        self.assertEqual(first.cache_info["kind"], "fresh-certificate")
        self.assertEqual(second.cache_info["kind"], "exact-certificate")
        self.assertEqual(calls[0], 1)
        self.assertEqual(session.stats.exact_cache_hits, 1)
        self.assertFalse(second.cache_info["model_reexecuted"])

    def test_exact_cache_metadata_does_not_mutate_prior_result(self):
        def model(p):
            return p[0] * p[0]

        session = prepare(model, order=2)
        box = [Interval(-1.0, 1.0)]
        first = session.certify(box)
        self.assertEqual(first.cache_info["kind"], "fresh-certificate")
        second = session.certify(box)
        self.assertEqual(second.cache_info["kind"], "exact-certificate")
        self.assertEqual(first.cache_info["kind"], "fresh-certificate")
        self.assertIsNot(first, second)


    def test_interior_subbox_restricts_certificate_without_model_execution(self):
        calls = [0]

        def model(p):
            calls[0] += 1
            x, y = p
            return np.exp(0.2*x) + x*y + y*y

        session = prepare(model, order=5, cache_max_entries=32)
        parent = [Interval(-1.0, 1.0), Interval(-0.5, 0.5)]
        session.certify(parent)
        query = [Interval(-0.2, 0.3), Interval(-0.1, 0.15)]
        restricted = session.certify(query)
        self.assertEqual(calls[0], 1)
        self.assertEqual(restricted.cache_info["kind"], "restricted-certificate")
        self.assertEqual(session.stats.restricted_cache_hits, 1)

        # Dense truth sampling exercises the recentered certificate itself.
        for x in np.linspace(-0.2, 0.3, 31):
            for y in np.linspace(-0.1, 0.15, 25):
                truth = np.exp(0.2*x) + x*y + y*y
                self.assertLessEqual(restricted.lower[0], truth)
                self.assertGreaterEqual(restricted.upper[0], truth)

        fresh = autoia(model, query, order=5)
        # A restricted parent certificate may be wider than a fresh certificate,
        # but it must at least contain the fresh certified range in this smooth test.
        self.assertLessEqual(restricted.lower[0], fresh.lower[0])
        self.assertGreaterEqual(restricted.upper[0], fresh.upper[0])

    def test_boundary_query_falls_back_when_symmetric_recenter_would_escape_parent(self):
        calls = [0]

        def model(p):
            calls[0] += 1
            return p[0]*p[0] + 0.1*p[0]

        session = prepare(model, order=4)
        session.certify([Interval(-1.0, 1.0)])
        # The outward centered representation of a boundary-touching child may
        # protrude one ulp below -1.  Reuse must fail closed and rerun the model.
        r = session.certify([Interval(-1.0, -0.5)])
        self.assertEqual(calls[0], 2)
        self.assertEqual(r.cache_info["kind"], "fresh-certificate")

    def test_generator_restriction_preserves_shared_ids(self):
        def model(p):
            x = p[0]
            y = x*x
            return np.array([y, 2.0*y])

        session = prepare(model, order=1, representation="generator")
        parent = session.certify([Interval(-1.0, 1.0)])
        child = session.certify([Interval(-0.25, 0.25)])
        self.assertEqual(child.cache_info["kind"], "restricted-certificate")
        common = set(parent.generator_ids) & set(child.generator_ids)
        self.assertTrue(common)
        # The original shared generator should remain with the same 1:2 row relation.
        for gid in common:
            pcol = parent.generator_ids.index(gid)
            ccol = child.generator_ids.index(gid)
            if parent.generator_matrix[0, pcol] != 0.0:
                self.assertAlmostEqual(
                    child.generator_matrix[1, ccol],
                    2.0*child.generator_matrix[0, ccol], places=12,
                )
                break
        else:
            self.fail("No nonzero preserved shared generator was found.")

    def test_certify_many_endpoint_array_and_thread_mode_preserve_order(self):
        session = prepare(process_test_model, order=4, cache_max_entries=0)
        boxes = np.array([
            [[-0.2, -0.1], [-0.1, 0.1]],
            [[-0.1,  0.0], [-0.1, 0.1]],
            [[ 0.0,  0.1], [-0.1, 0.1]],
            [[ 0.1,  0.2], [-0.1, 0.1]],
        ])
        serial = session.certify_many(boxes, use_cache=False)
        threaded = session.certify_many(
            boxes, workers=2, parallel="thread", chunk_size=2, use_cache=False
        )
        self.assertEqual(len(serial), len(threaded))
        for a, b in zip(serial, threaded):
            np.testing.assert_array_equal(a.lower, b.lower)
            np.testing.assert_array_equal(a.upper, b.upper)

    def test_process_mode_supports_local_model_with_optional_cloudpickle(self):
        def local_model(p):
            return p[0] + 1.0

        session = prepare(local_model)
        try:
            import cloudpickle  # noqa: F401
        except Exception:
            with self.assertRaises(InvalidInputError):
                session.certify_many(
                    [[Interval(0.0, 1.0)]], workers=2, parallel="process"
                )
            return
        out = session.certify_many(
            [[Interval(0.0, 1.0)]], workers=2, parallel="process"
        )
        self.assertEqual(len(out), 1)
        self.assertLessEqual(out[0].lower[0], 1.0)
        self.assertGreaterEqual(out[0].upper[0], 2.0)

    def test_compact_many_box_bounds_interfaces(self):
        session = prepare(process_test_model, order=3, cache_max_entries=0)
        boxes = [
            [Interval(-0.1, 0.0), Interval(-0.02, 0.02)],
            [Interval(0.0, 0.1), Interval(-0.02, 0.02)],
            [Interval(0.1, 0.2), Interval(-0.02, 0.02)],
        ]
        lo, hi = session.certify_many_bounds(boxes, use_cache=False)
        self.assertEqual(lo.shape, (3, 1))
        self.assertEqual(hi.shape, (3, 1))
        streamed = list(session.iter_bounds_many(boxes, chunk_size=2, use_cache=False))
        self.assertEqual(len(streamed), 3)
        for i, (l, u) in enumerate(streamed):
            np.testing.assert_array_equal(l, lo[i])
            np.testing.assert_array_equal(u, hi[i])

    def test_streaming_many_box_interface(self):
        session = PreparedAutoIA(process_test_model, order=3, cache_max_entries=0)
        boxes = (
            [Interval(0.01*i, 0.01*i + 0.005), Interval(-0.02, 0.02)]
            for i in range(7)
        )
        out = list(session.iter_certify_many(boxes, chunk_size=3, use_cache=False))
        self.assertEqual(len(out), 7)
        self.assertTrue(all(isinstance(r, AutoIAResult) for r in out))


if __name__ == "__main__":
    unittest.main()
