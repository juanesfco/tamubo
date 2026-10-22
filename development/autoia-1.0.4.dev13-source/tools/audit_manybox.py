#!/usr/bin/env python3
"""Step-9c repeated-query audit and benchmark.

The benchmark is intentionally small enough to run on a laptop.  Absolute
wall-clock values are machine dependent; containment and cache/model-execution
counts are the invariant checks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import autoia, prepare, Interval


def smooth_model(p):
    x, y, z = p
    a = np.exp(0.2*x) + np.sin(y) + x*y + z*z
    b = np.cos(0.3*z) + x*x*y + 0.1*np.exp(y)
    return a + b + 0.01*a*b


def make_fresh_boxes(n):
    out = []
    for i in range(n):
        c = -0.8 + 1.6*i/max(n - 1, 1)
        out.append([
            Interval(c - 0.01, c + 0.01),
            Interval(-0.05, 0.05),
            Interval(-0.04, 0.04),
        ])
    return out


def make_interior_queries(n):
    out = []
    for i in range(n):
        c = -0.75 + 1.5*i/max(n - 1, 1)
        out.append([
            Interval(c - 0.006, c + 0.006),
            Interval(-0.04, 0.04),
            Interval(-0.03, 0.03),
        ])
    return out


def elapsed(fn):
    t0 = time.perf_counter()
    value = fn()
    return time.perf_counter() - t0, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", type=int, default=40)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    n = int(args.boxes)
    workers = max(1, int(args.workers))
    fresh_boxes = make_fresh_boxes(n)

    # One-shot versus prepare-once with no certificate cache.  This quantifies
    # dispatch/backend preparation amortization separately from certificate reuse.
    t_one, one = elapsed(lambda: [autoia(smooth_model, b, order=4) for b in fresh_boxes])
    prepared = prepare(smooth_model, order=4, cache_max_entries=0)
    t_prepared, prep = elapsed(
        lambda: prepared.certify_many(fresh_boxes, use_cache=False)
    )
    for a, b in zip(one, prep):
        if not (np.array_equal(a.lower, b.lower) and np.array_equal(a.upper, b.upper)):
            raise AssertionError("Prepared fresh execution changed certified endpoints.")

    # Persistent master-domain certificate followed by interior restrictions.
    queries = make_interior_queries(n)
    master = [Interval(-1.0, 1.0), Interval(-0.2, 0.2), Interval(-0.15, 0.15)]
    cached = prepare(smooth_model, order=5, cache_max_entries=max(64, n + 4))
    cached.certify(master)
    t_restrict, restricted = elapsed(lambda: cached.certify_many(queries))

    fresh = prepare(smooth_model, order=5, cache_max_entries=0)
    t_fresh_queries, fresh_results = elapsed(
        lambda: fresh.certify_many(queries, use_cache=False)
    )
    if cached.stats.engineering_model_executions != 1:
        raise AssertionError("Interior certificate restriction unexpectedly reran the model.")
    if cached.stats.restricted_cache_hits != n:
        raise AssertionError("Not all interior queries used restricted certificates.")

    # Sample exact mathematical truth in every restricted query.
    sampled = 0
    for box, result in zip(queries, restricted):
        xs = np.linspace(box[0].lo, box[0].hi, 3)
        ys = np.linspace(box[1].lo, box[1].hi, 3)
        zs = np.linspace(box[2].lo, box[2].hi, 3)
        for x in xs:
            for y in ys:
                for z in zs:
                    a = np.exp(0.2*x) + np.sin(y) + x*y + z*z
                    b = np.cos(0.3*z) + x*x*y + 0.1*np.exp(y)
                    truth = a + b + 0.01*a*b
                    if not (result.lower[0] <= truth <= result.upper[0]):
                        raise AssertionError("Restricted certificate containment failure.")
                    sampled += 1

    # Exact repeat cache.
    repeated = prepare(smooth_model, order=4, cache_max_entries=8)
    repeated.certify(fresh_boxes[0])
    t_exact, _ = elapsed(lambda: [repeated.certify(fresh_boxes[0]) for _ in range(100)])
    if repeated.stats.engineering_model_executions != 1 or repeated.stats.exact_cache_hits != 100:
        raise AssertionError("Exact cache did not eliminate repeated model execution.")

    # Outer CPU parallelism for unrelated boxes.  Thread mode is measured too;
    # for the current Python-object-heavy executor process mode is normally the
    # useful CPU path.
    serial_parallel = prepare(smooth_model, order=4, cache_max_entries=0)
    t_serial, serial_results = elapsed(
        lambda: serial_parallel.certify_many(fresh_boxes, use_cache=False)
    )
    thread_parallel = prepare(smooth_model, order=4, cache_max_entries=0)
    t_thread, thread_results = elapsed(
        lambda: thread_parallel.certify_many(
            fresh_boxes, use_cache=False, parallel="thread",
            workers=workers, chunk_size=max(1, n // max(2*workers, 1)),
        )
    )
    process_parallel = prepare(smooth_model, order=4, cache_max_entries=0)
    t_process, process_results = elapsed(
        lambda: process_parallel.certify_many(
            fresh_boxes, use_cache=False, parallel="process",
            workers=workers, chunk_size=max(1, n // max(2*workers, 1)),
        )
    )
    for ref, other in zip(serial_results, process_results):
        if not (np.array_equal(ref.lower, other.lower) and np.array_equal(ref.upper, other.upper)):
            raise AssertionError("Process-parallel results differ from serial certified results.")
    for ref, other in zip(serial_results, thread_results):
        if not (np.array_equal(ref.lower, other.lower) and np.array_equal(ref.upper, other.upper)):
            raise AssertionError("Thread-parallel results differ from serial certified results.")

    print(f"Repeated-query audit boxes: {n}")
    print(f"Restricted-certificate truth samples passed: {sampled}")
    print()
    print(f"One-shot repeated autoia:       {t_one:.6f} s")
    print(f"Prepared fresh execution:      {t_prepared:.6f} s")
    print(f"Preparation-only speed ratio:  {t_one/t_prepared:.3f}x")
    print()
    print(f"Fresh interior queries:        {t_fresh_queries:.6f} s")
    print(f"Restricted cached queries:     {t_restrict:.6f} s")
    print(f"Certificate-restriction speed: {t_fresh_queries/t_restrict:.3f}x")
    print(f"Model executions for {n} cached queries after master proof: "
          f"{cached.stats.engineering_model_executions}")
    print(f"100 exact-cache hits time:      {t_exact:.6f} s")
    print()
    print(f"Fresh serial many-box:         {t_serial:.6f} s")
    print(f"Fresh thread many-box ({workers} workers): {t_thread:.6f} s  "
          f"({t_serial/t_thread:.3f}x)")
    print(f"Fresh process many-box ({workers} workers): {t_process:.6f} s  "
          f"({t_serial/t_process:.3f}x)")


if __name__ == "__main__":
    main()
