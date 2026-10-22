#!/usr/bin/env python3
"""Focused Step-9d benchmark: scalar vs packed interval-coefficient ranging."""

from __future__ import annotations

import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia.polynomial import range_interval_polynomial_family
from autoia.semantic.linear_solve import range_interval_coefficient_poly
from autoia.trusted import Interval, RigorousBinary64Backend


def make_problem(n=28, m=4, order=3, seed=2026082905):
    rng = np.random.default_rng(seed)
    support = []
    for a0 in range(order + 1):
        for a1 in range(order + 1 - a0):
            for a2 in range(order + 1 - a0 - a1):
                for a3 in range(order + 1 - a0 - a1 - a2):
                    support.append((a0, a1, a2, a3))
    coeff = {}
    for alpha in support:
        center = rng.normal(scale=0.04, size=(n, n))
        if alpha == (0,) * m:
            center += np.eye(n)
        radius = rng.uniform(0.0, 2.0e-12, size=(n, n))
        arr = np.empty((n, n), dtype=object)
        for idx in np.ndindex(arr.shape):
            arr[idx] = Interval(center[idx] - radius[idx], center[idx] + radius[idx])
        coeff[alpha] = arr
    h = np.array([0.11, 0.16, 0.09, 0.13], dtype=float)
    return coeff, h


def scalar_run(coeff, h, backend):
    shape = next(iter(coeff.values())).shape
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        p = {alpha: values[idx] for alpha, values in coeff.items()}
        out[idx], _ = range_interval_coefficient_poly(p, h, backend, "termwise")
    return out


def packed_run(coeff, h, backend):
    return range_interval_polynomial_family(coeff, h, backend, "termwise")[0]


def timed(fn, repeats=3):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def main():
    coeff, h = make_problem()
    b_scalar = RigorousBinary64Backend(reduction_method="auto", vectorized_reduction_min_terms=16)
    b_packed = RigorousBinary64Backend(reduction_method="auto", vectorized_reduction_min_terms=16)

    scalar_time, scalar = timed(lambda: scalar_run(coeff, h, b_scalar), repeats=2)
    packed_time, packed = timed(lambda: packed_run(coeff, h, b_packed), repeats=4)

    for idx in np.ndindex(scalar.shape):
        lo = max(scalar[idx].lo, packed[idx].lo)
        hi = min(scalar[idx].hi, packed[idx].hi)
        if lo > hi:
            raise AssertionError(f"Disjoint scalar/packed enclosures at {idx}")

    print("Step 9d packed interval-coefficient ranging benchmark")
    print(f"matrix_shape={scalar.shape}")
    print(f"term_count={len(coeff)}")
    print(f"scalar_seconds={scalar_time:.6f}")
    print(f"packed_seconds={packed_time:.6f}")
    print(f"speedup={scalar_time/packed_time:.3f}x")
    print("packed_reduction_stats=", b_packed.reduction_stats)


if __name__ == "__main__":
    main()
