#!/usr/bin/env python3
"""Focused Step 9d phase-3 benchmark for packed separable Bernstein ranging."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import itertools
import time
import numpy as np

from autoia.trusted import RigorousBinary64Backend
from autoia.polynomial import (
    pack_point_polynomial_family,
    range_packed_point_bernstein,
    range_packed_point_monotonicity,
    range_packed_point_reduced_bernstein,
)
from autoia.polynomial.ranging import (
    range_bernstein_scalar,
    range_monotonicity_scalar,
    range_monotonicity_reduced_bernstein_scalar,
)


def total_degree_support(m, degree):
    return [
        alpha for alpha in itertools.product(range(degree + 1), repeat=m)
        if sum(alpha) <= degree
    ]


def scalar_full(coeff_arrays, support, h, backend):
    shape = next(iter(coeff_arrays.values())).shape
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        p = {alpha: float(coeff_arrays[alpha][idx]) for alpha in support}
        out[idx] = range_bernstein_scalar(p, h, backend)[0]
    return out


def scalar_reduced(coeff_arrays, support, h, backend):
    shape = next(iter(coeff_arrays.values())).shape
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        p = {alpha: float(coeff_arrays[alpha][idx]) for alpha in support}
        _, mm = range_monotonicity_scalar(p, h, backend)
        out[idx] = range_monotonicity_reduced_bernstein_scalar(
            p, h, backend, monotonicity_meta=mm
        )[0]
    return out


def main():
    rng = np.random.default_rng(2026083014)
    m = 3
    support = total_degree_support(m, 3)  # 20 retained terms, dense 4^3=64 grid
    family_shape = (8, 4)
    h = np.array([0.18, 0.24, 0.31])

    # Keep nonlinear terms small and impose dominant q0/q1 linear terms.  This
    # gives a representative reduced-Bernstein case with q2 usually unresolved.
    coeff_arrays = {
        alpha: rng.uniform(-0.025, 0.025, size=family_shape)
        for alpha in support
    }
    parity = np.indices(family_shape).sum(axis=0)
    coeff_arrays[(1, 0, 0)] = np.where(parity % 2 == 0, 2.0, -2.0)
    coeff_arrays[(0, 1, 0)] = np.where(parity % 3 == 0, 1.5, -1.5)
    coeff_arrays[(0, 0, 1)] = np.zeros(family_shape)
    coeff_arrays[(0, 0, 2)] = np.full(family_shape, 0.35)

    packed_backend = RigorousBinary64Backend(
        reduction_method="auto", vectorized_reduction_min_terms=16
    )
    scalar_backend = RigorousBinary64Backend()
    packed = pack_point_polynomial_family(coeff_arrays, m)

    # Warm structural code and normal allocator paths.
    range_packed_point_bernstein(packed, h, packed_backend)
    _, _, mm = range_packed_point_monotonicity(packed, h, packed_backend)
    range_packed_point_reduced_bernstein(
        packed, h, packed_backend, monotonicity_meta=mm
    )

    t0 = time.perf_counter()
    scalar_full_ranges = scalar_full(coeff_arrays, support, h, scalar_backend)
    full_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_lo, full_hi, full_meta = range_packed_point_bernstein(
        packed, h, packed_backend
    )
    full_packed = time.perf_counter() - t0

    t0 = time.perf_counter()
    scalar_reduced_ranges = scalar_reduced(coeff_arrays, support, h, scalar_backend)
    reduced_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, mm = range_packed_point_monotonicity(packed, h, packed_backend)
    red_lo, red_hi, red_meta = range_packed_point_reduced_bernstein(
        packed, h, packed_backend, monotonicity_meta=mm
    )
    reduced_packed = time.perf_counter() - t0

    for idx in np.ndindex(family_shape):
        ref = scalar_full_ranges[idx]
        if max(ref.lo, float(full_lo[idx])) > min(ref.hi, float(full_hi[idx])):
            raise AssertionError(("disjoint-full", idx))
        ref = scalar_reduced_ranges[idx]
        if max(ref.lo, float(red_lo[idx])) > min(ref.hi, float(red_hi[idx])):
            raise AssertionError(("disjoint-reduced", idx))

    unresolved_counts = [
        len(red_meta[idx]["unresolved_variables"])
        for idx in np.ndindex(family_shape)
    ]

    print(f"physical_dimensions={m}")
    print(f"term_count={len(support)}")
    print(f"dense_bernstein_coefficients={full_meta['num_coefficients']}")
    print(f"family_shape={family_shape}")
    print(f"family_members={int(np.prod(family_shape))}")
    print(f"full_scalar_seconds={full_scalar:.6f}")
    print(f"full_packed_seconds={full_packed:.6f}")
    print(f"full_speedup={full_scalar / full_packed:.3f}x")
    print(f"reduced_scalar_seconds={reduced_scalar:.6f}")
    print(f"reduced_packed_seconds={reduced_packed:.6f}")
    print(f"reduced_speedup={reduced_scalar / reduced_packed:.3f}x")
    print(f"unresolved_min={min(unresolved_counts)}")
    print(f"unresolved_max={max(unresolved_counts)}")
    print(f"unresolved_mean={float(np.mean(unresolved_counts)):.3f}")
    print(f"packed_reduction_stats={packed_backend.reduction_stats}")
    print("status=PASS")


if __name__ == "__main__":
    main()
