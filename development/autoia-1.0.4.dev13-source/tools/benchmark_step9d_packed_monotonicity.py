#!/usr/bin/env python3
"""Focused Step 9d phase-2 benchmark for batched derivative/monotonicity ranging."""

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
    range_packed_point_derivatives,
    range_packed_point_monotonicity,
)
from autoia.polynomial.ranging import derivative_range_termwise, range_monotonicity_scalar


def total_degree_support(m, degree):
    return [
        alpha for alpha in itertools.product(range(degree + 1), repeat=m)
        if sum(alpha) <= degree
    ]


def scalar_derivatives(coeff_arrays, support, h, backend):
    shape = next(iter(coeff_arrays.values())).shape
    out = np.empty(shape + (len(h),), dtype=object)
    for idx in np.ndindex(shape):
        p = {alpha: float(coeff_arrays[alpha][idx]) for alpha in support}
        for i in range(len(h)):
            out[idx + (i,)] = derivative_range_termwise(p, i, h, backend)
    return out


def scalar_monotonicity(coeff_arrays, support, h, backend):
    shape = next(iter(coeff_arrays.values())).shape
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        p = {alpha: float(coeff_arrays[alpha][idx]) for alpha in support}
        out[idx] = range_monotonicity_scalar(p, h, backend)[0]
    return out


def main():
    rng = np.random.default_rng(2026083011)
    m = 4
    support = total_degree_support(m, 3)  # 35 terms
    family_shape = (28, 28)
    h = np.array([0.18, 0.24, 0.31, 0.27])
    coeff_arrays = {alpha: rng.normal(size=family_shape) for alpha in support}
    packed = pack_point_polynomial_family(coeff_arrays, m)

    scalar_backend = RigorousBinary64Backend()
    packed_backend = RigorousBinary64Backend(
        reduction_method="auto", vectorized_reduction_min_terms=16
    )

    # Warm the structural and monomial caches used in normal prepared execution.
    range_packed_point_derivatives(packed, h, packed_backend)
    range_packed_point_monotonicity(packed, h, packed_backend)

    t0 = time.perf_counter()
    scalar_derivatives(coeff_arrays, support, h, scalar_backend)
    derivative_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    dlo, dhi = range_packed_point_derivatives(packed, h, packed_backend)
    derivative_packed = time.perf_counter() - t0

    t0 = time.perf_counter()
    scalar_ranges = scalar_monotonicity(coeff_arrays, support, h, scalar_backend)
    monotonicity_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    plo, phi, meta = range_packed_point_monotonicity(packed, h, packed_backend)
    monotonicity_packed = time.perf_counter() - t0

    # Sanity: two sound enclosures of the same polynomial range must overlap.
    for idx in np.ndindex(family_shape):
        ref = scalar_ranges[idx]
        if max(ref.lo, float(plo[idx])) > min(ref.hi, float(phi[idx])):
            raise AssertionError(("disjoint", idx))

    print(f"physical_dimensions={m}")
    print(f"term_count={len(support)}")
    print(f"family_shape={family_shape}")
    print(f"family_members={int(np.prod(family_shape))}")
    print(f"derivative_scalar_seconds={derivative_scalar:.6f}")
    print(f"derivative_packed_seconds={derivative_packed:.6f}")
    print(f"derivative_speedup={derivative_scalar / derivative_packed:.3f}x")
    print(f"monotonicity_scalar_seconds={monotonicity_scalar:.6f}")
    print(f"monotonicity_packed_seconds={monotonicity_packed:.6f}")
    print(f"monotonicity_speedup={monotonicity_scalar / monotonicity_packed:.3f}x")
    print(f"unique_sign_patterns={meta['unique_sign_patterns'].shape[0]}")
    print(f"packed_reduction_stats={packed_backend.reduction_stats}")
    print("status=PASS")


if __name__ == "__main__":
    main()
