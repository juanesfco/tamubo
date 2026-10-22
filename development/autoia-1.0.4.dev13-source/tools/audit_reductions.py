#!/usr/bin/env python3
"""Independent exact-rational stress audit for AutoIA trusted reductions."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia.trusted import (
    RigorousBinary64Backend,
    rigorous_dot_point,
    nonnegative_dot_upper,
    row_sum_norm_upper,
)
from tests.support.exact_binary64 import exact_float, interval_contains_exact, upper_endpoint_ge_exact


def exact_dot(a, b):
    return sum((exact_float(x) * exact_float(y) for x, y in zip(a, b)), Fraction(0, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dot-cases", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026082905)
    args = parser.parse_args()

    backend = RigorousBinary64Backend()
    rng = np.random.default_rng(args.seed)

    # Deterministic cancellation probes.
    probes = [
        (np.array([1e16, 1.0, -1e16]), np.ones(3)),
        (np.array([2.0**500, 1.0, -(2.0**500)]), np.ones(3)),
        (np.array([1.0, -1.0] * 1000 + [2.0**-1000]), np.ones(2001)),
    ]
    for a, b in probes:
        got = rigorous_dot_point(a, b, backend)
        exact = exact_dot(a, b)
        if not interval_contains_exact(got, exact):
            raise AssertionError(("cancellation probe", exact, got))

    checked = 0
    for _ in range(args.dot_cases):
        n = int(rng.integers(0, args.max_length + 1))
        ea = rng.integers(-700, 700, size=n)
        eb = rng.integers(-200, 200, size=n)
        a = np.ldexp(rng.uniform(-1.0, 1.0, size=n), ea).astype(float)
        b = np.ldexp(rng.uniform(-1.0, 1.0, size=n), eb).astype(float)
        got = rigorous_dot_point(a, b, backend)
        exact = exact_dot(a, b)
        if not interval_contains_exact(got, exact):
            raise AssertionError(("dot soundness", n, exact, got))
        checked += 1

    # Nonnegative upper reductions and row norms.
    upper_checked = 0
    for _ in range(max(1000, args.dot_cases // 5)):
        n = int(rng.integers(1, args.max_length + 1))
        a = np.abs(rng.normal(size=n))
        b = np.abs(rng.normal(size=n))
        got = nonnegative_dot_upper(a, b, backend)
        exact = exact_dot(a, b)
        if not upper_endpoint_ge_exact(got, exact):
            raise AssertionError(("nonnegative upper", n, exact, got))
        A = np.abs(rng.normal(size=(4, n)))
        rho = row_sum_norm_upper(A, backend)
        exact_rho = max(
            sum((exact_float(v) for v in row), Fraction(0, 1))
            for row in A
        )
        if not upper_endpoint_ge_exact(rho, exact_rho):
            raise AssertionError(("row norm", exact_rho, rho))
        upper_checked += 1

    print(f"Exact-rational dot-product checks passed: {checked:,}")
    print(f"Nonnegative upper-reduction checks passed: {upper_checked:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
