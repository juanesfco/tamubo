#!/usr/bin/env python3
"""Standalone independent audit of AutoIA's scalar trusted binary64 backend.

Example
-------
    python tools/audit_scalar_binary64.py --random-point-ops 100000 \
        --random-interval-ops 10000

The audit uses exact Python ``Fraction`` arithmetic as an independent oracle for
the real result of finite binary64 operands.  It exits nonzero on the first
soundness failure and reports the number of successful checks otherwise.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia.trusted import Interval, RigorousBinary64Backend, audit_binary64_environment
from tests.support.exact_binary64 import (
    exact_float,
    exact_binary_op,
    interval_contains_exact,
    exact_interval_extrema,
    interval_contains_exact_range,
)

MAX_FINITE_Q = exact_float(sys.float_info.max)


def finite_float_from_bits(bits: int) -> float:
    return struct.unpack(">d", int(bits).to_bytes(8, "big"))[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-point-ops", type=int, default=50000)
    parser.add_argument("--random-interval-ops", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    report = audit_binary64_environment()
    print("Environment audit:", report.summary())
    if not report.supported:
        return 2

    backend = RigorousBinary64Backend()
    rng = np.random.default_rng(args.seed)

    values: list[float] = []
    target_values = max(5000, min(50000, args.random_point_ops // 2 + 1))
    while len(values) < target_values:
        bits = int(rng.integers(0, 2**64, dtype=np.uint64))
        x = finite_float_from_bits(bits)
        if math.isfinite(x):
            values.append(x)

    point_checked = 0
    overflow_fail_closed = 0
    for _ in range(args.random_point_ops):
        a = values[int(rng.integers(0, len(values)))]
        b = values[int(rng.integers(0, len(values)))]
        op = ("add", "sub", "mul", "div")[int(rng.integers(0, 4))]
        if op == "div" and b == 0.0:
            continue
        exact = exact_binary_op(a, b, op)
        try:
            got = getattr(backend, op)(Interval(a, a), Interval(b, b))
        except FloatingPointError:
            if abs(exact) <= MAX_FINITE_Q:
                raise AssertionError(("unexpected trusted overflow", op, a, b, exact))
            overflow_fail_closed += 1
            continue
        if not interval_contains_exact(got, exact):
            raise AssertionError(("point soundness failure", op, a, b, exact, got))
        point_checked += 1

    interval_checked = 0
    for _ in range(args.random_interval_ops):
        aa = np.sort(rng.uniform(-1.0e150, 1.0e150, size=2))
        bb = np.sort(rng.uniform(-1.0e150, 1.0e150, size=2))
        x = Interval(float(aa[0]), float(aa[1]))
        y = Interval(float(bb[0]), float(bb[1]))
        for op in ("add", "sub", "mul"):
            got = getattr(backend, op)(x, y)
            lo, hi = exact_interval_extrema(x, y, op)
            if not interval_contains_exact_range(got, lo, hi):
                raise AssertionError(("interval soundness failure", op, x, y, got))
            interval_checked += 1
        if not (y.lo <= 0.0 <= y.hi):
            got = backend.div(x, y)
            lo, hi = exact_interval_extrema(x, y, "div")
            if not interval_contains_exact_range(got, lo, hi):
                raise AssertionError(("interval soundness failure", "div", x, y, got))
            interval_checked += 1

    print(f"Exact-rational point checks passed: {point_checked:,}")
    print(f"Fail-closed finite-operation overflow cases: {overflow_fail_closed:,}")
    print(f"Exact-rational interval checks passed: {interval_checked:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
