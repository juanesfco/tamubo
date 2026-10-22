#!/usr/bin/env python3
"""Independent high-precision stress audit for AutoIA elementary kernels.

This tool is not part of the installed AutoIA package.  It uses mpmath only as
an external high-precision test oracle; production proof code does not depend on
mpmath or on the platform transcendental functions being correctly rounded.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import numpy as np
from pathlib import Path

# Prefer the source tree containing this audit over any previously installed
# AutoIA version.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mpmath as mp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("audit_elementary.py requires optional test dependency mpmath") from exc

from autoia.trusted import (
    Interval,
    RigorousBinary64Backend,
    sqrt_point_interval,
    exp_point_interval,
    log_point_interval,
    log10_point_interval,
    sin_interval,
    cos_interval,
    pi_interval,
    ln2_interval,
    ln10_interval,
)


def mp_exact_float(x: float):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def contains(I, y):
    return mp_exact_float(I.lo) <= y <= mp_exact_float(I.hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2000, help="random cases per function family")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--dps", type=int, default=180)
    args = ap.parse_args()

    mp.mp.dps = args.dps
    rng = random.Random(args.seed)
    b = RigorousBinary64Backend()
    checks = 0

    constants = [(pi_interval(b), mp.pi, "pi"), (ln2_interval(b), mp.log(2), "ln2"), (ln10_interval(b), mp.log(10), "ln10")]
    for I, y, name in constants:
        if not contains(I, y):
            raise AssertionError((name, I, y))
        checks += 1

    # Deterministic exponent-boundary probes.  These complement random testing
    # and specifically guard the power-of-two scaling used by sqrt at both ends
    # of the finite binary64 lattice.
    positive_edges = [
        float.fromhex("0x0.0000000000001p-1022"),
        float(np.nextafter(np.finfo(np.float64).tiny, 0.0)),
        float(np.finfo(np.float64).tiny),
        1.0, 2.0,
        float(np.nextafter(np.finfo(np.float64).max, 0.0)),
        float(np.finfo(np.float64).max),
    ]
    for xp in positive_edges:
        xx = mp_exact_float(xp)
        for name, fn, exact_fn in [
            ("sqrt-edge", sqrt_point_interval, mp.sqrt),
            ("log-edge", log_point_interval, mp.log),
            ("log10-edge", log10_point_interval, mp.log10),
        ]:
            I = fn(xp, b)
            y = exact_fn(xx)
            if not contains(I, y):
                raise AssertionError((name, xp, I, y))
            checks += 1

    for i in range(args.cases):
        # Broad positive binary64 values for sqrt/log.
        e = rng.randint(-1070, 1000)
        m = rng.uniform(1.0, 2.0)
        try:
            xp = math.ldexp(m, e)
        except OverflowError:
            xp = math.inf
        if math.isfinite(xp) and xp > 0.0:
            for name, fn, exact_fn in [
                ("sqrt", sqrt_point_interval, mp.sqrt),
                ("log", log_point_interval, mp.log),
                ("log10", log10_point_interval, mp.log10),
            ]:
                I = fn(xp, b)
                y = exact_fn(mp_exact_float(xp))
                if not contains(I, y):
                    raise AssertionError((i, name, xp, I, y))
                checks += 1

        xe = rng.uniform(-740.0, 709.0)
        I = exp_point_interval(xe, b)
        y = mp.exp(mp_exact_float(xe))
        if not contains(I, y):
            raise AssertionError((i, "exp", xe, I, y))
        checks += 1

        xt = rng.uniform(-1e7, 1e7)
        xx = mp_exact_float(xt)
        for name, fn, exact_fn in [
            ("sin", sin_interval, mp.sin),
            ("cos", cos_interval, mp.cos),
        ]:
            I = fn(Interval(xt, xt), b)
            y = exact_fn(xx)
            if not contains(I, y):
                raise AssertionError((i, name, xt, I, y))
            checks += 1

    print(f"Elementary high-precision checks passed: {checks:,}")
    print(f"Random cases per family: {args.cases:,}")
    print(f"mpmath precision: {args.dps} decimal digits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
