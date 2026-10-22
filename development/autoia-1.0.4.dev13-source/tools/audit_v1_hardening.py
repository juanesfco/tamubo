#!/usr/bin/env python3
"""Deterministic Version-1 adversarial/cross-platform hardening audit.

This tool is designed to be run unchanged on every candidate CPU/OS/Python
combination.  It records a platform fingerprint, exercises the fail-closed
binary64 environment contract, checks critical scalar and reduction cases
against exact rational arithmetic, checks trusted elementary endpoints against
high precision, and exercises release-critical cross-layer scenarios.

The audit is independent validation evidence, not part of AutoIA's proof.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mpmath as mp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("audit_v1_hardening.py requires optional audit dependency mpmath") from exc

try:
    import scipy
except ImportError:  # pragma: no cover
    scipy = None

from autoia import autoia, prepare, Interval, CertificationError, __version__
from autoia.trusted import (
    RigorousBinary64Backend,
    audit_binary64_environment,
    rigorous_dot_point,
    sqrt_point_interval,
    log_point_interval,
    exp_point_interval,
)
from tests.support.exact_binary64 import (
    exact_binary_op,
    exact_float,
    interval_contains_exact,
    lower_endpoint_le_exact,
    upper_endpoint_ge_exact,
)
from tests.support.hardening_models import (
    prepared_process_model,
    near_singular_semantic_model,
    near_singular_generic_model,
    one_ulp_branch_model,
)


MIN_SUB = float.fromhex("0x0.0000000000001p-1022")
MAX_FINITE_Q = exact_float(sys.float_info.max)


def critical_values() -> list[float]:
    vals = {
        0.0, -0.0, MIN_SUB, -MIN_SUB, 2*MIN_SUB, -2*MIN_SUB,
        float(np.nextafter(sys.float_info.min, 0.0)),
        -float(np.nextafter(sys.float_info.min, 0.0)),
        sys.float_info.min, -sys.float_info.min,
        0.5, -0.5, 1.0, -1.0,
        float(np.nextafter(1.0, -math.inf)),
        float(np.nextafter(1.0, math.inf)),
        float(np.nextafter(sys.float_info.max, 0.0)),
        -float(np.nextafter(sys.float_info.max, 0.0)),
        sys.float_info.max, -sys.float_info.max,
    }
    for exponent in (-1000, -500, -1, 0, 1, 500, 1000):
        x = math.ldexp(1.0, exponent)
        vals.update((x, -x, float(np.nextafter(x, 0.0)), float(np.nextafter(x, math.inf))))
    return sorted(float(x) for x in vals)


def mp_exact_float(x: float):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def contains_mp(I, y) -> bool:
    return mp_exact_float(I.lo) <= y <= mp_exact_float(I.hi)


def exact_reciprocal_range(lo: float, hi: float) -> tuple[Fraction, Fraction]:
    return Fraction(1, 1) / exact_float(hi), Fraction(1, 1) / exact_float(lo)


def platform_fingerprint(environment_report) -> dict:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": None if scipy is None else scipy.__version__,
        "mpmath": mp.__version__,
        "byteorder": sys.byteorder,
        "float_info": {
            "radix": sys.float_info.radix,
            "mant_dig": sys.float_info.mant_dig,
            "max_exp": sys.float_info.max_exp,
            "min_exp": sys.float_info.min_exp,
            "rounds": sys.float_info.rounds,
        },
        "binary64_environment": {
            "supported": environment_report.supported,
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in environment_report.checks
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--random-reductions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026083019)
    args = ap.parse_args()

    started = time.perf_counter()
    env = audit_binary64_environment()
    if not env.supported:
        raise AssertionError(env.summary())
    backend = RigorousBinary64Backend()
    mp.mp.dps = 200
    rng = np.random.default_rng(args.seed)
    counts: dict[str, int] = {}
    notes: list[str] = []

    # 1. Exact scalar lattice audit.
    scalar = 0
    overflow_fail_closed = 0
    vals = critical_values()
    for a in vals:
        for b in vals:
            for op in ("add", "sub", "mul", "div"):
                if op == "div" and b == 0.0:
                    continue
                exact = exact_binary_op(a, b, op)
                try:
                    got = getattr(backend, op)(Interval(a, a), Interval(b, b))
                except FloatingPointError:
                    if abs(exact) <= MAX_FINITE_Q:
                        raise AssertionError(("unexpected finite scalar failure", op, a, b, exact))
                    overflow_fail_closed += 1
                    continue
                if not interval_contains_exact(got, exact):
                    raise AssertionError(("scalar containment", op, a, b, exact, got))
                scalar += 1
    counts["exact_scalar_lattice"] = scalar
    counts["scalar_overflow_fail_closed"] = overflow_fail_closed

    # 2. Reduction tree boundaries plus randomized exponent-stress cases.
    reduction = 0
    for n in (15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129):
        families = []
        a = np.empty(n, dtype=float)
        a[:-1] = np.resize(np.array([2.0**500, -(2.0**500)]), n-1)
        a[-1] = 2.0**-1000
        families.append((a, np.ones(n)))
        families.append((np.full(n, MIN_SUB), np.full(n, 0.5)))
        for a, b in families:
            exact = sum((exact_float(x)*exact_float(y) for x, y in zip(a, b)), Fraction(0, 1))
            for mode in ("reference", "vectorized", "auto"):
                got = rigorous_dot_point(a, b, RigorousBinary64Backend(reduction_method=mode))
                if not interval_contains_exact(got, exact):
                    raise AssertionError(("reduction boundary", n, mode, exact, got))
                reduction += 1

    for _ in range(args.random_reductions):
        n = int(rng.integers(1, 130))
        ea = rng.integers(-900, 900, size=n)
        eb = rng.integers(-100, 100, size=n)
        a = np.ldexp(rng.uniform(-1, 1, size=n), ea).astype(float)
        b = np.ldexp(rng.uniform(-1, 1, size=n), eb).astype(float)
        products = [exact_float(x)*exact_float(y) for x, y in zip(a, b)]
        if any(abs(q) > MAX_FINITE_Q for q in products):
            continue
        exact = sum(products, Fraction(0, 1))
        for mode in ("reference", "vectorized"):
            got = rigorous_dot_point(a, b, RigorousBinary64Backend(reduction_method=mode))
            if not interval_contains_exact(got, exact):
                raise AssertionError(("random reduction", n, mode, exact, got))
            reduction += 1
    counts["exact_reductions"] = reduction

    # 3. High-precision trusted elementary endpoint audit.
    elementary = 0
    pos_values = [
        MIN_SUB, float(np.nextafter(sys.float_info.min, 0.0)), sys.float_info.min,
        0.5, 1.0, 2.0, 2.0**500,
        float(np.nextafter(sys.float_info.max, 0.0)), sys.float_info.max,
    ]
    for x in pos_values:
        xx = mp_exact_float(x)
        for name, fn, exact_fn in (
            ("sqrt", sqrt_point_interval, mp.sqrt),
            ("log", log_point_interval, mp.log),
        ):
            got = fn(x, backend)
            exact = exact_fn(xx)
            if not contains_mp(got, exact):
                raise AssertionError((name, x, got, exact))
            elementary += 1

    # exp values include deep subnormal outputs and a value very close to the
    # finite overflow frontier.  If the last case cannot be enclosed without an
    # intermediate binary64 overflow, fail-closed is acceptable and recorded.
    exp_fail_closed = 0
    for x in (-745.2, -745.13, -740.0, -1.0, 0.0, 1.0, 700.0, 709.78, 709.7827, 709.782712893384):
        exact = mp.exp(mp_exact_float(x))
        try:
            got = exp_point_interval(x, backend)
        except FloatingPointError:
            if exact <= mp_exact_float(sys.float_info.max):
                exp_fail_closed += 1
                notes.append(f"exp({x!r}) is finite but currently fails closed near the binary64 overflow frontier")
                continue
            raise
        if not contains_mp(got, exact):
            raise AssertionError(("exp", x, got, exact))
        elementary += 1
    counts["high_precision_elementary"] = elementary
    counts["finite_exp_frontier_fail_closed"] = exp_fail_closed

    # 4. Near-singular semantic solve against exact analytic range.
    solve_checks = 0
    for exponent in (20, 30, 40, 50):
        delta = 2.0**(-exponent)
        radius = 2.0**(-(exponent+2))

        def model(p, delta=delta):
            x = p[0]
            A = np.array([[1.0, 0.0], [0.0, delta+x]], dtype=object)
            return np.linalg.solve(A, np.array([1.0, 1.0], dtype=object))

        result = autoia(model, [Interval(-radius, radius)], order=4)
        lo, hi = exact_reciprocal_range(delta-radius, delta+radius)
        if not lower_endpoint_le_exact(result.lower[1], lo):
            raise AssertionError(("solve lower", exponent, result.lower[1], lo))
        if not upper_endpoint_ge_exact(result.upper[1], hi):
            raise AssertionError(("solve upper", exponent, result.upper[1], hi))
        solve_checks += 2
    counts["near_singular_exact_endpoint_checks"] = solve_checks

    delta = 2.0**-30
    try:
        autoia(near_singular_semantic_model, [Interval(-1.1*delta, 0.1*delta)], order=4)
    except CertificationError:
        counts["near_singular_crossing_fail_closed"] = 1
    else:
        raise AssertionError("near-singular zero-crossing family unexpectedly certified")

    # 5. Semantic-vs-generic equivalence means common exact inclusion, not bit
    # identity; proof paths may reorder outward arithmetic.
    radius = 2.0**-32
    box = [Interval(-radius, radius)]
    sem = autoia(near_singular_semantic_model, box, order=4)
    gen = autoia(near_singular_generic_model, box, order=4)
    lo, hi = exact_reciprocal_range(delta-radius, delta+radius)
    for result in (sem, gen):
        if not lower_endpoint_le_exact(result.lower[1], lo) or not upper_endpoint_ge_exact(result.upper[1], hi):
            raise AssertionError(("semantic-generic exact inclusion", result.lower, result.upper))
    if max(sem.lower[1], gen.lower[1]) > min(sem.upper[1], gen.upper[1]):
        raise AssertionError("semantic and generic solve enclosures are disjoint")
    counts["semantic_generic_equivalence_checks"] = 5

    # 6. One-ulp branch boundary.
    left = float(np.nextafter(1.0, -math.inf))
    right = float(np.nextafter(1.0, math.inf))
    branch = autoia(one_ulp_branch_model, [Interval(left, right)], order=3)
    exact_left = exact_float(left) - Fraction(1, 1)
    exact_right = 2*(exact_float(right) - Fraction(1, 1))
    if not lower_endpoint_le_exact(branch.lower[0], exact_left) or not upper_endpoint_ge_exact(branch.upper[0], exact_right):
        raise AssertionError(("branch boundary", branch.lower, branch.upper))
    counts["one_ulp_branch_boundary"] = 2

    # 7. Prepared serial/thread/process equivalence under explicit spawn.
    boxes = np.array([
        [[-0.2, -0.1], [-0.1, 0.1]],
        [[-0.1,  0.0], [-0.1, 0.1]],
        [[ 0.0,  0.1], [-0.1, 0.1]],
        [[ 0.1,  0.2], [-0.1, 0.1]],
    ])
    session = prepare(prepared_process_model, order=4, cache_max_entries=0)
    serial = session.certify_many(boxes, parallel="serial", use_cache=False)
    threaded = session.certify_many(boxes, workers=2, parallel="thread", chunk_size=2, use_cache=False)
    process = session.certify_many(boxes, workers=2, parallel="process", chunk_size=2, use_cache=False)
    parallel_checks = 0
    for base, other in zip(serial, threaded):
        if not np.array_equal(base.lower.view(np.uint64), other.lower.view(np.uint64)) or not np.array_equal(base.upper.view(np.uint64), other.upper.view(np.uint64)):
            raise AssertionError("threaded prepared result differs from serial")
        parallel_checks += 2
    for base, other in zip(serial, process):
        if not np.array_equal(base.lower.view(np.uint64), other.lower.view(np.uint64)) or not np.array_equal(base.upper.view(np.uint64), other.upper.view(np.uint64)):
            raise AssertionError("spawn-process prepared result differs from serial")
        parallel_checks += 2
    counts["prepared_parallel_bitwise_checks"] = parallel_checks

    output = {
        "status": "PASS",
        "checkpoint": __version__,
        "elapsed_seconds": time.perf_counter() - started,
        "platform": platform_fingerprint(env),
        "counts": counts,
        "notes": notes,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
