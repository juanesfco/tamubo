"""Independent exact-rational oracle for the scalar trusted-kernel audit.

This module deliberately does not use AutoIA interval arithmetic.  Every finite
Python float is converted to the exact rational number represented by its
binary64 bit pattern via ``Fraction.from_float``.  Exact arithmetic on those
rationals is then used to check that AutoIA's returned binary64 interval contains
the mathematical result.
"""

from __future__ import annotations

from fractions import Fraction
import math


def exact_float(x: float) -> Fraction:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("exact_float requires a finite float")
    return Fraction.from_float(x)


def exact_binary_op(a: float, b: float, op: str) -> Fraction:
    a_q = exact_float(a)
    b_q = exact_float(b)
    if op == "add":
        return a_q + b_q
    if op == "sub":
        return a_q - b_q
    if op == "mul":
        return a_q * b_q
    if op == "div":
        if b_q == 0:
            raise ZeroDivisionError
        return a_q / b_q
    raise ValueError(op)


def lower_endpoint_le_exact(endpoint: float, q: Fraction) -> bool:
    endpoint = float(endpoint)
    if endpoint == -math.inf:
        return True
    if not math.isfinite(endpoint):
        return False
    return exact_float(endpoint) <= q


def upper_endpoint_ge_exact(endpoint: float, q: Fraction) -> bool:
    endpoint = float(endpoint)
    if endpoint == math.inf:
        return True
    if not math.isfinite(endpoint):
        return False
    return q <= exact_float(endpoint)


def interval_contains_exact(interval, q: Fraction) -> bool:
    return (
        lower_endpoint_le_exact(interval.lo, q)
        and upper_endpoint_ge_exact(interval.hi, q)
    )


def exact_interval_extrema(x, y, op: str) -> tuple[Fraction, Fraction]:
    if op == "add":
        vals = [exact_float(x.lo) + exact_float(y.lo),
                exact_float(x.hi) + exact_float(y.hi)]
    elif op == "sub":
        vals = [exact_float(x.lo) - exact_float(y.hi),
                exact_float(x.hi) - exact_float(y.lo)]
    elif op == "mul":
        vals = [
            exact_float(a) * exact_float(b)
            for a in (x.lo, x.hi)
            for b in (y.lo, y.hi)
        ]
    elif op == "div":
        if y.lo <= 0.0 <= y.hi:
            raise ZeroDivisionError
        vals = [
            exact_float(a) / exact_float(b)
            for a in (x.lo, x.hi)
            for b in (y.lo, y.hi)
        ]
    else:
        raise ValueError(op)
    return min(vals), max(vals)


def interval_contains_exact_range(interval, lo: Fraction, hi: Fraction) -> bool:
    return lower_endpoint_le_exact(interval.lo, lo) and upper_endpoint_ge_exact(interval.hi, hi)


__all__ = [
    "exact_float",
    "exact_binary_op",
    "lower_endpoint_le_exact",
    "upper_endpoint_ge_exact",
    "interval_contains_exact",
    "exact_interval_extrema",
    "interval_contains_exact_range",
]
