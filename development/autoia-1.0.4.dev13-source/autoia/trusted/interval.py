"""Trusted interval primitives for AutoIA.

This module contains only the small interval data type and exact/non-arithmetic
helpers needed by the proof-critical binary64 kernel.  It is intentionally
independent of the higher-level AutoIA representation and controller layers.

The algorithms here are extracted unchanged from the Version 0.11 reference
core.  The reference-core snapshot remains available as
the test-only historical Version-0.11 oracle for migration regression.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .environment import require_binary64_environment, nextafter_binary64


@dataclass(frozen=True)
class Interval:
    """Closed interval with binary64 endpoints."""

    lo: float
    hi: float

    def __post_init__(self):
        lo = float(self.lo)
        hi = float(self.hi)
        if np.isnan(lo) or np.isnan(hi):
            raise ValueError("NaN is not allowed in an Interval.")
        if lo > hi:
            raise ValueError(f"Invalid interval [{lo}, {hi}]")
        # The trusted arithmetic contract is specifically a binary64-endpoint
        # contract.  Canonicalize accepted numeric inputs immediately so Python
        # ints, NumPy scalars, Decimal-like values, etc. cannot leak a wider or
        # non-binary64 arithmetic type into proof-critical operations.
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def mid(self):
        # Convenience only.  Proof code does not rely on this midpoint.
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self):
        # Convenience only.  Proof code does not rely on this radius.
        return 0.5 * (self.hi - self.lo)

    def __repr__(self):
        return f"Interval({self.lo:.17g}, {self.hi:.17g})"


def point_interval(x: float) -> Interval:
    """Interpret a stored binary64 number as an exact real point."""
    x = float(x)
    if np.isnan(x):
        raise ValueError("NaN point value is not supported.")
    return Interval(x, x)


def integer_interval(n: int, backend=None) -> Interval:
    """Rigorous binary64 enclosure of an exact Python integer.

    When ``backend`` is omitted this standalone helper performs the full
    binary64 environment audit, preserving the historical fail-closed API.
    Proof code that already owns a freshly audited rigorous backend may pass it
    here; the resulting exact-integer enclosure is then cached on that backend
    for the lifetime of the execution.  This avoids repeatedly re-auditing the
    same floating-point environment while constructing Taylor/combinatorial
    constants.
    """
    n = int(n)
    cache = None
    if backend is None:
        require_binary64_environment()
    else:
        cache = getattr(backend, "_integer_interval_cache", None)
        if cache is not None and n in cache:
            return cache[n]
        # A machine-rigorous backend audits the environment at construction.
        # Development backends make no rigor claim and need no runtime audit.
        if bool(getattr(backend, "machine_rigorous", False)) and not hasattr(backend, "environment_report"):
            require_binary64_environment()

    f = float(n)
    if not np.isfinite(f):
        raise OverflowError("Integer is too large for the binary64 reference backend.")
    nf = int(f)
    if nf == n:
        out = Interval(f, f)
    elif nf < n:
        out = Interval(f, nextafter_binary64(f, np.inf))
    else:
        out = Interval(nextafter_binary64(f, -np.inf), f)
    if cache is not None:
        cache[n] = out
    return out


def abs_upper_interval(x: Interval) -> float:
    """Exact comparison-based upper bound on |z| for z in ``x``."""
    return float(max(abs(x.lo), abs(x.hi)))


def interval_neg(x: Interval) -> Interval:
    """Exact unary negation of an interval with binary64 endpoints."""
    # Unary negation of a finite/extended binary64 endpoint is exact.
    return Interval(-x.hi, -x.lo)


__all__ = [
    "Interval",
    "point_interval",
    "integer_interval",
    "abs_upper_interval",
    "interval_neg",
]
