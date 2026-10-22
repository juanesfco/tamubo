"""Reference binary64 interval arithmetic backend used by AutoIA proofs.

The trusted assumption is intentionally small and is now checked at runtime by
:mod:`autoia.trusted.environment`: Python ``float`` and NumPy ``float64`` must
behave as IEEE-754 binary64 with round-to-nearest and gradual underflow enabled.
Each proof-critical scalar result is expanded outward with the adjacent binary64
value returned by ``numpy.nextafter``.

Higher-level trusted reductions are built in :mod:`autoia.trusted.reductions`
from these scalar operations rather than from BLAS reductions.
"""

from __future__ import annotations

import numpy as np

from .interval import Interval, interval_neg
from .environment import require_binary64_environment, nextafter_binary64


class FloatBackend:
    """Development backend: same algorithms, but no machine-rigorous claim."""

    name = "float-development-v6"
    machine_rigorous = False

    @staticmethod
    def down(x):
        return float(x)

    @staticmethod
    def up(x):
        return float(x)

    def add(self, x: Interval, y: Interval) -> Interval:
        return Interval(x.lo + y.lo, x.hi + y.hi)

    def sub(self, x: Interval, y: Interval) -> Interval:
        return Interval(x.lo - y.hi, x.hi - y.lo)

    def mul(self, x: Interval, y: Interval) -> Interval:
        vals = [x.lo*y.lo, x.lo*y.hi, x.hi*y.lo, x.hi*y.hi]
        if any(np.isnan(v) for v in vals):
            raise FloatingPointError("NaN encountered in interval multiplication.")
        return Interval(min(vals), max(vals))

    def div(self, x: Interval, y: Interval) -> Interval:
        if y.lo <= 0.0 <= y.hi:
            raise ZeroDivisionError("Interval division by an interval containing zero.")
        vals = [x.lo/y.lo, x.lo/y.hi, x.hi/y.lo, x.hi/y.hi]
        if any(np.isnan(v) for v in vals):
            raise FloatingPointError("NaN encountered in interval division.")
        return Interval(min(vals), max(vals))

    def hull(self, x: Interval, y: Interval) -> Interval:
        return Interval(min(x.lo, y.lo), max(x.hi, y.hi))


class RigorousBinary64Backend(FloatBackend):
    """Reference outward-rounded binary64 interval backend.

    The constructor performs a fail-closed runtime audit of the binary64
    environment.  Each endpoint of each scalar ``+``, ``-``, ``*`` and ``/`` is
    then expanded by one adjacent binary64 value.  Higher-level proof
    calculations use only compositions of these trusted operations.
    """

    name = "binary64-nextafter-reference"
    machine_rigorous = True

    def __init__(self, reduction_method="auto", vectorized_reduction_min_terms=16):
        self.environment_report = require_binary64_environment()
        if reduction_method not in {"auto", "reference", "vectorized"}:
            raise ValueError("reduction_method must be 'auto', 'reference', or 'vectorized'.")
        if isinstance(vectorized_reduction_min_terms, bool) or int(vectorized_reduction_min_terms) < 1:
            raise ValueError("vectorized_reduction_min_terms must be a positive integer.")
        self.reduction_method = str(reduction_method)
        self.vectorized_reduction_min_terms = int(vectorized_reduction_min_terms)
        self.name = f"binary64-nextafter-{self.reduction_method}"
        self._reduction_stats = {
            "reference_calls": 0, "vectorized_calls": 0,
            "reference_terms": 0, "vectorized_terms": 0,
            "by_kind": {},
        }
        # Per-execution proof-value caches.  They are intentionally backend-local
        # so a fresh backend still reruns the floating-point environment audit.
        self._monomial_magnitude_cache = {}
        self._integer_interval_cache = {}

    def _record_reduction(self, kind, mode, terms):
        key = f"{mode}_calls"
        term_key = f"{mode}_terms"
        self._reduction_stats[key] = int(self._reduction_stats.get(key, 0)) + 1
        self._reduction_stats[term_key] = int(self._reduction_stats.get(term_key, 0)) + int(terms)
        by_kind = self._reduction_stats.setdefault("by_kind", {})
        item = by_kind.setdefault(str(kind), {"reference": 0, "vectorized": 0})
        item[str(mode)] = int(item.get(str(mode), 0)) + 1

    def _record_reduction_many(self, kind, mode, terms, count):
        count = int(count); terms = int(terms)
        if count <= 0:
            return
        key = f"{mode}_calls"; term_key = f"{mode}_terms"
        self._reduction_stats[key] = int(self._reduction_stats.get(key, 0)) + count
        self._reduction_stats[term_key] = int(self._reduction_stats.get(term_key, 0)) + count * terms
        by_kind = self._reduction_stats.setdefault("by_kind", {})
        item = by_kind.setdefault(str(kind), {"reference": 0, "vectorized": 0})
        item[str(mode)] = int(item.get(str(mode), 0)) + count

    @property
    def reduction_stats(self):
        out = dict(self._reduction_stats)
        out["by_kind"] = {k: dict(v) for k, v in self._reduction_stats.get("by_kind", {}).items()}
        return out

    @staticmethod
    def down(x):
        x = float(x)
        if np.isnan(x):
            raise FloatingPointError("NaN cannot be rounded downward.")
        if x == -np.inf:
            return -np.inf
        return nextafter_binary64(x, -np.inf)

    @staticmethod
    def up(x):
        x = float(x)
        if np.isnan(x):
            raise FloatingPointError("NaN cannot be rounded upward.")
        if x == np.inf:
            return np.inf
        return nextafter_binary64(x, np.inf)

    def add(self, x: Interval, y: Interval) -> Interval:
        # Exact identity cases avoid gratuitous ulp inflation.
        if y.lo == y.hi == 0.0:
            return x
        if x.lo == x.hi == 0.0:
            return y
        lo_rn = x.lo + y.lo
        hi_rn = x.hi + y.hi
        if (np.isfinite(x.lo) and np.isfinite(y.lo) and not np.isfinite(lo_rn)) or \
           (np.isfinite(x.hi) and np.isfinite(y.hi) and not np.isfinite(hi_rn)):
            raise FloatingPointError("Overflow in trusted interval addition.")
        return Interval(self.down(lo_rn), self.up(hi_rn))

    def sub(self, x: Interval, y: Interval) -> Interval:
        if y.lo == y.hi == 0.0:
            return x
        lo_rn = x.lo - y.hi
        hi_rn = x.hi - y.lo
        if (np.isfinite(x.lo) and np.isfinite(y.hi) and not np.isfinite(lo_rn)) or \
           (np.isfinite(x.hi) and np.isfinite(y.lo) and not np.isfinite(hi_rn)):
            raise FloatingPointError("Overflow in trusted interval subtraction.")
        return Interval(self.down(lo_rn), self.up(hi_rn))

    def mul(self, x: Interval, y: Interval) -> Interval:
        if x.lo == x.hi == 0.0 or y.lo == y.hi == 0.0:
            return Interval(0.0, 0.0)
        if x.lo == x.hi == 1.0:
            return y
        if y.lo == y.hi == 1.0:
            return x
        if x.lo == x.hi == -1.0:
            return interval_neg(y)
        if y.lo == y.hi == -1.0:
            return interval_neg(x)

        pairs = (
            (x.lo, y.lo), (x.lo, y.hi),
            (x.hi, y.lo), (x.hi, y.hi),
        )
        lows = []
        highs = []
        for a, b in pairs:
            z = a * b
            if np.isnan(z):
                raise FloatingPointError(
                    "NaN encountered in interval multiplication; "
                    "extended 0*inf cases are not supported by the reference backend."
                )
            if np.isfinite(a) and np.isfinite(b) and not np.isfinite(z):
                raise FloatingPointError("Overflow in trusted interval multiplication.")
            lows.append(self.down(z))
            highs.append(self.up(z))
        return Interval(min(lows), max(highs))

    def div(self, x: Interval, y: Interval) -> Interval:
        if y.lo <= 0.0 <= y.hi:
            raise ZeroDivisionError("Interval division by an interval containing zero.")
        if x.lo == x.hi == 0.0:
            return Interval(0.0, 0.0)
        if y.lo == y.hi == 1.0:
            return x
        if y.lo == y.hi == -1.0:
            return interval_neg(x)

        pairs = (
            (x.lo, y.lo), (x.lo, y.hi),
            (x.hi, y.lo), (x.hi, y.hi),
        )
        lows = []
        highs = []
        for a, b in pairs:
            z = a / b
            if np.isnan(z):
                raise FloatingPointError("NaN encountered in interval division.")
            if np.isfinite(a) and np.isfinite(b) and not np.isfinite(z):
                raise FloatingPointError("Overflow in trusted interval division.")
            lows.append(self.down(z))
            highs.append(self.up(z))
        return Interval(min(lows), max(highs))


# Backwards-friendly class alias retained for research code.
NextafterBackend = RigorousBinary64Backend


__all__ = [
    "FloatBackend",
    "RigorousBinary64Backend",
    "NextafterBackend",
]
