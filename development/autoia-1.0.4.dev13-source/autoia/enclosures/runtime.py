"""Small representation-runtime helpers.

This module deliberately contains no controller algorithms.  It provides
sound interval intersection, a safe point representative, and context-local
proof-policy state used by representation-level proof obligations.

As of dev8, policy/event state is carried with :mod:`contextvars` rather than
process-global mutable stacks.  Nested AutoIA executions therefore restore
their parent context correctly, and concurrent threads/tasks do not share the
active policy by mutation.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import numpy as np

from ..trusted import Interval


def interval_intersection(a: Interval, b: Interval) -> Interval:
    """Intersect two independently sound scalar interval enclosures."""
    lo = max(a.lo, b.lo)
    hi = min(a.hi, b.hi)
    if lo > hi:
        raise RuntimeError(
            "Two independently sound interval enclosures are disjoint; "
            "treat this as an internal soundness error."
        )
    return Interval(lo, hi)


def safe_interval_representative(I: Interval) -> float:
    """Return a finite binary64 point lying inside ``I`` when possible."""
    if I.lo == I.hi:
        return float(I.lo)
    c = float(0.5 * I.lo + 0.5 * I.hi)
    if not np.isfinite(c):
        c = float(I.lo)
    return float(min(max(c, I.lo), I.hi))


@dataclass(frozen=True)
class _DefaultProofPolicy:
    """Fallback used only when a representation proof runs outside autoia()."""

    proof_bernstein_max_coefficients: int = 256


_ACTIVE_RANGE_POLICY: ContextVar[object | None] = ContextVar(
    "autoia_active_range_policy", default=None
)
_ACTIVE_RANGE_EVENTS: ContextVar[list[dict] | None] = ContextVar(
    "autoia_active_range_events", default=None
)


class RangePolicyContext:
    """Context-local dynamic proof policy and event recorder."""

    def __init__(self, policy):
        self.policy = policy
        self.events: list[dict] = []
        self._policy_token = None
        self._events_token = None

    def __enter__(self):
        self._policy_token = _ACTIVE_RANGE_POLICY.set(self.policy)
        self._events_token = _ACTIVE_RANGE_EVENTS.set(self.events)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset in reverse order of installation.  Tokens restore the exact
        # previous nested context rather than assuming a global stack discipline.
        if self._events_token is not None:
            _ACTIVE_RANGE_EVENTS.reset(self._events_token)
        if self._policy_token is not None:
            _ACTIVE_RANGE_POLICY.reset(self._policy_token)
        self._events_token = None
        self._policy_token = None
        return False



_ACTIVE_APPROX_DIAGNOSTICS: ContextVar[object | None] = ContextVar(
    "autoia_active_approx_diagnostics", default=None
)


class ApproximationDiagnosticContext:
    """Cheap non-proof provenance diagnostics for configuration selection.

    The runtime assigns small run-local bit tokens to unstructured remainder
    sources and records when two downstream values with overlapping remainder
    ancestry are recombined.  This information is *controller evidence only*;
    it never participates in a proof and false positives can at worst trigger
    an unnecessary generator retry.
    """

    def __init__(self, max_exact_tokens: int = 256):
        self.max_exact_tokens = int(max_exact_tokens)
        self.tokens_created = 0
        self.recombination_events = 0
        self.recombined_source_bits = 0
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_APPROX_DIAGNOSTICS.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._token is not None:
            _ACTIVE_APPROX_DIAGNOSTICS.reset(self._token)
        self._token = None
        return False

    def new_source_mask(self) -> int:
        # Exact distinct bit positions are cheap for the modest number of
        # remainder-creating events expected in a single model execution.
        # Above the cap, fold deterministically into the existing bit budget;
        # collisions only create conservative controller false positives.
        i = self.tokens_created
        self.tokens_created += 1
        if self.max_exact_tokens <= 0:
            return 0
        bit = i if i < self.max_exact_tokens else (i % self.max_exact_tokens)
        return 1 << bit

    def record_recombination(self, overlap_mask: int) -> None:
        overlap_mask = int(overlap_mask)
        if overlap_mask == 0:
            return
        self.recombination_events += 1
        self.recombined_source_bits |= overlap_mask

    def summary(self) -> dict:
        return {
            "approximation_sources_created": int(self.tokens_created),
            "shared_error_recombination_events": int(self.recombination_events),
            "shared_error_recombined_sources": int(self.recombined_source_bits.bit_count()),
            "shared_error_opportunity": bool(self.recombination_events > 0),
        }


def active_approximation_diagnostics():
    return _ACTIVE_APPROX_DIAGNOSTICS.get()


def new_approximation_source_mask() -> int:
    runtime = _ACTIVE_APPROX_DIAGNOSTICS.get()
    return 0 if runtime is None else int(runtime.new_source_mask())


def record_approximation_recombination(overlap_mask: int) -> None:
    runtime = _ACTIVE_APPROX_DIAGNOSTICS.get()
    if runtime is not None:
        runtime.record_recombination(int(overlap_mask))

def record_range_policy_event(event) -> None:
    events = _ACTIVE_RANGE_EVENTS.get()
    if events is not None:
        events.append(dict(event))


def active_range_policy():
    policy = _ACTIVE_RANGE_POLICY.get()
    if policy is not None:
        return policy
    return _DefaultProofPolicy()


__all__ = [
    "interval_intersection",
    "safe_interval_representative",
    "RangePolicyContext",
    "record_range_policy_event",
    "active_range_policy",
    "ApproximationDiagnosticContext",
    "active_approximation_diagnostics",
    "new_approximation_source_mask",
    "record_approximation_recombination",
]
