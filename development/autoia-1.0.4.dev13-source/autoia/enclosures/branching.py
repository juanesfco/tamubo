"""Validated scalar predicates and context-local path decisions.

This module contains only scalar predicate semantics and internal control-flow
signals.  The controller owns geometric subdivision and path enumeration.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Mapping


class BranchDecisionRequired(BaseException):
    """Internal control signal: a scalar predicate is not rigorously decided."""

    def __init__(self, occurrence_index, predicate, interval, metadata=None):
        super().__init__(f"Validated branch decision {occurrence_index} is unresolved.")
        self.occurrence_index = int(occurrence_index)
        self.predicate = predicate
        self.interval = interval
        self.metadata = {} if metadata is None else dict(metadata)


class InfeasiblePath(BaseException):
    """Internal control signal: a previously forced path contradicts this box."""

    def __init__(self, occurrence_index, forced, rigorous):
        super().__init__(
            f"Forced branch decision {occurrence_index}={forced} contradicts "
            f"rigorous decision {rigorous}."
        )
        self.occurrence_index = int(occurrence_index)
        self.forced = bool(forced)
        self.rigorous = bool(rigorous)


@dataclass
class BranchTraceEntry:
    occurrence_index: int
    operator: str
    rigorous_decision: bool | None
    chosen_decision: bool | None
    forced: bool
    range_lo: float
    range_hi: float


@dataclass
class BranchRuntime:
    """Context-local branch-decision state for one model execution."""

    forced_decisions: Mapping[int, bool] = field(default_factory=dict)
    resolver: Callable | None = None
    next_occurrence: int = 0
    trace: list[BranchTraceEntry] = field(default_factory=list)

    def decide(self, predicate):
        idx = self.next_occurrence
        self.next_occurrence += 1

        if self.resolver is None:
            interval, metadata = predicate.range()
        else:
            interval, metadata = self.resolver(predicate)
        rigorous = None
        if isinstance(metadata, dict) and "exact_affine_decision" in metadata:
            rigorous = bool(metadata["exact_affine_decision"])
        if rigorous is None:
            rigorous = predicate.decision_from_interval(interval)
        has_forced = idx in self.forced_decisions

        if has_forced:
            forced = bool(self.forced_decisions[idx])
            if rigorous is not None and rigorous != forced:
                self.trace.append(BranchTraceEntry(
                    idx, predicate.operator, rigorous, None, True,
                    float(interval.lo), float(interval.hi),
                ))
                raise InfeasiblePath(idx, forced, rigorous)
            self.trace.append(BranchTraceEntry(
                idx, predicate.operator, rigorous, forced, True,
                float(interval.lo), float(interval.hi),
            ))
            return forced

        if rigorous is not None:
            self.trace.append(BranchTraceEntry(
                idx, predicate.operator, rigorous, rigorous, False,
                float(interval.lo), float(interval.hi),
            ))
            return rigorous

        self.trace.append(BranchTraceEntry(
            idx, predicate.operator, None, None, False,
            float(interval.lo), float(interval.hi),
        ))
        signal = BranchDecisionRequired(idx, predicate, interval, metadata)
        signal.trace = tuple(self.trace)
        raise signal


_ACTIVE_BRANCH_RUNTIME: ContextVar[BranchRuntime | None] = ContextVar(
    "autoia_active_branch_runtime", default=None
)


class BranchExecutionContext:
    """Install one context-local branch runtime for a model execution."""

    def __init__(self, runtime: BranchRuntime):
        self.runtime = runtime
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_BRANCH_RUNTIME.set(self.runtime)
        return self.runtime

    def __exit__(self, exc_type, exc, tb):
        _ACTIVE_BRANCH_RUNTIME.reset(self._token)
        return False


def active_branch_runtime() -> BranchRuntime | None:
    return _ACTIVE_BRANCH_RUNTIME.get()


class ValidatedPredicate:
    """A scalar real comparison whose truth value may depend on uncertainty.

    The predicate is represented by a validated difference ``d = lhs-rhs`` and
    an operator comparing ``d`` with zero.  A Boolean value is returned only if
    the entire certified range proves one outcome, unless the path controller
    explicitly forces one outcome while enumerating an ambiguous path.
    """

    __slots__ = ("difference", "operator")

    def __init__(self, difference, operator: str):
        if operator not in {"lt", "le", "gt", "ge", "eq", "ne"}:
            raise ValueError(f"Unknown validated comparison operator {operator!r}.")
        self.difference = difference
        self.operator = operator

    @property
    def representation_name(self):
        return self.difference.representation_name

    def range(self, method=None):
        return self.difference.range(method)

    def decision_from_exact_affine_box(self, box):
        shadow = getattr(self.difference, "exact_affine_shadow", None)
        if shadow is None:
            return None
        return shadow.decision_on_box(self.operator, box)

    def exact_affine_root(self, box):
        shadow = getattr(self.difference, "exact_affine_shadow", None)
        if shadow is None:
            return None
        return shadow.exactly_representable_single_root(box)

    def decision_from_interval(self, interval):
        lo = float(interval.lo)
        hi = float(interval.hi)
        op = self.operator
        if op == "lt":
            if hi < 0.0:
                return True
            if lo >= 0.0:
                return False
        elif op == "le":
            if hi <= 0.0:
                return True
            if lo > 0.0:
                return False
        elif op == "gt":
            if lo > 0.0:
                return True
            if hi <= 0.0:
                return False
        elif op == "ge":
            if lo >= 0.0:
                return True
            if hi < 0.0:
                return False
        elif op == "eq":
            if lo == 0.0 and hi == 0.0:
                return True
            if hi < 0.0 or lo > 0.0:
                return False
        elif op == "ne":
            if hi < 0.0 or lo > 0.0:
                return True
            if lo == 0.0 and hi == 0.0:
                return False
        return None

    def __bool__(self):
        runtime = active_branch_runtime()
        if runtime is not None:
            return runtime.decide(self)
        interval, _ = self.range()
        decision = self.decision_from_interval(interval)
        if decision is None:
            raise RuntimeError(
                "An uncertain validated comparison requires AutoIA path-aware execution."
            )
        return decision

    def __repr__(self):
        interval, _ = self.range()
        return (
            f"ValidatedPredicate(op={self.operator!r}, "
            f"difference_range=[{interval.lo:.12g}, {interval.hi:.12g}])"
        )


__all__ = [
    "ValidatedPredicate",
    "BranchDecisionRequired",
    "InfeasiblePath",
    "BranchTraceEntry",
    "BranchRuntime",
    "BranchExecutionContext",
    "active_branch_runtime",
]
