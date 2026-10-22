"""General rigorous real-domain predicate proving for validated operations.

A domain proof is a universal statement about the *actual physical input box*.
It is not a proposal computation and it never clips an interval to a desired
mathematical domain.

The prover may intersect several independently rigorous enclosures of the same
scalar expression:

* the optional natural-interval shadow propagated over the physical input box;
* the representation's ordinary polynomial/generator range;
* monotonicity ranging; and
* Bernstein ranging when the active proof-cost policy allows it.

If the resulting intersection proves the required predicate, execution may
continue.  If it proves that every possible value violates the predicate, the
operation fails.  Otherwise a :class:`DomainRefinementRequired` signal asks the
path controller to geometrically subdivide the physical input box and retry.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Mapping

from ..polynomial import _degree_vector_and_bernstein_cost
from ..trusted import ElementaryDomainError, Interval
from .runtime import (
    active_range_policy,
    interval_intersection,
    record_range_policy_event,
)


@dataclass(frozen=True)
class DomainRequirement:
    """One universal scalar real-domain requirement.

    ``relation`` is ``"ge"`` for ``x >= bound``, ``"gt"`` for ``x > bound``,
    or ``"nonzero"`` for ``x != 0``.  ``bound`` is currently a finite binary64
    scalar for ordered relations and is used as an exact real constant represented
    by that binary64 value.
    """

    operation: str
    relation: str
    bound: float = 0.0
    description: str = ""

    def __post_init__(self):
        if self.relation not in {"ge", "gt", "nonzero"}:
            raise ValueError("DomainRequirement.relation must be 'ge', 'gt', or 'nonzero'.")
        if self.relation == "nonzero" and float(self.bound) != 0.0:
            raise ValueError("nonzero domain requirements currently require bound=0.")

    def proved_by(self, interval: Interval) -> bool:
        if self.relation == "ge":
            return float(interval.lo) >= float(self.bound)
        if self.relation == "gt":
            return float(interval.lo) > float(self.bound)
        return float(interval.hi) < 0.0 or float(interval.lo) > 0.0

    def wholly_invalid_by(self, interval: Interval) -> bool:
        # This is deliberately stronger than merely failing the universal
        # predicate.  We report a proved violation only when the entire
        # certified image lies outside the allowed domain.
        if self.relation == "ge":
            return float(interval.hi) < float(self.bound)
        if self.relation == "gt":
            return float(interval.hi) <= float(self.bound)
        return float(interval.lo) == 0.0 and float(interval.hi) == 0.0

    @property
    def mathematical_text(self) -> str:
        if self.relation == "nonzero":
            return "argument separated from zero (argument != 0)"
        op = ">=" if self.relation == "ge" else ">"
        return f"argument {op} {self.bound:.17g}"


@dataclass(frozen=True)
class DomainProofResult:
    requirement: DomainRequirement
    interval: Interval
    status: str
    methods_tried: tuple[str, ...]
    estimated_bernstein_coefficients: int | None = None

    @property
    def proved(self) -> bool:
        return self.status == "proved"

    @property
    def wholly_invalid(self) -> bool:
        return self.status == "wholly-invalid"


class DomainRefinementRequired(Exception):
    """Internal signal that a universal real-domain predicate is unresolved."""

    def __init__(self, occurrence_index, value, proof: DomainProofResult):
        self.occurrence_index = int(occurrence_index)
        self.value = value
        self.proof = proof
        super().__init__(
            f"Domain proof {self.occurrence_index} for {proof.requirement.operation!r} "
            f"is unresolved on [{proof.interval.lo}, {proof.interval.hi}]."
        )


@dataclass
class DomainTraceEntry:
    occurrence_index: int
    operation: str
    relation: str
    bound: float
    status: str
    range_lo: float
    range_hi: float
    methods_tried: tuple[str, ...]


@dataclass
class DomainRuntime:
    """Context-local domain-proof state for one engineering-model execution."""

    next_occurrence: int = 0
    trace: list[DomainTraceEntry] = field(default_factory=list)

    def require(self, value, requirement: DomainRequirement) -> Interval:
        idx = self.next_occurrence
        self.next_occurrence += 1
        proof = prove_domain(value, requirement)
        self.trace.append(DomainTraceEntry(
            idx,
            requirement.operation,
            requirement.relation,
            float(requirement.bound),
            proof.status,
            float(proof.interval.lo),
            float(proof.interval.hi),
            tuple(proof.methods_tried),
        ))
        if proof.proved:
            return proof.interval
        if proof.wholly_invalid:
            raise ElementaryDomainError(
                f"Cannot evaluate {requirement.operation}: a rigorous domain proof "
                f"shows {requirement.mathematical_text} is false throughout the "
                f"current physical box; certified argument range is "
                f"[{proof.interval.lo}, {proof.interval.hi}]."
            )
        raise DomainRefinementRequired(idx, value, proof)


_ACTIVE_DOMAIN_RUNTIME: ContextVar[DomainRuntime | None] = ContextVar(
    "autoia_active_domain_runtime", default=None
)


class DomainExecutionContext:
    def __init__(self, runtime: DomainRuntime):
        self.runtime = runtime
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_DOMAIN_RUNTIME.set(self.runtime)
        return self.runtime

    def __exit__(self, exc_type, exc_value, traceback):
        _ACTIVE_DOMAIN_RUNTIME.reset(self._token)
        self._token = None
        return False


def active_domain_runtime() -> DomainRuntime | None:
    return _ACTIVE_DOMAIN_RUNTIME.get()


def _candidate_methods(rep) -> tuple[list[str], int | None]:
    """Return proof-range methods in inexpensive-to-expensive order."""
    name = getattr(rep, "representation_name", None)
    if name == "interval":
        return ["interval"], None

    methods: list[str] = []
    configured = str(getattr(rep, "range_method", "termwise"))
    if configured in {"termwise", "monotonicity", "bernstein"}:
        methods.append(configured)
    for method in ("termwise", "monotonicity"):
        if method not in methods:
            methods.append(method)

    bern_cost = None
    coeff = getattr(rep, "coeff", None)
    m = getattr(rep, "m", None)
    if coeff is not None and m is not None:
        _, bern_cost = _degree_vector_and_bernstein_cost(coeff, m)
        policy = active_range_policy()
        if int(bern_cost) <= int(policy.proof_bernstein_max_coefficients):
            if "bernstein" not in methods:
                methods.append("bernstein")
    return methods, bern_cost


def _status(requirement: DomainRequirement, interval: Interval) -> str:
    if requirement.proved_by(interval):
        return "proved"
    if requirement.wholly_invalid_by(interval):
        return "wholly-invalid"
    return "unresolved"


def prove_domain(value, requirement: DomainRequirement) -> DomainProofResult:
    """Try to prove one universal domain predicate without changing the model.

    Every range included in the intersection must independently enclose the
    expression on the actual physical input box.  Intersecting such enclosures
    can only tighten the proof and never removes a feasible exact-real value.
    """
    rep = getattr(value, "rep", value)
    natural = getattr(value, "natural_interval", None)
    methods, bern_cost = _candidate_methods(rep)

    current = None
    tried: list[str] = []

    def incorporate(label: str, interval: Interval):
        nonlocal current
        current = interval if current is None else interval_intersection(current, interval)
        tried.append(label)

    if isinstance(natural, Interval):
        incorporate("natural-physical-box", natural)
        st = _status(requirement, current)
        if st == "proved" or st == "wholly-invalid":
            result = DomainProofResult(
                requirement, current, st, tuple(tried), bern_cost
            )
            _record_domain_event(result)
            return result

    for method in methods:
        if method == "interval":
            interval, _ = rep.range()
        else:
            interval, _ = rep.range(method)
        incorporate(method, interval)
        st = _status(requirement, current)
        if st == "proved" or st == "wholly-invalid":
            result = DomainProofResult(
                requirement, current, st, tuple(tried), bern_cost
            )
            _record_domain_event(result)
            return result

    if current is None:
        current, _ = rep.range(getattr(rep, "range_method", None))
        tried.append("representation-range")

    result = DomainProofResult(
        requirement, current, _status(requirement, current), tuple(tried), bern_cost
    )
    _record_domain_event(result)
    return result


def _record_domain_event(result: DomainProofResult) -> None:
    record_range_policy_event({
        "kind": "domain-proof",
        "operation": result.requirement.operation,
        "relation": result.requirement.relation,
        "bound": float(result.requirement.bound),
        "methods_tried": tuple(result.methods_tried),
        "final_range": result.interval,
        "estimated_bernstein_coefficients": result.estimated_bernstein_coefficients,
        "status": result.status,
        "success": result.proved,
    })


def require_precomputed_domain(value, requirement: DomainRequirement, interval: Interval,
                               methods_tried=(), estimated_bernstein_coefficients=None) -> Interval:
    """Apply a domain requirement to an already-computed rigorous range.

    This is used by enclosure primitives such as reciprocal that already own a
    specialized proof portfolio.  An unresolved proof is routed through the same
    context-local geometric refinement mechanism as elementary-function domains.
    """
    st = _status(requirement, interval)
    result = DomainProofResult(
        requirement, interval, st, tuple(methods_tried), estimated_bernstein_coefficients
    )
    _record_domain_event(result)
    runtime = active_domain_runtime()
    if runtime is not None:
        idx = runtime.next_occurrence
        runtime.next_occurrence += 1
        runtime.trace.append(DomainTraceEntry(
            idx, requirement.operation, requirement.relation, float(requirement.bound),
            result.status, float(interval.lo), float(interval.hi), tuple(methods_tried),
        ))
        if result.proved:
            return interval
        if result.wholly_invalid:
            raise ElementaryDomainError(
                f"Cannot evaluate {requirement.operation}: a rigorous domain proof "
                f"shows {requirement.mathematical_text} is false throughout the "
                f"current physical box; certified argument range is "
                f"[{interval.lo}, {interval.hi}]."
            )
        raise DomainRefinementRequired(idx, value, result)

    if result.proved:
        return interval
    if result.wholly_invalid:
        raise ElementaryDomainError(
            f"Cannot evaluate {requirement.operation}: certified argument range "
            f"[{interval.lo}, {interval.hi}] violates {requirement.mathematical_text}."
        )
    raise ElementaryDomainError(
        f"Cannot certify the real domain for {requirement.operation}: the required "
        f"predicate {requirement.mathematical_text} remains unresolved; final rigorous "
        f"argument range is [{interval.lo}, {interval.hi}]."
    )


def require_domain(value, requirement: DomainRequirement) -> Interval:
    """Require a real-domain predicate in or outside an AutoIA execution."""
    runtime = active_domain_runtime()
    if runtime is not None:
        return runtime.require(value, requirement)

    proof = prove_domain(value, requirement)
    if proof.proved:
        return proof.interval
    if proof.wholly_invalid:
        raise ElementaryDomainError(
            f"Cannot evaluate {requirement.operation}: certified argument range "
            f"[{proof.interval.lo}, {proof.interval.hi}] lies outside the required "
            f"domain {requirement.mathematical_text}."
        )
    raise ElementaryDomainError(
        f"Cannot certify the real domain for {requirement.operation}: the required "
        f"predicate {requirement.mathematical_text} remains unresolved after "
        f"{', '.join(proof.methods_tried)}; final rigorous argument range is "
        f"[{proof.interval.lo}, {proof.interval.hi}]."
    )


def elementary_requirement(name: str) -> DomainRequirement | None:
    name = str(name)
    if name == "sqrt":
        return DomainRequirement("sqrt", "ge", 0.0, "square-root real domain")
    if name in {"log", "log10"}:
        return DomainRequirement(name, "gt", 0.0, "logarithm real domain")
    if name == "log1p":
        return DomainRequirement("log1p", "gt", -1.0, "log1p real domain")
    return None


def real_power_requirement(power: float) -> DomainRequirement | None:
    p = float(power)
    if p.is_integer():
        return None
    if p > 0.0:
        return DomainRequirement(
            f"real power {p!r}", "ge", 0.0,
            "non-integer positive-power real domain",
        )
    return DomainRequirement(
        f"real power {p!r}", "gt", 0.0,
        "non-integer negative-power real domain",
    )


__all__ = [
    "DomainRequirement",
    "DomainProofResult",
    "DomainRefinementRequired",
    "DomainTraceEntry",
    "DomainRuntime",
    "DomainExecutionContext",
    "active_domain_runtime",
    "prove_domain",
    "require_domain",
    "require_precomputed_domain",
    "request_sufficient_condition_refinement",
    "elementary_requirement",
    "real_power_requirement",
]


def request_sufficient_condition_refinement(value, *, operation: str, interval: Interval,
                                            description: str = "",
                                            methods_tried=()):
    """Request physical-box refinement for an inconclusive sufficient theorem.

    Unlike :func:`require_precomputed_domain`, this helper never interprets
    failure of the sufficient condition as proof that the underlying
    mathematical operation is invalid.  It is intended for theorem portfolios
    such as preconditioned SPD certification where, for example, ``rho < 1`` is
    sufficient for success but ``rho >= 1`` says only that this proof route is
    inconclusive on the current box.
    """
    requirement = DomainRequirement(
        str(operation), "gt", 0.0,
        str(description) or "sufficient proof condition positive",
    )
    result = DomainProofResult(
        requirement, interval, "unresolved", tuple(methods_tried), None
    )
    _record_domain_event(result)
    runtime = active_domain_runtime()
    if runtime is not None:
        idx = runtime.next_occurrence
        runtime.next_occurrence += 1
        runtime.trace.append(DomainTraceEntry(
            idx, requirement.operation, requirement.relation, float(requirement.bound),
            result.status, float(interval.lo), float(interval.hi), tuple(methods_tried),
        ))
        raise DomainRefinementRequired(idx, value, result)
    raise ElementaryDomainError(
        f"Cannot certify {operation} on the current physical box: the sufficient "
        f"proof condition {description or 'is positive'} was inconclusive; "
        f"certified proof-margin interval is [{interval.lo}, {interval.hi}]."
    )
