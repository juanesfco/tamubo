"""Core representation-independent AutoIA execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..trusted import Interval, point_interval, RigorousBinary64Backend
from ..enclosures import (
    PolynomialEnclosure,
    IntervalEnclosure,
    SharedGeneratorPolynomialEnclosure,
    ValidatedValue,
    RangePolicyContext,
    BranchRuntime,
    BranchExecutionContext,
    DomainRuntime,
    DomainExecutionContext,
    ApproximationDiagnosticContext,
)
from ..enclosures.value import ExactAffineShadow
from ..semantic import (
    GenericSolveContext, ImplicitSolveContext, ImplicitSolveExecutionContext,
    LinearProgramContext, LinearProgramExecutionContext,
    execute_model_with_dispatch,
)
from .policies import RangePolicy
from .ranging import adaptive_final_range
from .diagnostics import FinalRangeDiagnostic, ProofRangePolicyEvent


@dataclass
class AutoIAResult:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    backend_name: str
    machine_rigorous: bool
    certificates: list
    requested_representation: str
    range_policy: RangePolicy
    final_range_diagnostics: list[FinalRangeDiagnostic]
    range_policy_events: list[ProofRangePolicyEvent]
    output_shape: tuple = ()

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def lower_shaped(self):
        return np.asarray(self.lower).reshape(self.output_shape)

    @property
    def upper_shaped(self):
        return np.asarray(self.upper).reshape(self.output_shape)

    @property
    def width_shaped(self):
        return np.asarray(self.width).reshape(self.output_shape)

    @property
    def objects_shaped(self):
        return np.asarray(self.objects, dtype=object).reshape(self.output_shape)

    @property
    def representation_names(self):
        return [v.representation_name for v in self.objects]

    @property
    def generator_ids(self):
        from ..enclosures import generator_matrix_from_values
        ids, _ = generator_matrix_from_values(self.objects)
        return ids

    @property
    def generator_matrix(self):
        from ..enclosures import generator_matrix_from_values
        _, G = generator_matrix_from_values(self.objects)
        return G

    def __repr__(self):
        lines = [
            f"AutoIAResult(machine_rigorous={self.machine_rigorous}, "
            "automatic_configuration=False)"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] in [{lo:.12g}, {hi:.12g}] width={hi-lo:.12g}"
            )
        lines.append(
            f"  configuration: fixed {self.requested_representation}; "
            f"advanced diagnostics available in final_range_diagnostics"
        )
        return "\n".join(lines)


# Historical name retained only as a migration compatibility alias.
AutoIAResultV11 = AutoIAResult


def rigorous_center_and_halfwidth(I: Interval, backend):
    """Choose a finite binary64 center and rigorously enclosing half-width."""
    if not np.isfinite(I.lo) or not np.isfinite(I.hi):
        raise ValueError("Version 0.6 reference implementation requires finite inputs.")

    center = float(0.5 * I.lo + 0.5 * I.hi)
    if not np.isfinite(center):
        center = float(I.lo)

    qbox = backend.sub(I, point_interval(center))
    h = max(abs(qbox.lo), abs(qbox.hi))
    if not np.isfinite(h):
        raise ValueError("Could not construct a finite centered parameter box.")
    return center, float(h)


def resolve_range_policy(range_method, internal_range_method,
                         final_range_method, range_policy):
    if range_policy is not None:
        if any(v is not None for v in (range_method, internal_range_method, final_range_method)):
            raise ValueError(
                "Pass either range_policy or the range_method/internal/final options, not both."
            )
        return range_policy

    if range_method is not None:
        return RangePolicy(
            internal_method=range_method,
            final_method=range_method,
        )

    return RangePolicy(
        internal_method=(internal_range_method or "termwise"),
        final_method=(final_range_method or "adaptive"),
    )


def autoia_single(model, parameter_intervals, order=4, range_method=None,
                  backend=None, representation="polynomial",
                  internal_range_method=None, final_range_method=None,
                  range_policy=None, *, branch_decisions=None, prepared_dispatch=None,
                  track_natural_domain=False):
    """Representation-independent certified execution entry point."""
    backend = backend if backend is not None else RigorousBinary64Backend()
    policy = resolve_range_policy(
        range_method, internal_range_method, final_range_method, range_policy
    )

    intervals = list(parameter_intervals)
    if not intervals:
        raise ValueError("At least one uncertain parameter is required.")
    if representation not in {"polynomial", "interval", "generator"}:
        raise ValueError("representation must be 'polynomial', 'interval', or 'generator'.")

    centers = np.empty(len(intervals))
    h = np.empty(len(intervals))
    for i, I in enumerate(intervals):
        centers[i], h[i] = rigorous_center_and_halfwidth(I, backend)

    p = np.empty(len(intervals), dtype=object)
    for i, (I, center) in enumerate(zip(intervals, centers)):
        if representation == "polynomial":
            rep = PolynomialEnclosure.parameter(
                center, i, h, order, backend, policy.internal_method
            )
        elif representation == "generator":
            rep = SharedGeneratorPolynomialEnclosure.parameter(
                center, i, h, order, backend, policy.internal_method
            )
        else:
            rep = IntervalEnclosure(I, h, order, backend, policy.internal_method)
        p[i] = ValidatedValue(
            rep, natural_interval=(I if track_natural_domain else None),
            _exact_affine_shadow=ExactAffineShadow.parameter(len(intervals), i),
        )

    def _resolve_predicate(predicate):
        # Branch decisions are proof obligations, so use the same strong final
        # range controller used for reported output certification.  When the
        # predicate remains exactly affine in the original physical inputs, an
        # exact-rational metadata shadow decides closed-box endpoint semantics
        # without being perturbed by outward-rounding noise in the enclosure.
        interval, meta = adaptive_final_range(predicate.difference, policy)
        exact_decision = predicate.decision_from_exact_affine_box(intervals)
        if exact_decision is not None:
            meta = dict(meta) if isinstance(meta, dict) else {"range_meta": meta}
            meta["exact_affine_decision"] = bool(exact_decision)
        return interval, meta

    branch_runtime = BranchRuntime(
        forced_decisions={} if branch_decisions is None else dict(branch_decisions),
        resolver=_resolve_predicate,
    )
    domain_runtime = DomainRuntime()
    policy_context = RangePolicyContext(policy)
    approx_context = ApproximationDiagnosticContext()
    with approx_context:
        with policy_context:
            with BranchExecutionContext(branch_runtime):
                with DomainExecutionContext(domain_runtime):
                    with GenericSolveContext(p[0]) as solve_context:
                        implicit_context = ImplicitSolveContext(p[0])
                        linear_program_context = LinearProgramContext(p[0], intervals)
                        with ImplicitSolveExecutionContext(implicit_context):
                            with LinearProgramExecutionContext(linear_program_context):
                                y = execute_model_with_dispatch(
                                    prepared_dispatch if prepared_dispatch is not None else model,
                                    p, solve_context
                                )
                        certificates = (
                            list(solve_context.certificates)
                            + list(implicit_context.certificates)
                            + list(linear_program_context.certificates)
                        )
    range_policy_events = list(policy_context.events)

    arr = np.asarray(y, dtype=object)
    objs = []
    lo = []
    hi = []
    final_meta = []
    for v in arr.flat:
        if not isinstance(v, ValidatedValue):
            v = p[0]._constant_like(v)
        r, meta = adaptive_final_range(v, policy)
        objs.append(v)
        lo.append(r.lo)
        hi.append(r.hi)
        final_meta.append(meta)

    result = AutoIAResult(
        np.asarray(lo), np.asarray(hi), objs,
        backend.name, backend.machine_rigorous,
        certificates, representation, policy, final_meta, range_policy_events, arr.shape,
    )
    result.branch_trace = tuple(branch_runtime.trace)
    result.branch_decisions = dict(branch_runtime.forced_decisions)
    result.domain_trace = tuple(domain_runtime.trace)
    result.approximation_diagnostics = approx_context.summary()
    return result


# Historical internal name retained for subdivision and migration tests.
autoia = autoia_single


__all__ = [
    "AutoIAResult",
    "AutoIAResultV11",
    "rigorous_center_and_halfwidth",
    "resolve_range_policy",
    "autoia_single",
    "autoia",
]
