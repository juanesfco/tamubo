"""Automatic representation/order configuration utilities.

This module is deliberately *not* trusted numerical arithmetic.  It constructs,
compares, and schedules independently rigorous AutoIA runs.  A poor selector
choice may waste time or miss a tighter optional enclosure; it cannot make an
accepted certificate unsound.

Step 20a has two controller modes:

``reference``
    An intentionally expensive bounded portfolio used as a research oracle.
``auto``
    A cheap structural selector.  It starts from the caller's baseline and only
    escalates when the baseline itself exposes one of two representation-level
    mechanisms: unstructured remainder dominates retained physical variation,
    or approximation-error ancestry is reused and recombined downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class ConfigurationPolicy:
    """Bounded representation/order configuration policy.

    Parameters
    ----------
    mode:
        ``"fixed"`` runs only the caller's baseline, ``"reference"`` runs the
        small explicit portfolio, and ``"auto"`` uses cheap structural
        diagnostics from the baseline to decide whether an optional retry is
        justified.
    extra_orders, extra_representations:
        Reference-oracle candidates.  They are ignored by the cheap ``auto``
        selector except as explicit expert research controls.
    max_attempts:
        Hard cap including the baseline attempt.
    auto_order_increment:
        When truncation/remainder dominates retained physical variation, the
        cheap selector retries at ``baseline_order + auto_order_increment``.
    auto_try_higher_order, auto_try_generator:
        Expert switches for the two cheap escalation mechanisms.

    No fitted weights or application-specific formulas appear in the release
    selector.  Its two triggers are structural and interpretable.
    """

    mode: str = "auto"
    extra_orders: tuple[int, ...] = (5,)
    extra_representations: tuple[str, ...] = ("generator",)
    max_attempts: int = 3
    auto_order_increment: int = 1
    auto_try_higher_order: bool = True
    auto_try_generator: bool = True

    def __post_init__(self):
        if self.mode not in {"fixed", "reference", "auto"}:
            raise ValueError(
                "ConfigurationPolicy.mode must be 'fixed', 'reference', or 'auto'."
            )
        for order in self.extra_orders:
            if isinstance(order, bool) or int(order) < 0:
                raise ValueError("extra_orders must contain nonnegative integers.")
        for rep in self.extra_representations:
            if rep not in {"polynomial", "generator", "interval"}:
                raise ValueError("Invalid representation in extra_representations.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        if isinstance(self.auto_order_increment, bool) or int(self.auto_order_increment) < 1:
            raise ValueError("auto_order_increment must be a positive integer.")


@dataclass
class AutoIAConfiguredResult:
    """Numeric certificate formed by intersecting independent rigorous runs.

    The final interval hull is the componentwise intersection of all successful
    configuration attempts.  ``selected_result`` is one real structural result
    retained for expert inspection; the intersected certificate itself is not
    claimed to be one polynomial/generator representation.
    """

    lower: np.ndarray
    upper: np.ndarray
    attempt_results: list
    selected_result: object
    backend_name: str
    machine_rigorous: bool
    configuration_policy: ConfigurationPolicy
    configuration_diagnostics: dict
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
    def representation_names(self):
        return ["configuration-intersection"] * len(self.lower)

    @property
    def objects(self):
        return getattr(self.selected_result, "objects", [])

    @property
    def certificates(self):
        return getattr(self.selected_result, "certificates", [])

    # Structural/path metadata belongs to the selected independently rigorous
    # result.  The configured wrapper's numeric bounds may be the tighter
    # intersection of several attempts, but users should not lose access to
    # the execution-path certificate simply because automatic configuration
    # wrapped that result.
    @property
    def branch_results(self):
        return getattr(self.selected_result, "branch_results", [])

    @property
    def branch_boxes(self):
        return getattr(self.selected_result, "branch_boxes", [])

    @property
    def branch_decisions(self):
        return getattr(self.selected_result, "branch_decisions", [])

    @property
    def branch_traces(self):
        return getattr(self.selected_result, "branch_traces", [])

    @property
    def path_events(self):
        return getattr(self.selected_result, "path_events", [])

    @property
    def automatic_configuration(self):
        return self.configuration_policy.mode == "auto"

    @property
    def configuration_summary(self):
        attempts = tuple(self.configuration_diagnostics.get("attempts", ()))
        if not attempts:
            return "no configuration attempts recorded"
        successful = [a for a in attempts if a.get("success")]
        if len(attempts) == 1:
            return "automatic baseline was sufficient; no optional retry was triggered"
        reason_labels = {
            "remainder-dominates-retained-variation": "higher-order retry",
            "shared-error-recombination": "shared-error retry",
            "baseline-failed-higher-order-fallback": "higher-order fallback",
            "baseline-failed-generator-fallback": "shared-error fallback",
            "reference-portfolio": "reference retry",
        }
        reasons = [
            reason_labels.get(str(a.get("reason", "optional retry")), "optional retry")
            for a in attempts[1:]
        ]
        return (
            f"{len(successful)} rigorous configuration(s) contributed; "
            + ", ".join(reasons)
        )

    def explain_configuration(self):
        """Return a compact human-readable explanation of automatic choices.

        Full machine-readable details remain available through
        ``configuration_diagnostics``.  This helper is informational only and
        never participates in proof arithmetic.
        """
        attempts = tuple(self.configuration_diagnostics.get("attempts", ()))
        lines = [
            f"Configuration mode: {self.configuration_policy.mode}",
            f"Baseline: {self.configuration_diagnostics.get('baseline_representation')}/"
            f"order-{self.configuration_diagnostics.get('baseline_order')}",
        ]
        for i, a in enumerate(attempts):
            status = "certified" if a.get("success") else "failed closed"
            lines.append(
                f"Attempt {i+1}: {a.get('representation')}/order-{a.get('order')} - "
                f"{status}; reason={a.get('reason', 'reference-portfolio')}"
            )
        lines.append(
            "Returned bounds are the componentwise intersection of all successful rigorous attempts."
        )
        return "\n".join(lines)

    def __repr__(self):
        lines = [
            f"AutoIAResult(machine_rigorous={self.machine_rigorous}, "
            f"automatic_configuration={self.automatic_configuration})"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] in [{lo:.12g}, {hi:.12g}] width={hi-lo:.12g}"
            )
        if self.branch_results:
            lines.append(
                f"  selected structural certificate: {len(self.branch_results)} execution branch(es)"
            )
        lines.append(f"  configuration: {self.configuration_summary}")
        return "\n".join(lines)


def configuration_candidates(
    policy: ConfigurationPolicy, *, baseline_representation: str, baseline_order: int
) -> tuple[tuple[str, int], ...]:
    """Return the deterministic, duplicate-free reference candidate sequence."""
    candidates: list[tuple[str, int]] = [(baseline_representation, int(baseline_order))]
    if policy.mode == "reference":
        for order in policy.extra_orders:
            candidates.append((baseline_representation, int(order)))
        for rep in policy.extra_representations:
            candidates.append((rep, int(baseline_order)))

    out: list[tuple[str, int]] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= policy.max_attempts:
            break
    return tuple(out)


def interval_result_dominates(candidate, incumbent) -> bool:
    """Return True when one result is no wider on every output and tighter on one."""
    clo = np.asarray(candidate.lower, dtype=float)
    chi = np.asarray(candidate.upper, dtype=float)
    ilo = np.asarray(incumbent.lower, dtype=float)
    ihi = np.asarray(incumbent.upper, dtype=float)
    if clo.shape != ilo.shape or chi.shape != ihi.shape:
        return False
    weak = np.all(clo >= ilo) and np.all(chi <= ihi)
    strict = np.any(clo > ilo) or np.any(chi < ihi)
    return bool(weak and strict)


def select_dominating_result(successful_results: Iterable):
    """Select by enclosure dominance, preserving the baseline on incomparability."""
    results = list(successful_results)
    if not results:
        raise ValueError("At least one successful result is required.")
    selected_index = 0
    selected = results[0]
    for i, candidate in enumerate(results[1:], start=1):
        if interval_result_dominates(candidate, selected):
            selected = candidate
            selected_index = i
    return selected_index, selected


def rigorous_intersection_bounds(successful_results):
    """Return the componentwise intersection of successful rigorous result hulls."""
    results = list(successful_results)
    if not results:
        raise ValueError("At least one successful result is required.")
    lower = np.max(np.stack([np.asarray(r.lower, dtype=float) for r in results]), axis=0)
    upper = np.min(np.stack([np.asarray(r.upper, dtype=float) for r in results]), axis=0)
    if np.any(lower > upper):
        raise RuntimeError(
            "Independently rigorous configuration results have disjoint output enclosures."
        )
    return lower, upper


def _result_value_groups(result):
    """Return one list of scalar structural values for each reported output."""
    branch_results = getattr(result, "branch_results", None)
    if branch_results is None:
        return [[v] for v in getattr(result, "objects", ())]
    nout = len(np.asarray(result.lower).reshape(-1))
    groups = [[] for _ in range(nout)]
    for branch in branch_results:
        for j, value in enumerate(getattr(branch, "objects", ())):
            if j < nout:
                groups[j].append(value)
    return groups


def _one_value_structure(value) -> dict:
    rep = getattr(value, "rep", None)
    if rep is None:
        return {
            "remainder_width": 0.0,
            "polynomial_termwise_width": 0.0,
            "polynomial_terms": 0,
            "generator_count": 0,
            "shared_error_recombination": bool(getattr(value, "_diag_recombined", False)),
        }

    rem = getattr(rep, "remainder", None)
    remainder_width = 0.0
    if rem is not None:
        remainder_width = max(0.0, float(rem.hi) - float(rem.lo))

    polynomial_width = 0.0
    coeff = getattr(rep, "coeff", None)
    if coeff is not None:
        try:
            # Ranging the retained polynomial only is a cheap diagnostic.  It is
            # not used as a proof result by the configuration selector.
            from ..polynomial import range_polynomial
            pr, _ = range_polynomial(coeff, rep.h, rep.backend, method="termwise")
            polynomial_width = max(0.0, float(pr.hi) - float(pr.lo))
        except Exception:
            polynomial_width = 0.0

    return {
        "remainder_width": float(remainder_width),
        "polynomial_termwise_width": float(polynomial_width),
        "polynomial_terms": int(len(coeff)) if coeff is not None else 0,
        "generator_count": int(len(getattr(rep, "generators", {}))),
        "shared_error_recombination": bool(getattr(value, "_diag_recombined", False)),
        "remainder_dominates_retained_variation": bool(
            remainder_width > 0.0 and remainder_width >= polynomial_width
        ),
    }


def _approximation_summary(result) -> dict:
    direct = getattr(result, "approximation_diagnostics", None)
    if direct is not None:
        return dict(direct)
    branches = getattr(result, "branch_results", None)
    if branches is None:
        return {
            "approximation_sources_created": 0,
            "shared_error_recombination_events": 0,
            "shared_error_recombined_sources": 0,
            "shared_error_opportunity": False,
        }
    entries = [getattr(b, "approximation_diagnostics", {}) for b in branches]
    return {
        "approximation_sources_created": int(sum(int(e.get("approximation_sources_created", 0)) for e in entries)),
        "shared_error_recombination_events": int(sum(int(e.get("shared_error_recombination_events", 0)) for e in entries)),
        "shared_error_recombined_sources": int(sum(int(e.get("shared_error_recombined_sources", 0)) for e in entries)),
        "shared_error_opportunity": bool(any(bool(e.get("shared_error_opportunity", False)) for e in entries)),
    }


def structural_summary(result) -> dict:
    """Extract cheap post-run diagnostics for release-policy decisions."""
    groups = _result_value_groups(result)
    output_summaries = []
    for j, values in enumerate(groups):
        parts = [_one_value_structure(v) for v in values]
        final_width = float(np.asarray(result.width, dtype=float).reshape(-1)[j])
        output_summaries.append({
            "output_index": int(j),
            "final_width": final_width,
            "max_remainder_width": max((p["remainder_width"] for p in parts), default=0.0),
            "max_polynomial_termwise_width": max((p["polynomial_termwise_width"] for p in parts), default=0.0),
            "max_polynomial_terms": max((p["polynomial_terms"] for p in parts), default=0),
            "max_generator_count": max((p["generator_count"] for p in parts), default=0),
            "shared_error_recombination": bool(any(p["shared_error_recombination"] for p in parts)),
            "remainder_dominates_retained_variation": bool(any(
                p["remainder_dominates_retained_variation"] for p in parts
            )),
        })

    final_diags = []
    branch_results = getattr(result, "branch_results", None)
    if branch_results is None:
        final_diags.extend(getattr(result, "final_range_diagnostics", ()))
    else:
        for branch_result in branch_results:
            final_diags.extend(getattr(branch_result, "final_range_diagnostics", ()))

    reduced_costs = [
        int(d.get("estimated_reduced_bernstein_coefficients", 0))
        for d in final_diags
        if d.get("estimated_reduced_bernstein_coefficients") is not None
    ]
    unresolved_fractions = [
        float(d.get("unresolved_fraction", 0.0))
        for d in final_diags
        if d.get("unresolved_fraction") is not None
    ]
    approx = _approximation_summary(result)
    return {
        "max_remainder_width": max((o["max_remainder_width"] for o in output_summaries), default=0.0),
        "sum_remainder_width": float(sum(o["max_remainder_width"] for o in output_summaries)),
        "max_polynomial_terms": max((o["max_polynomial_terms"] for o in output_summaries), default=0),
        "max_generator_count": max((o["max_generator_count"] for o in output_summaries), default=0),
        "max_reduced_bernstein_cost": max(reduced_costs, default=0),
        "max_unresolved_fraction": max(unresolved_fractions, default=0.0),
        "outputs": tuple(output_summaries),
        **approx,
    }


def automatic_escalation_candidates(
    baseline_result,
    policy: ConfigurationPolicy,
    *,
    baseline_representation: str,
    baseline_order: int,
) -> tuple[tuple[str, int, str], ...]:
    """Return cheap deterministic retries justified by baseline structure.

    The selector has no fitted thresholds:

    * higher order is tried only when an output's unstructured remainder is at
      least as wide as its retained physical-polynomial variation **and** that
      output is not already marked as a shared-error recombination;
    * generator mode is tried only when a reported output contains downstream
      recombination of shared approximation-error ancestry, retains nonzero
      unstructured remainder, and its retained physical-polynomial variation
      has cancelled exactly to zero.
    """
    if policy.mode != "auto" or policy.max_attempts <= 1:
        return ()
    summary = structural_summary(baseline_result)
    outputs = summary.get("outputs", ())
    out: list[tuple[str, int, str]] = []

    if policy.auto_try_generator and baseline_representation != "generator":
        generator_needed = any(
            bool(o.get("shared_error_recombination"))
            and float(o.get("max_remainder_width", 0.0)) > 0.0
            and float(o.get("max_polynomial_termwise_width", 0.0)) == 0.0
            for o in outputs
        )
        if generator_needed:
            out.append(("generator", int(baseline_order), "shared-error-recombination"))

    if policy.auto_try_higher_order and baseline_representation in {"polynomial", "generator"}:
        order_needed = any(
            bool(o.get("remainder_dominates_retained_variation"))
            and not (
                bool(o.get("shared_error_recombination"))
                and float(o.get("max_polynomial_termwise_width", 0.0)) == 0.0
            )
            for o in outputs
        )
        if order_needed:
            out.append((
                baseline_representation,
                int(baseline_order) + int(policy.auto_order_increment),
                "remainder-dominates-retained-variation",
            ))

    deduped = []
    seen = {(baseline_representation, int(baseline_order))}
    for rep, order, reason in out:
        key = (rep, int(order))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((rep, int(order), reason))
        if len(deduped) >= policy.max_attempts - 1:
            break
    return tuple(deduped)


__all__ = [
    "ConfigurationPolicy",
    "AutoIAConfiguredResult",
    "configuration_candidates",
    "automatic_escalation_candidates",
    "interval_result_dominates",
    "select_dominating_result",
    "rigorous_intersection_bounds",
    "structural_summary",
]
