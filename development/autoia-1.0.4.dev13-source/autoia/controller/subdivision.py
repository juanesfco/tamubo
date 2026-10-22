"""Fixed and adaptive certified subdivision controllers."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from ..trusted import Interval, RigorousBinary64Backend
from ..enclosures import PiecewiseEnclosure
from .policies import RangePolicy, SubdivisionPolicy
from .execution import autoia_single
from .branching import autoia_pathwise
from ..enclosures import BranchDecisionRequired
from .diagnostics import SubdivisionEvent
from .split_scoring import score_split_direction, cheap_score_key, cheap_score_accept

def _partition_interval(I, count):
    count = int(count)
    if count < 1:
        raise ValueError("Subdivision counts must be positive integers.")
    if count == 1:
        return [I]
    edges = []
    for k in range(count + 1):
        if k == 0:
            x = float(I.lo)
        elif k == count:
            x = float(I.hi)
        else:
            # Any binary64 cut is acceptable: adjacent closed boxes share it.
            t = k / count
            x = float((1.0 - t) * I.lo + t * I.hi)
            x = min(max(x, I.lo), I.hi)
        edges.append(x)
    # Enforce nondecreasing shared boundaries.
    for k in range(1, len(edges)):
        if edges[k] < edges[k-1]:
            edges[k] = edges[k-1]
    return [Interval(edges[k], edges[k+1]) for k in range(count)]

def subdivide_parameter_box(parameter_intervals, split_counts):
    intervals = list(parameter_intervals)
    if np.isscalar(split_counts):
        split_counts = [int(split_counts)] * len(intervals)
    split_counts = list(split_counts)
    if len(split_counts) != len(intervals):
        raise ValueError("split_counts must have one entry per parameter.")
    parts = [_partition_interval(I, n) for I, n in zip(intervals, split_counts)]
    return [list(box) for box in itertools.product(*parts)]

@dataclass
class AutoIAPiecewiseResult:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    branch_results: list
    backend_name: str
    machine_rigorous: bool
    requested_representation: str
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

    def __repr__(self):
        lines = [
            f"AutoIAPiecewiseResult(branches={len(self.branch_results)}, "
            f"requested_representation='{self.requested_representation}', "
            f"machine_rigorous={self.machine_rigorous})"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] piecewise hull in [{lo:.12g}, {hi:.12g}] "
                f"width={hi-lo:.12g}"
            )
        return "\n".join(lines)

def autoia_subdivide(model, parameter_intervals, split_counts,
                     order=4, range_method=None, backend=None,
                     representation="polynomial", internal_range_method=None,
                     final_range_method=None, range_policy=None):
    """
    Top-level generic subdivision prototype.

    Each subbox is independently certified with unchanged model source code.  The
    returned PiecewiseEnclosure retains the finite union; lower/upper are only its
    convenient interval hull.
    """
    backend = backend if backend is not None else RigorousBinary64Backend()
    boxes = subdivide_parameter_box(parameter_intervals, split_counts)
    branch_results = []
    for box in boxes:
        branch_results.append(autoia_pathwise(
            model, box, order=order, range_method=range_method,
            backend=backend, representation=representation,
            internal_range_method=internal_range_method,
            final_range_method=final_range_method,
            range_policy=range_policy,
        ))

    nout = len(branch_results[0].objects)
    objects = []
    lo = np.empty(nout)
    hi = np.empty(nout)
    for j in range(nout):
        branches = [r.objects[j] for r in branch_results]
        piece = PiecewiseEnclosure(branches, backend)
        objects.append(piece)
        # Use each branch's already-computed final policy range rather than
        # silently reverting to its cheap internal range method.
        lo[j] = min(r.lower[j] for r in branch_results)
        hi[j] = max(r.upper[j] for r in branch_results)

    return AutoIAPiecewiseResult(
        lo, hi, objects, branch_results,
        backend.name, backend.machine_rigorous,
        representation, branch_results[0].output_shape,
    )

@dataclass
class _AdaptiveBranch:
    box: list
    result: object
    depth: int
    branch_id: int

def _branch_controller_metrics(result):
    """Aggregate final-range diagnostics into branch-level controller metrics."""
    unresolved_union = set()
    worst_reduced_cost = 0
    full_cost = 0
    capped_outputs = []
    output_unresolved = []

    for j, meta in enumerate(result.final_range_diagnostics):
        unresolved = tuple(int(i) for i in meta.get("unresolved_variables", ()))
        unresolved_union.update(unresolved)
        output_unresolved.append(unresolved)
        rcost = int(meta.get("estimated_reduced_bernstein_coefficients", 0) or 0)
        fcost = int(meta.get("estimated_bernstein_coefficients", 0) or 0)
        worst_reduced_cost = max(worst_reduced_cost, rcost)
        full_cost = max(full_cost, fcost)
        if meta.get("bernstein_reason") == "reduced-cost-cap":
            capped_outputs.append(j)

    widths = np.asarray(result.width, dtype=float)
    total_width = float(np.sum(np.maximum(widths, 0.0)))
    max_width = float(np.max(np.maximum(widths, 0.0))) if len(widths) else 0.0

    return {
        "unresolved_variables": tuple(sorted(unresolved_union)),
        "num_unresolved": len(unresolved_union),
        "worst_reduced_bernstein_cost": int(worst_reduced_cost),
        "worst_full_bernstein_cost": int(full_cost),
        "capped_outputs": tuple(capped_outputs),
        "needs_subdivision": bool(capped_outputs and unresolved_union),
        "total_width": total_width,
        "max_width": max_width,
        "output_unresolved": tuple(output_unresolved),
    }

def _direction_ambiguity_scores(result, box):
    """Cheap pre-ranking of unresolved directions from derivative uncertainty.

    For an unresolved derivative interval D_i=[l_i,u_i], the factor

        width(D_i) * halfwidth(q_i)

    has output units and estimates how much unresolved directional variation can
    be associated with that coordinate.  Scores are normalized by each output's
    current width before being summed across outputs.
    """
    m = len(box)
    scores = np.zeros(m, dtype=float)
    for j, meta in enumerate(result.final_range_diagnostics):
        mm = meta.get("monotonicity_meta", {}) or {}
        signs = tuple(int(s) for s in mm.get("signs", ()))
        dranges = tuple(mm.get("derivative_ranges", ()))
        scale = max(float(result.width[j]), np.finfo(float).tiny)
        obj = result.objects[j]
        rep = getattr(obj, "rep", None)
        h = getattr(rep, "h", None)
        if h is None:
            continue
        for i in range(min(m, len(signs), len(dranges))):
            if signs[i] != 0:
                continue
            D = dranges[i]
            span = max(0.0, float(D.hi - D.lo))
            scores[i] += span * float(h[i]) / scale
    return scores

def _split_one_box(box, direction):
    left_right = _partition_interval(box[direction], 2)
    if len(left_right) != 2:
        raise RuntimeError("Binary split did not produce two subintervals.")
    out = []
    for part in left_right:
        child = list(box)
        child[direction] = part
        out.append(child)
    return out

def _combined_child_hull_metrics(child_results):
    if not child_results:
        raise ValueError("At least one child result is required.")
    lower = np.min(np.vstack([r.lower for r in child_results]), axis=0)
    upper = np.max(np.vstack([r.upper for r in child_results]), axis=0)
    widths = np.maximum(upper - lower, 0.0)
    metrics = [_branch_controller_metrics(r) for r in child_results]
    unresolved_union = set()
    for M in metrics:
        unresolved_union.update(M["unresolved_variables"])
    return {
        "lower": lower,
        "upper": upper,
        "width": widths,
        "total_width": float(np.sum(widths)),
        "max_width": float(np.max(widths)) if len(widths) else 0.0,
        "num_unresolved": len(unresolved_union),
        "unresolved_variables": tuple(sorted(unresolved_union)),
        "worst_reduced_bernstein_cost": max(
            M["worst_reduced_bernstein_cost"] for M in metrics
        ),
        "needs_subdivision": any(M["needs_subdivision"] for M in metrics),
    }

def _adaptive_single_autoia(*args, **kwargs):
    try:
        return autoia_single(*args, **kwargs)
    except BranchDecisionRequired as exc:
        raise NotImplementedError(
            "autoia_adaptive_subdivide does not yet combine the structural range "
            "controller with uncertain Python control flow. Use autoia(...) for "
            "path-aware execution or autoia_subdivide(...) for a prescribed "
            "geometric subdivision."
        ) from exc


def _evaluate_subdivision_candidate(model, branch, direction, *, order, backend,
                                    representation, range_policy):
    child_boxes = _split_one_box(branch.box, direction)
    child_results = [
        _adaptive_single_autoia(
            model, child_box, order=order, backend=backend,
            representation=representation, range_policy=range_policy,
        )
        for child_box in child_boxes
    ]
    combined = _combined_child_hull_metrics(child_results)
    parent = _branch_controller_metrics(branch.result)

    parent_cost = max(1, parent["worst_reduced_bernstein_cost"])
    child_cost = combined["worst_reduced_bernstein_cost"]
    structural_gain = max(0.0, (parent_cost - child_cost) / parent_cost)

    parent_width = max(parent["total_width"], np.finfo(float).tiny)
    width_gain = max(0.0, (parent["total_width"] - combined["total_width"]) / parent_width)

    unresolved_gain = max(0, parent["num_unresolved"] - combined["num_unresolved"])

    return {
        "direction": int(direction),
        "child_boxes": child_boxes,
        "child_results": child_results,
        "combined": combined,
        "structural_gain": float(structural_gain),
        "relative_width_gain": float(width_gain),
        "unresolved_gain": int(unresolved_gain),
        "parent_cost": int(parent["worst_reduced_bernstein_cost"]),
        "child_cost": int(child_cost),
        "parent_unresolved": int(parent["num_unresolved"]),
        "child_unresolved": int(combined["num_unresolved"]),
    }

@dataclass
class AutoIAAdaptiveSubdivisionResult:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    branch_results: list
    branch_boxes: list
    branch_depths: list
    backend_name: str
    machine_rigorous: bool
    requested_representation: str
    range_policy: RangePolicy
    subdivision_policy: SubdivisionPolicy
    subdivision_events: list[SubdivisionEvent]
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

    def __repr__(self):
        lines = [
            f"AutoIAAdaptiveSubdivisionResult(branches={len(self.branch_results)}, "
            f"requested_representation='{self.requested_representation}', "
            f"machine_rigorous={self.machine_rigorous})"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] piecewise hull in [{lo:.12g}, {hi:.12g}] "
                f"width={hi-lo:.12g}"
            )
        lines.append(f"  subdivision_events={len(self.subdivision_events)}")
        return "\n".join(lines)

def autoia_adaptive_subdivide(model, parameter_intervals, order=4, *, backend=None,
                               representation="polynomial", range_policy=None,
                               subdivision_policy=None):
    """Targeted top-level adaptive subdivision.

    ``SubdivisionPolicy.split_method`` chooses only the *strategy* for selecting
    a split.  The default ``"cheap"`` method scores hypothetical half-boxes from
    the already certified parent polynomial and then certifies only the selected
    children.  ``"lookahead"`` retains the historical reference controller that
    certifies every probed candidate direction before choosing among them.

    Both methods are equally sound: every retained child is independently
    certified by the ordinary AutoIA execution path.  Split scoring can affect
    runtime and sharpness, never validity of an accepted enclosure.
    """
    backend = backend if backend is not None else RigorousBinary64Backend()
    range_policy = range_policy if range_policy is not None else RangePolicy()
    subdivision_policy = (
        subdivision_policy if subdivision_policy is not None else SubdivisionPolicy()
    )
    intervals = list(parameter_intervals)

    root_result = _adaptive_single_autoia(
        model, intervals, order=order, backend=backend,
        representation=representation, range_policy=range_policy,
    )
    branches = [_AdaptiveBranch(intervals, root_result, 0, 0)]
    next_branch_id = 1
    events = []

    while len(branches) < subdivision_policy.max_branches:
        eligible = []
        for idx, branch in enumerate(branches):
            M = _branch_controller_metrics(branch.result)
            if M["needs_subdivision"] and branch.depth < subdivision_policy.max_depth:
                eligible.append((idx, branch, M))
        if not eligible:
            break

        # Work first on the branch with the largest unresolved Bernstein burden;
        # break ties by total output width.  This branch-selection logic is shared
        # by the cheap and reference split-direction policies.
        eligible.sort(
            key=lambda item: (
                item[2]["worst_reduced_bernstein_cost"],
                item[2]["total_width"],
            ),
            reverse=True,
        )
        branch_index, branch, parent_metrics = eligible[0]

        unresolved = list(parent_metrics["unresolved_variables"])
        ambiguity = _direction_ambiguity_scores(branch.result, branch.box)
        unresolved.sort(key=lambda i: ambiguity[i], reverse=True)
        candidates = unresolved[:subdivision_policy.max_probe_directions]
        if not candidates:
            break

        if subdivision_policy.split_method == "lookahead":
            # Historical Version-0.11 reference policy: fully certify every
            # probed pair of child boxes and choose using observed gains.
            probe_records = []
            for direction in candidates:
                try:
                    probe = _evaluate_subdivision_candidate(
                        model, branch, direction, order=order, backend=backend,
                        representation=representation, range_policy=range_policy,
                    )
                    probe["ambiguity_score"] = float(ambiguity[direction])
                    probe_records.append(probe)
                except Exception as exc:
                    probe_records.append({
                        "direction": int(direction),
                        "failed": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "ambiguity_score": float(ambiguity[direction]),
                    })

            successful = [p for p in probe_records if not p.get("failed", False)]
            if not successful:
                events.append({
                    "branch_id": branch.branch_id,
                    "depth": branch.depth,
                    "action": "stop-no-successful-candidate",
                    "parent_metrics": parent_metrics,
                    "probes": probe_records,
                })
                break

            best = max(
                successful,
                key=lambda p: (
                    p["unresolved_gain"],
                    p["structural_gain"],
                    p["relative_width_gain"],
                    p["ambiguity_score"],
                ),
            )
            accept = (
                best["unresolved_gain"] > 0
                or best["structural_gain"] >= subdivision_policy.min_structural_gain
                or best["relative_width_gain"] >= subdivision_policy.min_relative_width_gain
            )
            event = {
                "branch_id": branch.branch_id,
                "depth": branch.depth,
                "action": "split" if accept else "stop-insufficient-gain",
                "selected_direction": int(best["direction"]),
                "parent_metrics": parent_metrics,
                "selected_metrics": {
                    k: best[k] for k in (
                        "structural_gain", "relative_width_gain", "unresolved_gain",
                        "parent_cost", "child_cost", "parent_unresolved",
                        "child_unresolved", "ambiguity_score"
                    )
                },
                "probes": [
                    {k: v for k, v in p.items()
                     if k not in {"child_boxes", "child_results", "combined"}}
                    for p in probe_records
                ],
            }
            events.append(event)
            if not accept:
                break

            selected_child_boxes = best["child_boxes"]
            selected_child_results = best["child_results"]

        else:
            # Release-oriented policy: score all candidate directions from the
            # existing parent polynomial.  No user-model execution occurs here.
            score_records = []
            for direction in candidates:
                try:
                    score_records.append(score_split_direction(
                        branch.result, branch.box, direction,
                        ambiguity_score=float(ambiguity[direction]),
                    ))
                except Exception as exc:
                    score_records.append({
                        "direction": int(direction),
                        "failed": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "ambiguity_score": float(ambiguity[direction]),
                    })

            successful = [p for p in score_records if not p.get("failed", False)]
            if not successful:
                events.append({
                    "branch_id": branch.branch_id,
                    "depth": branch.depth,
                    "action": "stop-no-successful-candidate",
                    "split_method": "cheap",
                    "parent_metrics": parent_metrics,
                    "probes": score_records,
                })
                break

            best_score = max(successful, key=cheap_score_key)
            accept = cheap_score_accept(best_score, subdivision_policy)
            if not accept:
                events.append({
                    "branch_id": branch.branch_id,
                    "depth": branch.depth,
                    "action": "stop-insufficient-gain",
                    "split_method": "cheap",
                    "selected_direction": int(best_score["direction"]),
                    "parent_metrics": parent_metrics,
                    "selected_metrics": dict(best_score),
                    "probes": score_records,
                })
                break

            # Only now execute the selected pair of children.  This is the key
            # cost reduction versus reference lookahead.
            try:
                realized = _evaluate_subdivision_candidate(
                    model, branch, int(best_score["direction"]), order=order,
                    backend=backend, representation=representation,
                    range_policy=range_policy,
                )
            except Exception as exc:
                events.append({
                    "branch_id": branch.branch_id,
                    "depth": branch.depth,
                    "action": "stop-selected-candidate-failed",
                    "split_method": "cheap",
                    "selected_direction": int(best_score["direction"]),
                    "parent_metrics": parent_metrics,
                    "selected_metrics": dict(best_score),
                    "error": f"{type(exc).__name__}: {exc}",
                    "probes": score_records,
                })
                break

            realized_accept = (
                realized["unresolved_gain"] > 0
                or realized["structural_gain"] >= subdivision_policy.min_structural_gain
                or realized["relative_width_gain"] >= subdivision_policy.min_relative_width_gain
            )
            compact_best = {k: v for k, v in best_score.items() if k != "outputs"}
            compact_probes = [
                {k: v for k, v in rec.items() if k != "outputs"}
                for rec in score_records
            ]
            event = {
                "branch_id": branch.branch_id,
                "depth": branch.depth,
                "action": "split" if realized_accept else "stop-realized-insufficient-gain",
                "split_method": "cheap",
                "selected_direction": int(best_score["direction"]),
                "parent_metrics": parent_metrics,
                "selected_metrics": {
                    **compact_best,
                    "realized_unresolved_gain": int(realized["unresolved_gain"]),
                    "realized_structural_gain": float(realized["structural_gain"]),
                    "realized_relative_width_gain": float(realized["relative_width_gain"]),
                    "realized_child_cost": int(realized["child_cost"]),
                    "realized_child_unresolved": int(realized["child_unresolved"]),
                },
                "probes": compact_probes,
            }
            events.append(event)
            if not realized_accept:
                break
            selected_child_boxes = realized["child_boxes"]
            selected_child_results = realized["child_results"]

        if len(branches) + 1 > subdivision_policy.max_branches:
            break

        children = []
        for child_box, child_result in zip(selected_child_boxes, selected_child_results):
            children.append(_AdaptiveBranch(
                child_box, child_result, branch.depth + 1, next_branch_id
            ))
            next_branch_id += 1
        branches[branch_index:branch_index + 1] = children

    branch_results = [b.result for b in branches]
    branch_boxes = [b.box for b in branches]
    branch_depths = [b.depth for b in branches]
    if not branch_results:
        raise RuntimeError("Adaptive subdivision produced no branches.")

    nout = len(branch_results[0].objects)
    lower = np.empty(nout)
    upper = np.empty(nout)
    objects = []
    for j in range(nout):
        lower[j] = min(r.lower[j] for r in branch_results)
        upper[j] = max(r.upper[j] for r in branch_results)
        objects.append(PiecewiseEnclosure([r.objects[j] for r in branch_results], backend))

    return AutoIAAdaptiveSubdivisionResult(
        lower, upper, objects, branch_results, branch_boxes, branch_depths,
        backend.name, backend.machine_rigorous, representation,
        range_policy, subdivision_policy, events, branch_results[0].output_shape,
    )

# Stable internal aliases used by migration tests.
PiecewiseResult = AutoIAPiecewiseResult
AdaptiveSubdivisionResult = AutoIAAdaptiveSubdivisionResult

__all__ = [
    "subdivide_parameter_box",
    "AutoIAPiecewiseResult", "PiecewiseResult", "autoia_subdivide",
    "AutoIAAdaptiveSubdivisionResult", "AdaptiveSubdivisionResult",
    "autoia_adaptive_subdivide",
]
