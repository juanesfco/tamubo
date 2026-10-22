"""Path-aware certified execution for ordinary Python scalar conditionals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from ..trusted import Interval, RigorousBinary64Backend, ElementaryDomainError
from ..enclosures import (
    PiecewiseEnclosure,
    BranchDecisionRequired,
    InfeasiblePath,
    DomainRefinementRequired,
)
from ..polynomial import zero_alpha, basis_alpha, derivative_range_termwise
from .execution import autoia_single, resolve_range_policy
from .policies import BranchPolicy, RangePolicy


@dataclass
class _PathState:
    box: list[Interval]
    decisions: dict[int, bool]
    split_counts: dict[int, int] = field(default_factory=dict)
    branch_id: int = 0
    parent_id: int | None = None
    track_natural_domain: bool = False


@dataclass
class AutoIAPathwiseResult:
    """Finite union of independently certified execution paths."""

    lower: np.ndarray
    upper: np.ndarray
    objects: list
    branch_results: list
    branch_boxes: list
    branch_decisions: list
    branch_traces: list
    path_events: list
    backend_name: str
    machine_rigorous: bool
    requested_representation: str
    range_policy: RangePolicy
    branch_policy: BranchPolicy
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
        return ["piecewise"] * len(self.objects)

    @property
    def certificates(self):
        out = []
        for r in self.branch_results:
            out.extend(getattr(r, "certificates", ()))
        return out

    def __repr__(self):
        lines = [
            f"AutoIAPathwiseResult(branches={len(self.branch_results)}, "
            f"requested_representation={self.requested_representation!r}, "
            f"machine_rigorous={self.machine_rigorous})"
        ]
        for j, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{j}] piecewise hull in [{lo:.12g}, {hi:.12g}] "
                f"width={hi-lo:.12g}"
            )
        return "\n".join(lines)


def _interval_midpoint(I: Interval) -> float:
    return float(0.5 * float(I.lo) + 0.5 * float(I.hi))


def _exact_single_variable_root(predicate, box):
    """Return ``(dimension, root)`` for an exact affine one-variable predicate.

    Prefer the proof-only exact affine shadow in physical coordinates.  This
    survives harmless coefficient/remainder rounding that can otherwise obscure
    roots exactly on a closed-box endpoint.  The older zero-remainder polynomial
    recognizer remains as a fallback.
    """
    exact = predicate.exact_affine_root(box)
    if exact is not None:
        return exact
    diff = predicate.difference
    rep = getattr(diff, "rep", None)
    if getattr(rep, "representation_name", None) != "polynomial":
        return None
    R = getattr(rep, "remainder", None)
    if R is None or float(R.lo) != 0.0 or float(R.hi) != 0.0:
        return None
    m = len(rep.h)
    z = zero_alpha(m)
    c0 = float(rep.coeff.get(z, 0.0))
    active = []
    allowed = {z}
    for i in range(m):
        a = basis_alpha(m, i)
        g = float(rep.coeff.get(a, 0.0))
        if g != 0.0:
            active.append((i, g))
            allowed.add(a)
    for alpha, coeff in rep.coeff.items():
        if float(coeff) != 0.0 and alpha not in allowed:
            return None
    if len(active) != 1:
        return None
    i, g = active[0]
    qroot = -c0 / g
    center = _interval_midpoint(box[i])
    root = float(center + qroot)
    if not math.isfinite(root):
        return None
    if root < float(box[i].lo) or root > float(box[i].hi):
        return None
    return i, root


def _validated_value_split_direction(value, box):
    """Choose a geometric refinement direction from physical-q sensitivity.

    This is proposal/controller logic only.  Every child is independently
    re-executed and certified, so this heuristic can affect cost but not rigor.
    """
    rep = getattr(value, "rep", value)
    if getattr(rep, "representation_name", None) in {"polynomial", "generator-polynomial"}:
        scores = []
        for i in range(len(box)):
            try:
                D = derivative_range_termwise(rep.coeff, i, rep.h, rep.backend)
                mag = max(abs(float(D.lo)), abs(float(D.hi)))
                scores.append(mag * float(rep.h[i]))
            except Exception:
                scores.append(0.0)
        if scores and max(scores) > 0.0:
            return int(np.argmax(scores))
    widths = [max(0.0, float(I.hi) - float(I.lo)) for I in box]
    if not widths or max(widths) <= 0.0:
        return None
    return int(np.argmax(widths))


def _predicate_split_direction(predicate, box):
    return _validated_value_split_direction(predicate.difference, box)


def _split_box(box, direction, cut=None):
    I = box[direction]
    lo = float(I.lo)
    hi = float(I.hi)
    if cut is None:
        cut = _interval_midpoint(I)
    cut = float(cut)
    if not (lo < cut < hi):
        return None
    left = list(box)
    right = list(box)
    left[direction] = Interval(lo, cut)
    right[direction] = Interval(cut, hi)
    return left, right


def _equality_outcome(operator: str) -> bool:
    return operator in {"le", "ge", "eq"}


def _truth_for_nonzero_sign(operator: str, sign: int) -> bool:
    if sign not in {-1, +1}:
        raise ValueError("sign must be -1 or +1")
    if operator in {"lt", "le"}:
        return sign < 0
    if operator in {"gt", "ge"}:
        return sign > 0
    if operator == "eq":
        return False
    if operator == "ne":
        return True
    raise ValueError(operator)


def _exact_root_path_states(state, req, exact_root, next_branch_id):
    """Partition an exact affine one-variable predicate by its real root.

    Closed side boxes may include the equality point even when their forced path
    is the strict-side outcome.  That is sound because the branch formula is then
    certified over a *superset* of its feasible path domain.  If neither side's
    path outcome agrees with equality (``==``/``!=``), a degenerate root branch
    is added so the actual boundary semantics are retained.
    """
    direction, root = exact_root
    rep = req.predicate.difference.rep
    g = float(rep.coeff.get(basis_alpha(len(rep.h), direction), 0.0))
    if g == 0.0:
        return None, next_branch_id
    I = state.box[direction]
    lo = float(I.lo)
    hi = float(I.hi)
    eq_truth = _equality_outcome(req.predicate.operator)
    neg_truth = _truth_for_nonzero_sign(req.predicate.operator, -1 if g > 0.0 else +1)
    pos_truth = _truth_for_nonzero_sign(req.predicate.operator, +1 if g > 0.0 else -1)

    states = []
    outcomes_covering_boundary = set()
    if lo < root:
        left = list(state.box)
        left[direction] = Interval(lo, root)
        d = dict(state.decisions)
        d[req.occurrence_index] = neg_truth
        states.append(_PathState(left, d, dict(state.split_counts), next_branch_id, state.branch_id,
                               state.track_natural_domain))
        next_branch_id += 1
        outcomes_covering_boundary.add(neg_truth)
    if root < hi:
        right = list(state.box)
        right[direction] = Interval(root, hi)
        d = dict(state.decisions)
        d[req.occurrence_index] = pos_truth
        states.append(_PathState(right, d, dict(state.split_counts), next_branch_id, state.branch_id,
                               state.track_natural_domain))
        next_branch_id += 1
        outcomes_covering_boundary.add(pos_truth)

    if lo == hi == root:
        d = dict(state.decisions)
        d[req.occurrence_index] = eq_truth
        states.append(_PathState(list(state.box), d, dict(state.split_counts), next_branch_id, state.branch_id,
                               state.track_natural_domain))
        next_branch_id += 1
    elif eq_truth not in outcomes_covering_boundary:
        point = list(state.box)
        point[direction] = Interval(root, root)
        d = dict(state.decisions)
        d[req.occurrence_index] = eq_truth
        states.append(_PathState(point, d, dict(state.split_counts), next_branch_id, state.branch_id,
                               state.track_natural_domain))
        next_branch_id += 1

    return states, next_branch_id


def _one_sided_touch(interval):
    lo = float(interval.lo)
    hi = float(interval.hi)
    return (lo == 0.0 and hi > 0.0) or (hi == 0.0 and lo < 0.0)


def _make_piecewise_result(completed, backend, representation, range_policy,
                           branch_policy, events):
    if not completed:
        raise RuntimeError("Path-aware execution produced no feasible certified branches.")
    shapes = {tuple(r.output_shape) for _, r in completed}
    if len(shapes) != 1:
        raise ValueError("All feasible execution paths must return the same output shape.")
    nout = len(completed[0][1].objects)
    if any(len(r.objects) != nout for _, r in completed):
        raise ValueError("All feasible execution paths must return the same number of outputs.")

    lower = np.empty(nout)
    upper = np.empty(nout)
    objects = []
    metadata = []
    for state, result in completed:
        metadata.append({
            "box": tuple(state.box),
            "decisions": dict(state.decisions),
            "trace": tuple(getattr(result, "branch_trace", ())),
            "branch_id": state.branch_id,
        })
    for j in range(nout):
        lower[j] = min(r.lower[j] for _, r in completed)
        upper[j] = max(r.upper[j] for _, r in completed)
        branches = [r.objects[j] for _, r in completed]
        objects.append(PiecewiseEnclosure(branches, backend, branch_metadata=metadata))

    return AutoIAPathwiseResult(
        lower=lower,
        upper=upper,
        objects=objects,
        branch_results=[r for _, r in completed],
        branch_boxes=[list(state.box) for state, _ in completed],
        branch_decisions=[dict(state.decisions) for state, _ in completed],
        branch_traces=[tuple(getattr(r, "branch_trace", ())) for _, r in completed],
        path_events=list(events),
        backend_name=backend.name,
        machine_rigorous=backend.machine_rigorous,
        requested_representation=representation,
        range_policy=range_policy,
        branch_policy=branch_policy,
        output_shape=completed[0][1].output_shape,
    )


def autoia_pathwise(model, parameter_intervals, order=4, range_method=None,
                    backend=None, representation="polynomial",
                    internal_range_method=None, final_range_method=None,
                    range_policy=None, branch_policy=None, *, prepared_dispatch=None):
    """Certify ordinary scalar Python control flow by path-aware re-execution.

    A branchless model returns the ordinary :class:`AutoIAResult`.  If uncertain
    Boolean control flow is encountered, the returned object is an
    :class:`AutoIAPathwiseResult` retaining the finite union of certified paths.
    """
    backend = backend if backend is not None else RigorousBinary64Backend()
    range_policy = resolve_range_policy(
        range_method, internal_range_method, final_range_method, range_policy
    )
    branch_policy = BranchPolicy() if branch_policy is None else branch_policy

    initial_box = list(parameter_intervals)
    queue = [_PathState(initial_box, {}, {}, 0, None, False)]
    next_branch_id = 1
    total_splits = 0
    completed = []
    events = []
    encountered_ambiguity = False

    while queue:
        if len(queue) + len(completed) > branch_policy.max_branches:
            raise RuntimeError(
                "Path-aware branch budget exceeded max_branches; certification "
                "requires every feasible path to be retained."
            )
        state = queue.pop()
        try:
            result = autoia_single(
                model, state.box, order=order, backend=backend,
                representation=representation, range_policy=range_policy,
                branch_decisions=state.decisions,
                prepared_dispatch=prepared_dispatch,
                track_natural_domain=state.track_natural_domain,
            )
            completed.append((state, result))
            events.append({
                "action": "certified-path",
                "branch_id": state.branch_id,
                "decisions": dict(state.decisions),
                "box": tuple(state.box),
            })
            continue
        except InfeasiblePath as infeasible:
            events.append({
                "action": "prune-infeasible-path",
                "branch_id": state.branch_id,
                "occurrence_index": infeasible.occurrence_index,
                "forced": infeasible.forced,
                "rigorous": infeasible.rigorous,
            })
            continue
        except DomainRefinementRequired as domain_req:
            # The common fast path carries no natural interval shadow.  If a
            # structural domain proof is unresolved, replay this *same* physical
            # box once with trusted natural-interval propagation before spending
            # any geometric split budget.  This is a proof retry, not a branch.
            rep_name = getattr(getattr(domain_req.value, "rep", None), "representation_name", None)
            if (not state.track_natural_domain) and rep_name != "interval":
                queue.append(_PathState(
                    list(state.box), dict(state.decisions), dict(state.split_counts),
                    state.branch_id, state.parent_id, True,
                ))
                events.append({
                    "action": "replay-unresolved-domain-with-natural-shadow",
                    "branch_id": state.branch_id,
                    "occurrence_index": int(domain_req.occurrence_index),
                    "operation": domain_req.proof.requirement.operation,
                    "argument_range": (
                        float(domain_req.proof.interval.lo),
                        float(domain_req.proof.interval.hi),
                    ),
                    "methods_tried": tuple(domain_req.proof.methods_tried),
                })
                continue

            key = ("domain", int(domain_req.occurrence_index))
            split_count = int(state.split_counts.get(key, 0))
            can_split = (
                branch_policy.prefer_geometric_split
                and split_count < branch_policy.max_domain_split_depth_per_obligation
                and total_splits < branch_policy.max_total_splits
            )
            direction = (
                _validated_value_split_direction(domain_req.value, state.box)
                if can_split else None
            )
            children = None if direction is None else _split_box(state.box, direction)
            if children is not None:
                if len(queue) + len(completed) + 2 > branch_policy.max_branches:
                    raise RuntimeError(
                        "Path/domain refinement budget exceeded max_branches."
                    )
                new_counts = dict(state.split_counts)
                new_counts[key] = split_count + 1
                left, right = children
                queue.append(_PathState(
                    right, dict(state.decisions), dict(new_counts), next_branch_id,
                    state.branch_id, state.track_natural_domain,
                ))
                next_branch_id += 1
                queue.append(_PathState(
                    left, dict(state.decisions), dict(new_counts), next_branch_id,
                    state.branch_id, state.track_natural_domain,
                ))
                next_branch_id += 1
                total_splits += 1
                events.append({
                    "action": "split-unresolved-domain",
                    "branch_id": state.branch_id,
                    "occurrence_index": int(domain_req.occurrence_index),
                    "operation": domain_req.proof.requirement.operation,
                    "direction": int(direction),
                    "cut": float(children[0][direction].hi),
                    "argument_range": (
                        float(domain_req.proof.interval.lo),
                        float(domain_req.proof.interval.hi),
                    ),
                    "methods_tried": tuple(domain_req.proof.methods_tried),
                })
                continue

            proof = domain_req.proof
            message = (
                f"Cannot certify the real domain for {proof.requirement.operation}: "
                f"the universal predicate {proof.requirement.mathematical_text} "
                f"remains unresolved on rigorous argument range "
                f"[{proof.interval.lo}, {proof.interval.hi}] after the allowed "
                "geometric domain-refinement budget."
            )
            if proof.requirement.relation == "nonzero":
                # Preserve the established public failure taxonomy for a true or
                # unresolved zero-crossing denominator after refinement is exhausted.
                raise ZeroDivisionError(message)
            raise ElementaryDomainError(message)
        except BranchDecisionRequired as exc:
            encountered_ambiguity = True
            req = exc

        idx = req.occurrence_index
        if idx >= branch_policy.max_path_decisions:
            raise RuntimeError(
                "Path-aware branch budget exceeded max_path_decisions."
            )

        split_count = int(state.split_counts.get(idx, 0))
        exact_root = _exact_single_variable_root(req.predicate, state.box)

        # Exact one-variable affine predicates admit a much sharper partition:
        # split directly at the real comparison root and assign the correct path
        # outcome to each side.  This also avoids tiny centered-box roundoff from
        # making simple conditions such as ``x >= 0`` appear unresolved forever.
        if exact_root is not None:
            states, next_branch_id = _exact_root_path_states(
                state, req, exact_root, next_branch_id
            )
            if states:
                if len(queue) + len(completed) + len(states) > branch_policy.max_branches:
                    raise RuntimeError(
                        "Path-aware branch budget exceeded max_branches during exact root partition."
                    )
                queue.extend(reversed(states))
                events.append({
                    "action": "partition-exact-affine-predicate",
                    "branch_id": state.branch_id,
                    "occurrence_index": idx,
                    "direction": int(exact_root[0]),
                    "root": float(exact_root[1]),
                    "operator": req.predicate.operator,
                    "num_children": len(states),
                })
                continue

        can_split = (
            branch_policy.prefer_geometric_split
            and split_count < branch_policy.max_split_depth_per_decision
            and total_splits < branch_policy.max_total_splits
        )
        if can_split:
            direction = None
            cut = None
            if exact_root is not None:
                direction, cut = exact_root
            else:
                direction = _predicate_split_direction(req.predicate, state.box)
            children = None if direction is None else _split_box(state.box, direction, cut)
            if children is not None:
                new_counts = dict(state.split_counts)
                new_counts[idx] = split_count + 1
                left, right = children
                queue.append(_PathState(
                    right, dict(state.decisions), dict(new_counts), next_branch_id,
                    state.branch_id, state.track_natural_domain,
                ))
                next_branch_id += 1
                queue.append(_PathState(
                    left, dict(state.decisions), dict(new_counts), next_branch_id,
                    state.branch_id, state.track_natural_domain,
                ))
                next_branch_id += 1
                total_splits += 1
                events.append({
                    "action": "split-unresolved-predicate",
                    "branch_id": state.branch_id,
                    "occurrence_index": idx,
                    "direction": int(direction),
                    "cut": float(children[0][direction].hi),
                    "predicate_range": (float(req.interval.lo), float(req.interval.hi)),
                })
                continue

        # Finite box subdivision cannot in general resolve a hypersurface exactly.
        # Enumerating both Boolean outcomes is sound because each branch formula
        # is certified over a superset (the current box) of the inputs that can
        # actually follow that path.  Infeasible forced paths are pruned on rerun.
        d_false = dict(state.decisions)
        d_false[idx] = False
        d_true = dict(state.decisions)
        d_true[idx] = True
        queue.append(_PathState(
            list(state.box), d_false, dict(state.split_counts), next_branch_id,
            state.branch_id, state.track_natural_domain,
        ))
        next_branch_id += 1
        queue.append(_PathState(
            list(state.box), d_true, dict(state.split_counts), next_branch_id,
            state.branch_id, state.track_natural_domain,
        ))
        next_branch_id += 1
        events.append({
            "action": "enumerate-predicate-paths",
            "branch_id": state.branch_id,
            "occurrence_index": idx,
            "operator": req.predicate.operator,
            "predicate_range": (float(req.interval.lo), float(req.interval.hi)),
        })

    if not encountered_ambiguity and len(completed) == 1:
        return completed[0][1]
    return _make_piecewise_result(
        completed, backend, representation, range_policy, branch_policy, events
    )


__all__ = ["AutoIAPathwiseResult", "autoia_pathwise"]
