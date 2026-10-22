"""Cheap structural split scoring for adaptive subdivision.

The scorer never executes the user model.  It re-centers the already certified
parent output polynomial onto each hypothetical half-box and predicts how much
termwise derivative ambiguity, monotonicity-reduced Bernstein burden, and cheap
output width would remain after the split.

These calculations are controller heuristics only.  They are intentionally
ordinary binary64 arithmetic and are *not* part of the proof kernel.  Once a
split is selected, each child is independently re-executed and certified by the
normal AutoIA machinery, so score error can affect cost/sharpness but not
soundness.
"""

from __future__ import annotations

from math import comb
import numpy as np


def _clean_float_poly(coeff):
    return {tuple(a): float(c) for a, c in dict(coeff).items() if float(c) != 0.0}


def translate_polynomial_one_direction(coeff, direction: int, shift: float):
    """Re-center ``P(q)`` under ``q_d = q'_d + shift`` using float arithmetic.

    Only one coordinate is shifted, so a term of degree ``a_d`` creates only
    ``a_d + 1`` child terms rather than a full multivariate binomial expansion.
    """
    direction = int(direction)
    shift = float(shift)
    out = {}
    for alpha, c in _clean_float_poly(coeff).items():
        a = int(alpha[direction])
        for b in range(a + 1):
            beta = list(alpha)
            beta[direction] = b
            beta = tuple(beta)
            term = float(c) * float(comb(a, b)) * (shift ** (a - b))
            out[beta] = out.get(beta, 0.0) + term
    return {a: float(c) for a, c in out.items() if float(c) != 0.0}


def _mul_interval_float(A, B):
    vals = (
        A[0] * B[0], A[0] * B[1],
        A[1] * B[0], A[1] * B[1],
    )
    if not all(np.isfinite(v) for v in vals):
        raise FloatingPointError("Nonfinite controller interval product.")
    return (float(min(vals)), float(max(vals)))


def _power_range_symmetric_float(h: float, power: int):
    power = int(power)
    if power == 0:
        return (1.0, 1.0)
    H = abs(float(h)) ** power
    if not np.isfinite(H):
        raise FloatingPointError("Nonfinite controller power bound.")
    return (0.0, H) if (power % 2 == 0) else (-H, H)


def _monomial_range_float(alpha, h, fixed=None):
    fixed = fixed or {}
    out = (1.0, 1.0)
    for i, power in enumerate(alpha):
        power = int(power)
        if i in fixed:
            value = float(fixed[i]) ** power
            factor = (value, value)
        else:
            factor = _power_range_symmetric_float(float(h[i]), power)
        out = _mul_interval_float(out, factor)
    return out


def polynomial_range_termwise_float(coeff, h, fixed=None):
    lo = 0.0
    hi = 0.0
    for alpha, c in _clean_float_poly(coeff).items():
        M = _monomial_range_float(alpha, h, fixed=fixed)
        vals = (float(c) * M[0], float(c) * M[1])
        if not all(np.isfinite(v) for v in vals):
            raise FloatingPointError("Nonfinite controller polynomial term.")
        lo += min(vals)
        hi += max(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise FloatingPointError("Nonfinite controller polynomial range.")
    return (float(lo), float(hi))


def derivative_range_termwise_float(coeff, variable: int, h):
    variable = int(variable)
    lo = 0.0
    hi = 0.0
    for alpha, c in _clean_float_poly(coeff).items():
        k = int(alpha[variable])
        if k == 0:
            continue
        beta = list(alpha)
        beta[variable] -= 1
        M = _monomial_range_float(tuple(beta), h)
        ck = float(c) * float(k)
        vals = (ck * M[0], ck * M[1])
        if not all(np.isfinite(v) for v in vals):
            raise FloatingPointError("Nonfinite controller derivative term.")
        lo += min(vals)
        hi += max(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise FloatingPointError("Nonfinite controller derivative range.")
    return (float(lo), float(hi))


def _sign_from_float_interval(I):
    lo, hi = float(I[0]), float(I[1])
    if lo >= 0.0:
        return +1
    if hi <= 0.0:
        return -1
    return 0


def monotonicity_range_float(coeff, h, inherited_signs=None):
    """Cheap analogue of AutoIA monotonicity ranging on one hypothetical child."""
    m = len(h)
    inherited_signs = tuple(inherited_signs or (0,) * m)
    signs = []
    derivative_ranges = []
    lower_fixed = {}
    upper_fixed = {}

    for i in range(m):
        if i < len(inherited_signs) and int(inherited_signs[i]) != 0:
            sign = int(inherited_signs[i])
            # Parent proof remains valid on every child; no need to re-estimate.
            dr = (0.0, 0.0)
        else:
            dr = derivative_range_termwise_float(coeff, i, h)
            sign = _sign_from_float_interval(dr)
        derivative_ranges.append(dr)
        signs.append(sign)
        if sign > 0:
            lower_fixed[i] = -float(h[i])
            upper_fixed[i] = +float(h[i])
        elif sign < 0:
            lower_fixed[i] = +float(h[i])
            upper_fixed[i] = -float(h[i])

    rt = polynomial_range_termwise_float(coeff, h)
    rlo = polynomial_range_termwise_float(coeff, h, fixed=lower_fixed)
    rhi = polynomial_range_termwise_float(coeff, h, fixed=upper_fixed)
    mono = (float(rlo[0]), float(rhi[1]))

    # The true controller intersects the two sound bounds.  This is only a
    # heuristic, but mimic that geometry when finite float arithmetic agrees.
    lo = max(rt[0], mono[0])
    hi = min(rt[1], mono[1])
    if hi < lo:
        lo, hi = min(rt[0], mono[0]), max(rt[1], mono[1])
    return (float(lo), float(hi)), tuple(signs), tuple(derivative_ranges)


def _degree_vector(coeff, m):
    degrees = [0] * int(m)
    for alpha in coeff:
        for i, power in enumerate(alpha):
            degrees[i] = max(degrees[i], int(power))
    return tuple(degrees)


def reduced_bernstein_cost_from_signs(degrees, signs):
    degrees = tuple(int(d) for d in degrees)
    signs = tuple(int(s) for s in signs)
    active = tuple(i for i, d in enumerate(degrees) if d > 0)
    unresolved = tuple(i for i in active if signs[i] == 0)
    if not unresolved:
        return 0
    full = 1
    for d in degrees:
        full *= d + 1
    if len(unresolved) == len(active):
        return int(full)
    one_side = 1
    for i in unresolved:
        one_side *= degrees[i] + 1
    return int(2 * one_side)


def _collapse_rep_for_scoring(obj):
    rep = getattr(obj, "rep", obj)
    if getattr(rep, "representation_name", None) == "generator-polynomial":
        rep = rep.collapse_to_polynomial()
    if getattr(rep, "representation_name", None) != "polynomial":
        return None
    return rep


def _remainder_bounds(rep):
    R = getattr(rep, "remainder", None)
    if R is None:
        return (0.0, 0.0)
    return (float(R.lo), float(R.hi))


def score_split_direction(result, box, direction: int, ambiguity_score: float = 0.0):
    """Predict structural benefit of bisecting one unresolved input direction.

    The score is intentionally coefficient-free and lexicographic.  It reports
    counts/cost ratios that have direct controller meanings rather than a learned
    weighted scalar objective.
    """
    direction = int(direction)
    m = len(box)
    output_records = []
    child_unresolved_union = set()
    parent_unresolved_union = set()
    parent_incidence = 0
    child_incidence = 0
    parent_total_cost = 0
    child_total_cost = 0
    parent_worst_cost = 0
    child_worst_cost = 0
    predicted_lower = []
    predicted_upper = []

    for j, meta in enumerate(result.final_range_diagnostics):
        obj = result.objects[j]
        rep = _collapse_rep_for_scoring(obj)
        if rep is None:
            continue
        coeff = rep.coeff
        h = np.asarray(rep.h, dtype=float)
        if direction >= len(h) or h[direction] <= 0.0:
            continue

        mm = meta.get("monotonicity_meta", {}) or {}
        parent_signs = tuple(int(s) for s in mm.get("signs", (0,) * m))
        if len(parent_signs) < m:
            parent_signs = parent_signs + (0,) * (m - len(parent_signs))
        parent_unresolved = tuple(
            int(i) for i in meta.get("unresolved_variables", ())
        )
        parent_unresolved_union.update(parent_unresolved)
        parent_incidence += 2 * len(parent_unresolved)

        degrees = tuple(int(d) for d in meta.get("degree_vector", ()))
        if len(degrees) != m:
            degrees = _degree_vector(coeff, m)
        parent_cost = int(meta.get("estimated_reduced_bernstein_coefficients", 0) or 0)
        parent_worst_cost = max(parent_worst_cost, parent_cost)
        parent_total_cost += 2 * parent_cost

        child_ranges = []
        child_signs = []
        child_costs = []
        child_unresolved = []
        new_h = h.copy()
        new_h[direction] = 0.5 * float(h[direction])
        for side in (-1.0, +1.0):
            shift = side * 0.5 * float(h[direction])
            child_coeff = translate_polynomial_one_direction(
                coeff, direction, shift
            )
            crange, signs, _ = monotonicity_range_float(
                child_coeff, new_h, inherited_signs=parent_signs
            )
            unresolved = tuple(
                i for i, s in enumerate(signs)
                if int(s) == 0 and int(degrees[i]) > 0
            )
            cost = reduced_bernstein_cost_from_signs(degrees, signs)
            child_ranges.append(crange)
            child_signs.append(signs)
            child_costs.append(cost)
            child_unresolved.append(unresolved)
            child_unresolved_union.update(unresolved)
            child_incidence += len(unresolved)
            child_total_cost += cost
            child_worst_cost = max(child_worst_cost, cost)

        rem_lo, rem_hi = _remainder_bounds(rep)
        lo = min(r[0] + rem_lo for r in child_ranges)
        hi = max(r[1] + rem_hi for r in child_ranges)
        predicted_lower.append(lo)
        predicted_upper.append(hi)
        output_records.append({
            "output": int(j),
            "parent_unresolved": parent_unresolved,
            "child_unresolved": tuple(child_unresolved),
            "child_costs": tuple(int(c) for c in child_costs),
            "child_signs": tuple(child_signs),
            "predicted_hull": (float(lo), float(hi)),
        })

    parent_num_unresolved = len(parent_unresolved_union)
    child_num_unresolved = len(child_unresolved_union)
    unresolved_gain = max(0, parent_num_unresolved - child_num_unresolved)
    local_resolution_gain = max(0, parent_incidence - child_incidence)

    denom_worst = max(1, parent_worst_cost)
    structural_gain = max(
        0.0, (parent_worst_cost - child_worst_cost) / float(denom_worst)
    )
    denom_total = max(1, parent_total_cost)
    tree_cost_gain = max(
        0.0, (parent_total_cost - child_total_cost) / float(denom_total)
    )

    parent_width = float(np.sum(np.maximum(np.asarray(result.width, dtype=float), 0.0)))
    if predicted_lower:
        predicted_width = float(np.sum(
            np.maximum(np.asarray(predicted_upper) - np.asarray(predicted_lower), 0.0)
        ))
        relative_width_gain = max(
            0.0,
            (parent_width - predicted_width) / max(parent_width, np.finfo(float).tiny),
        )
    else:
        predicted_width = parent_width
        relative_width_gain = 0.0

    return {
        "direction": int(direction),
        "predicted_unresolved_gain": int(unresolved_gain),
        "predicted_local_resolution_gain": int(local_resolution_gain),
        "predicted_structural_gain": float(structural_gain),
        "predicted_tree_cost_gain": float(tree_cost_gain),
        "predicted_relative_width_gain": float(relative_width_gain),
        "parent_cost": int(parent_worst_cost),
        "predicted_child_cost": int(child_worst_cost),
        "parent_total_cost": int(parent_total_cost),
        "predicted_child_total_cost": int(child_total_cost),
        "parent_unresolved": int(parent_num_unresolved),
        "predicted_child_unresolved": int(child_num_unresolved),
        "ambiguity_score": float(ambiguity_score),
        "predicted_total_width": float(predicted_width),
        "outputs": output_records,
    }


def cheap_score_key(score):
    """Lexicographic generic score used to rank candidate directions."""
    return (
        int(score.get("predicted_unresolved_gain", 0)),
        float(score.get("predicted_structural_gain", 0.0)),
        float(score.get("predicted_relative_width_gain", 0.0)),
        int(score.get("predicted_local_resolution_gain", 0)),
        float(score.get("predicted_tree_cost_gain", 0.0)),
        float(score.get("ambiguity_score", 0.0)),
    )


def cheap_score_accept(score, policy):
    """Use existing policy thresholds plus exact structural progress signals."""
    return bool(
        int(score.get("predicted_unresolved_gain", 0)) > 0
        or int(score.get("predicted_local_resolution_gain", 0)) > 0
        or float(score.get("predicted_structural_gain", 0.0)) >= float(policy.min_structural_gain)
        or float(score.get("predicted_tree_cost_gain", 0.0)) >= float(policy.min_structural_gain)
        or float(score.get("predicted_relative_width_gain", 0.0)) >= float(policy.min_relative_width_gain)
    )


__all__ = [
    "translate_polynomial_one_direction",
    "polynomial_range_termwise_float",
    "derivative_range_termwise_float",
    "monotonicity_range_float",
    "reduced_bernstein_cost_from_signs",
    "score_split_direction",
    "cheap_score_key",
    "cheap_score_accept",
]
