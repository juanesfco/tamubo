"""Final-output range controller.

The mathematical range methods live in :mod:`autoia.polynomial`.  This module
only decides which already-rigorous method to invoke under a cost policy.
"""

from __future__ import annotations

import numpy as np

from ..enclosures import interval_intersection
from ..polynomial import (
    estimate_monotonicity_reduced_bernstein_cost,
    range_monotonicity_reduced_bernstein_scalar,
    _degree_vector_and_bernstein_cost,
)
from .policies import RangePolicy


def adaptive_final_range(validated_value, policy: RangePolicy):
    """Return a rigorous final range using structure-aware cost escalation."""
    rep = validated_value.rep

    if getattr(rep, "representation_name", None) == "interval":
        r, meta = rep.range()
        return r, {
            "method": "interval",
            "adaptive": False,
            "bernstein_used": False,
        }

    if getattr(rep, "representation_name", None) == "generator-polynomial":
        # The auxiliary generator contribution is a constant symmetric interval,
        # so scalar final ranging is exactly the polynomial final range followed
        # by that generator/remainder Minkowski sum.  Reuse the established
        # polynomial controller rather than teaching every range method about
        # generator bookkeeping.
        from ..enclosures import ValidatedValue
        collapsed = ValidatedValue(rep.collapse_to_polynomial())
        r, meta = adaptive_final_range(collapsed, policy)
        return r, {**meta, "representation": "generator-polynomial",
                   "generator_count": len(rep.generators)}

    if getattr(rep, "representation_name", None) != "polynomial":
        return validated_value.range(policy.final_method)

    if policy.final_method != "adaptive":
        r, meta = validated_value.range(policy.final_method)
        return r, {
            **meta,
            "adaptive": False,
            "bernstein_used": policy.final_method in {"bernstein", "portfolio"},
        }

    rt, mt = validated_value.range("termwise")
    rm, mm = validated_value.range("monotonicity")
    base = interval_intersection(rt, rm)

    wt = max(0.0, rt.hi - rt.lo)
    wm = max(0.0, rm.hi - rm.lo)
    wb = max(0.0, base.hi - base.lo)
    scale = max(wb, np.finfo(float).tiny)
    relative_disagreement = abs(wt - wm) / scale

    signs = tuple(int(s) for s in mm.get("signs", ()))
    reduced_cost, unresolved, degrees = estimate_monotonicity_reduced_bernstein_cost(
        rep.coeff, rep.m, signs
    )
    _, full_cost = _degree_vector_and_bernstein_cost(rep.coeff, rep.m)
    active = sum(int(d) > 0 for d in degrees)
    unresolved_active = sum(1 for i in unresolved if int(degrees[i]) > 0)
    unresolved_fraction = unresolved_active / active if active > 0 else 0.0

    bernstein_used = False
    bernstein_reason = "skipped"
    rb = None
    mb = None

    if unresolved_active == 0:
        bernstein_reason = "monotonicity-complete"
    elif reduced_cost <= policy.bernstein_max_coefficients:
        bernstein_used = True
        bernstein_reason = "unresolved-directions-affordable"
    else:
        bernstein_reason = "reduced-cost-cap"

    final = base
    if bernstein_used:
        rb, mb = range_monotonicity_reduced_bernstein_scalar(
            rep.coeff, rep.h, rep.backend, monotonicity_meta=mm
        )
        rb = rep.backend.add(rb, rep.remainder)
        final = interval_intersection(final, rb)

    return final, {
        "method": "adaptive-v10",
        "adaptive": True,
        "termwise": rt,
        "monotonicity": rm,
        "cheap_intersection": base,
        "relative_disagreement": relative_disagreement,
        "degree_vector": degrees,
        "estimated_bernstein_coefficients": int(full_cost),
        "estimated_reduced_bernstein_coefficients": int(reduced_cost),
        "unresolved_variables": unresolved,
        "unresolved_fraction": float(unresolved_fraction),
        "bernstein_used": bernstein_used,
        "bernstein_reason": bernstein_reason,
        "bernstein": rb,
        "bernstein_meta": mb,
        "termwise_meta": mt,
        "monotonicity_meta": mm,
    }


__all__ = ["adaptive_final_range"]
