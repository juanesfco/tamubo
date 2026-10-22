"""05 — Prescribed and adaptive subdivision.

Subdivision is a proof strategy for a broad physical box.  This example keeps
its policy explicit because the purpose is to compare subdivision strategies,
not to demonstrate the zero-knob top-level call.
"""
import numpy as np
from autoia import (
    Interval,
    autoia,
    autoia_subdivide,
    autoia_adaptive_subdivide,
    SubdivisionPolicy,
)


def model(p):
    x = p[0]
    return np.exp(-8.0 * x * x) + 0.2 * x


if __name__ == "__main__":
    box = [Interval(-1.0, 1.0)]
    whole = autoia(model, box)
    fixed = autoia_subdivide(model, box, split_counts=[4])
    adaptive = autoia_adaptive_subdivide(
        model,
        box,
        subdivision_policy=SubdivisionPolicy(max_branches=8, max_depth=4),
    )
    print("whole-box:", whole.lower, whole.upper)
    print(
        "fixed 4-way subdivision:",
        fixed.lower,
        fixed.upper,
        "branches=",
        len(fixed.branch_results),
    )
    print(
        "adaptive:",
        adaptive.lower,
        adaptive.upper,
        "branches=",
        len(adaptive.branch_results),
    )
