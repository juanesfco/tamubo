"""Example 38 -- ordinary SciPy linprog with a certified optimal value.

Requires SciPy and AutoIA >= 1.0.4.dev8.
"""

import numpy as np
from scipy.optimize import linprog

from autoia import Interval, autoia


def engineering_model(p):
    x = p[0]

    # Ordinary SciPy code.  The LP objective depends on the uncertain input.
    result = linprog(
        c=[-1.0, -x],
        A_ub=[[1.0, 1.0]],
        b_ub=[1.0],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
    )
    return result.fun


if __name__ == "__main__":
    box = [Interval.enclosing("0.5", "2.0")]
    result = autoia(engineering_model, box)

    print(result)
    certs = [
        c for c in result.certificates
        if c.get("operation") == "scipy.optimize.linprog"
    ]
    cert = certs[-1]

    print("\nlinprog certificate")
    for key in (
        "semantic_mode",
        "optimal_value_lower",
        "optimal_value_upper",
        "certified_value_width",
        "selected_dual_y",
        "selected_primal_z",
        "primal_residual_upper",
        "primal_feasible_for_all_q",
        "machine_rigorous",
    ):
        print(f"  {key:30s} {cert[key]}")

    # Non-rigorous dense diagnostic only.
    xs = np.linspace(0.5, 2.0, 1001)
    vals = -np.maximum(1.0, xs)
    print("\ndense sampled value range (non-rigorous diagnostic)")
    print(f"  [{vals.min():.12g}, {vals.max():.12g}]")
