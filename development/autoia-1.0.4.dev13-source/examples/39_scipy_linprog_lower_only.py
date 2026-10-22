"""
Example 39 -- one-sided certified scipy.optimize.linprog value.

This example is deliberately chosen so the LP is feasible for every x in the
AutoIA box, but its only feasible point moves with x. Consequently no single
constant primal witness can be feasible for the whole box.

AutoIA 1.0.4.dev9 therefore preserves the independent weak-duality lower
certificate and returns [L,+inf] rather than raising.
"""

import numpy as np
from scipy.optimize import linprog

from autoia import Interval, autoia


def engineering_model(p):
    x = p[0]

    # min z
    # s.t. z <= x
    #     -z <= -x
    #      0 <= z <= 1
    #
    # Thus z=x for every realization and the exact value is V(x)=x.
    result = linprog(
        c=[1.0],
        A_ub=[[1.0], [-1.0]],
        b_ub=[x, -x],
        bounds=[(0.0, 1.0)],
    )
    return result.fun


if __name__ == "__main__":
    box = [Interval.enclosing("0.0", "1.0")]
    result = autoia(engineering_model, box)

    print(result)

    certs = [
        c for c in result.certificates
        if c.get("operation") == "scipy.optimize.linprog"
    ]
    cert = certs[-1]

    print("\nlinprog certificate")
    for key in (
        "certificate_completeness",
        "value_lower_certified",
        "value_upper_certified",
        "optimal_value_lower",
        "optimal_value_upper",
        "universal_primal_witness_found",
        "verified_primal_candidate_count",
        "machine_rigorous",
    ):
        print(f"  {key:34s} {cert[key]}")

    # Exact mathematics: V(x)=x for x in [0,1].
    print("\nexact value range (known analytically)")
    print("  [0, 1]")
    print("\nAutoIA intentionally returns only the side it proved with the current")
    print("fixed-primal-witness semantic; +inf means no finite upper certificate")
    print("was established, not that the true value is unbounded.")

    assert result.machine_rigorous
    assert float(result.lower[0]) <= 0.0
    assert np.isposinf(float(result.upper[0]))
    assert cert["value_lower_certified"] is True
    assert cert["value_upper_certified"] is False
