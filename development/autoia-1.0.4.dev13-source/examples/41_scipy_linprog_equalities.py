"""Example 41 -- ordinary scipy.optimize.linprog with A_eq / b_eq.

AutoIA 1.0.4.dev10 supports equality constraints for certified ``res.fun``.
The equality is reduced exactly to paired inequalities inside the semantic;
users keep ordinary SciPy syntax.
"""

from scipy.optimize import linprog

from autoia import Interval, autoia


def engineering_model(p):
    x = p[0]
    result = linprog(
        c=[-1.0, -x],
        A_eq=[[1.0, 1.0]],
        b_eq=[1.0],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
    )
    return result.fun


if __name__ == "__main__":
    r = autoia(
        engineering_model,
        [Interval.enclosing("0.5", "2.0")],
    )
    print(r)

    cert = [
        c for c in r.certificates
        if c.get("operation") == "scipy.optimize.linprog"
    ][-1]

    print("\nlinprog certificate")
    for key in (
        "certificate_completeness",
        "value_lower_certified",
        "value_upper_certified",
        "optimal_value_lower",
        "optimal_value_upper",
        "equality_constraint_count",
        "equality_semantic",
        "machine_rigorous",
    ):
        print(f"  {key:32s}: {cert.get(key)}")

    # Exact value is V(x)=-max(1,x), hence range [-2,-1].
    assert r.machine_rigorous
    assert r.lower[0] <= -2.0
    assert r.upper[0] >= -1.0
