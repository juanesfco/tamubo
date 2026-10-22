"""Example 43 -- automatic McCormick/RLT + certified OBBT for parametric linprog.

The LP changes active basis at x=1:

    min  -z
    s.t. x*z <= 1
         z <= x
         0 <= z <= 10
         x in [0.5,2]

The exact value is V(x)=-min(1/x,x), so the exact range is [-1,-0.5].
A single fixed dual witness is necessarily loose across the basis switch.
AutoIA 1.0.4.dev13 automatically builds a lifted McCormick/RLT relaxation and,
when useful, performs two certified OBBT/rebuild rounds on the same physical box.
"""

from scipy.optimize import linprog

from autoia import Interval, autoia


def engineering_model(p):
    x = p[0]
    return linprog(
        [-1.0],
        A_ub=[[x], [1.0]],
        b_ub=[1.0, x],
        bounds=[(0.0, 10.0)],
    ).fun


if __name__ == "__main__":
    result = autoia(
        engineering_model,
        [Interval.enclosing("0.5", "2.0")],
    )
    print(result)

    cert = [
        c for c in result.certificates
        if c.get("operation") == "scipy.optimize.linprog"
    ][-1]

    print("\nlinprog lower-bound portfolio")
    for key in (
        "fixed_dual_lower_bound",
        "rlt_initial_lower",
        "rlt_final_lower",
        "rlt_obbt_rounds",
        "selected_lower_bound_method",
        "rlt_lifted_variable_count",
        "rlt_lifted_row_count",
        "rlt_monomial_count",
        "rlt_product_count",
        "machine_rigorous",
    ):
        print(f"  {key:34s}: {cert.get(key)}")

    print("\nexact value range (known analytically)")
    print("  [-1, -0.5]")

    assert result.machine_rigorous
    assert result.lower[0] <= -1.0
    assert result.upper[0] >= -0.5
    assert -1.063 < cert["rlt_final_lower"] < -1.062
    assert cert["selected_lower_bound_method"] == "lifted-mccormick-rlt"
