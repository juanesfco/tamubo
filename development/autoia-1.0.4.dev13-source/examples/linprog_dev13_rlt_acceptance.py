"""Standalone acceptance test for AutoIA 1.0.4.dev13 RLT/OBBT linprog bounds."""

import numpy as np
from scipy.optimize import linprog

import autoia
from autoia import Interval, autoia as run_autoia


def cert(result):
    return [c for c in result.certificates if c.get("operation") == "scipy.optimize.linprog"][-1]


print("AutoIA version:", autoia.__version__)
assert autoia.__version__ == "1.0.4.dev13"

print("\nTEST 1 -- active-set switch T3")
def t3(p):
    x = p[0]
    return linprog(
        [-1.0], A_ub=[[x], [1.0]], b_ub=[1.0, x], bounds=[(0.0, 10.0)]
    ).fun
r1 = run_autoia(t3, [Interval(0.5, 2.0)])
c1 = cert(r1)
print(r1)
for k in ("fixed_dual_lower_bound", "rlt_initial_lower", "rlt_final_lower",
          "rlt_obbt_rounds", "selected_lower_bound_method"):
    print(f"  {k:30s}: {c1.get(k)}")
assert c1["fixed_dual_lower_bound"] < -1.99
assert -1.751 < c1["rlt_initial_lower"] < -1.749
assert -1.063 < c1["rlt_final_lower"] < -1.062
assert c1["rlt_obbt_rounds"] == 2
assert r1.lower[0] <= -1.0 <= r1.upper[0]
print("PASS")

print("\nTEST 2 -- nonlinear polynomial A(q) entry")
def poly_A(p):
    x = p[0]
    return linprog([-1.0], A_ub=[[x*x]], b_ub=[1.0], bounds=[(0.0, 10.0)]).fun
r2 = run_autoia(poly_A, [Interval(0.5, 2.0)])
c2 = cert(r2)
print(r2)
print("  fixed dual:", c2["fixed_dual_lower_bound"])
print("  RLT final :", c2["rlt_final_lower"])
assert r2.lower[0] <= -4.0
assert r2.upper[0] >= -0.25
assert c2["rlt_lower_bound"] > c2["fixed_dual_lower_bound"]
for x in np.linspace(0.5, 2.0, 201):
    v = -1.0/(x*x)
    assert r2.lower[0] <= v <= r2.upper[0]
print("PASS")

print("\nTEST 3 -- infinite-factor fallback preserves existing theorem")
def unbounded_factor(p):
    x = p[0]
    return linprog([-1.0], A_ub=[[x]], b_ub=[1.0], bounds=[(0.0, None)]).fun
r3 = run_autoia(unbounded_factor, [Interval(0.5, 2.0)])
c3 = cert(r3)
print(r3)
print("  RLT reason:", c3["rlt_reason"])
assert c3["selected_lower_bound_method"] == "fixed-dual"
assert c3["rlt_lower_bound"] is None
assert "finite factor bounds" in c3["rlt_reason"]
print("PASS")

print("\nALL DEV13 RLT/OBBT ACCEPTANCE TESTS PASSED")
