import numpy as np
import pytest

pytest.importorskip("scipy.optimize")
from scipy.optimize import linprog

from autoia import autoia, Interval


def _cert(result):
    return [c for c in result.certificates if c.get("operation") == "scipy.optimize.linprog"][-1]


def test_rlt_t3_active_set_switch_improves_lower_and_obbt_tightens_again():
    # T3:
    #   min -z
    #   s.t. x z <= 1, z <= x, 0 <= z <= 10, x in [0.5,2].
    # Exact V(x)=-min(1/x,x), so the global range is [-1,-0.5].
    # A single fixed dual cannot be tight through the basis switch at x=1.
    def model(p):
        x = p[0]
        return linprog(
            [-1.0],
            A_ub=[[x], [1.0]],
            b_ub=[1.0, x],
            bounds=[(0.0, 10.0)],
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    c = _cert(r)
    assert r.machine_rigorous
    assert c["fixed_dual_lower_bound"] < -1.99
    # Initial McCormick/RLT relaxation reproduces the hand-derived -1.75 gate.
    assert -1.751 < c["rlt_initial_lower"] < -1.749
    # Two certified OBBT/rebuild rounds yield the sequence ending near -1.0625.
    assert c["rlt_obbt_rounds"] == 2
    assert -1.063 < c["rlt_final_lower"] < -1.062
    assert c["selected_lower_bound_method"] == "lifted-mccormick-rlt"
    assert r.lower[0] <= -1.0 <= r.upper[0]
    assert r.lower[0] > -1.07
    assert r.upper[0] >= -0.5


def test_rlt_handles_nonlinear_polynomial_constraint_matrix_entry_soundly():
    # Polynomial coefficient in A(q):
    #   min -z,  x^2 z <= 1, 0<=z<=10, x in [0.5,2].
    # Exact z*=1/x^2 and the exact value range is [-4,-0.25].
    def model(p):
        x = p[0]
        return linprog(
            [-1.0], A_ub=[[x * x]], b_ub=[1.0], bounds=[(0.0, 10.0)]
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    c = _cert(r)
    assert r.machine_rigorous
    assert r.lower[0] <= -4.0
    assert r.upper[0] >= -0.25
    assert c["rlt_status"] == "used"
    assert c["rlt_monomial_count"] >= 1
    assert c["rlt_product_count"] >= 1
    assert c["rlt_lower_bound"] > c["fixed_dual_lower_bound"]

    # Non-rigorous dense diagnostic for regression coverage only.
    for x in np.linspace(0.5, 2.0, 401):
        v = -(1.0 / (x * x))
        assert r.lower[0] <= v <= r.upper[0]


def test_rlt_unavailable_with_infinite_factor_bound_falls_back_to_fixed_dual():
    # McCormick needs finite factor bounds, but the pre-existing weak-dual path
    # can still certify this LP because the reduced-cost sign is uniform.
    def model(p):
        x = p[0]
        return linprog(
            [-1.0], A_ub=[[x]], b_ub=[1.0], bounds=[(0.0, None)]
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    c = _cert(r)
    assert r.machine_rigorous
    assert r.lower[0] <= -2.0
    assert r.upper[0] >= -0.5
    assert c["selected_lower_bound_method"] == "fixed-dual"
    assert c["rlt_lower_bound"] is None
    assert "finite factor bounds" in c["rlt_reason"]


def test_rlt_does_not_trigger_obbt_when_initial_lift_does_not_beat_fixed_dual():
    # Here the fixed-dual theorem is already essentially exact.  RLT is allowed
    # to run once as an independent portfolio candidate, but the expensive OBBT
    # loop should be skipped automatically.
    def model(p):
        x = p[0]
        return linprog(
            [-1.0, -x],
            A_ub=[[1.0, 1.0]],
            b_ub=[1.0],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    c = _cert(r)
    assert r.machine_rigorous
    assert c["selected_lower_bound_method"] == "fixed-dual"
    assert c["rlt_obbt_rounds"] == 0
    assert c["rlt_initial_lower"] <= c["fixed_dual_lower_bound"] + 1e-12
