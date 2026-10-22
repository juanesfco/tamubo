import numpy as np
import pytest

scipy_opt = pytest.importorskip("scipy.optimize")
import scipy.optimize as opt
from scipy.optimize import linprog

from autoia import autoia, Interval
from autoia.exceptions import UnsupportedOperationError, CertificationError


def _linprog_certs(result):
    return [c for c in result.certificates if c.get("operation") == "scipy.optimize.linprog"]


def test_linprog_constant_data_is_independently_certified():
    def model(p):
        _ = p[0]
        return linprog(
            [-1.0, -2.0],
            A_ub=[[1.0, 1.0]],
            b_ub=[1.0],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        ).fun

    r = autoia(model, [Interval(0.0, 1.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= -2.0 <= r.upper[0]
    assert r.width[0] < 1e-10
    certs = _linprog_certs(r)
    assert certs
    cert = certs[-1]
    assert cert["semantic_mode"] == "certified-optimal-value"
    assert cert["primal_feasible_for_all_q"] is True
    assert cert["certifies"] == "res.fun only"


def true_v2(x):
    return -max(1.0, x)


def test_linprog_uncertain_objective_contains_dense_true_values():
    def model(p):
        x = p[0]
        return linprog(
            [-1.0, -x],
            A_ub=[[1.0, 1.0]],
            b_ub=[1.0],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    assert r.machine_rigorous
    for q in np.linspace(0.5, 2.0, 1001):
        v = true_v2(float(q))
        assert r.lower[0] <= v <= r.upper[0]
    assert r.width[0] < 2.0
    assert _linprog_certs(r)


def test_linprog_dependency_coupling_gate_stable_dual():
    # The same x occurs in c, A, and b:
    #
    #   min  -x*z    s.t. x*z <= x,  0 <= z <= 10,   x in [0.5,2].
    #
    # Because x>0, the exact optimizer is z=1 and V(x)=-x.  The nominal dual
    # y=-1 gives d=c-A^T y=-x+x=0 *only if shared dependency survives*.
    # Treating the three x occurrences as independent intervals destroys this
    # cancellation and gives a much weaker safe lower bound.
    def model(p):
        x = p[0]
        return linprog(
            [-x],
            A_ub=[[x]],
            b_ub=[x],
            bounds=[(0.0, 10.0)],
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= -2.0
    assert r.upper[0] >= -0.5
    assert r.width[0] < 1.6
    # More importantly, it should be close to the true width 1.5 rather than
    # suffering the large extra loss from independent coefficient intervals.
    assert r.lower[0] > -2.01
    assert r.upper[0] < -0.49


def test_linprog_original_active_set_switch_case_is_sound_but_v1_looser():
    # Original specification T3.  This case changes active dual basis at x=1.
    # V1 uses fixed dual witnesses and deliberately does not hide physical
    # subdivision inside linprog merely for tightness.
    def true_v(x):
        return -min(1.0 / x, x)

    def model(p):
        x = p[0]
        return linprog(
            [-1.0],
            A_ub=[[x], [1.0]],
            b_ub=[1.0, x],
            bounds=[(0.0, 10.0)],
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    vals = np.asarray([true_v(float(q)) for q in np.linspace(0.5, 2.0, 2001)])
    assert r.machine_rigorous
    assert r.lower[0] <= vals.min()
    assert vals.max() <= r.upper[0]
    # Caller subdivision should tighten this substantially; V1 does not claim
    # a parametric dual-policy certificate across an active-set switch.
    assert r.width[0] < 2.0



def test_linprog_missing_universal_primal_witness_returns_lower_only():
    # Feasible for every x in [0,1], but the feasible set is the singleton
    # z=x, so no *fixed* z can be feasible for the entire physical box.
    #
    #   min z
    #   s.t. z <= x, -z <= -x, 0 <= z <= 1.
    #
    # Exact V(x)=x.  Weak duality with y=0 and the variable lower bound gives
    # the finite global lower bound 0.  The V1 fixed-primal witness portfolio
    # must therefore return [0,+inf], not discard the lower certificate.
    def model(p):
        x = p[0]
        return linprog(
            [1.0],
            A_ub=[[1.0], [-1.0]],
            b_ub=[x, -x],
            bounds=[(0.0, 1.0)],
        ).fun

    r = autoia(model, [Interval(0.0, 1.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= 0.0
    assert np.isposinf(r.upper[0])
    cert = _linprog_certs(r)[-1]
    assert cert["value_lower_certified"] is True
    assert cert["value_upper_certified"] is False
    assert cert["certificate_completeness"] == "lower-only"
    assert cert["universal_primal_witness_found"] is False
    assert cert["primal_feasible_for_all_q"] is None
    assert cert["selected_primal_z"] is None
    assert cert["primal_residual_upper"] is None
    assert cert["verified_primal_candidate_count"] == 0

    # The one-sided certified value remains an ordinary AutoIA value and can
    # participate in surrounding arithmetic without losing the proven side.
    def shifted_model(p):
        x = p[0]
        v = linprog(
            [1.0],
            A_ub=[[1.0], [-1.0]],
            b_ub=[x, -x],
            bounds=[(0.0, 1.0)],
        ).fun
        return 2.0 * v + 3.0

    s = autoia(shifted_model, [Interval(0.0, 1.0)])
    assert s.machine_rigorous
    assert s.lower[0] <= 3.0
    assert np.isposinf(s.upper[0])


def test_linprog_infeasible_for_part_of_box_returns_lower_only():
    def model(p):
        x = p[0]
        # z <= x and z >= 1.  The LP is infeasible for x<1.  V1 does not
        # attempt to prove where feasibility holds; when no universal primal
        # witness exists it preserves the valid dual lower certificate and
        # returns an extended one-sided value enclosure [L,+inf].
        return linprog(
            [1.0],
            A_ub=[[1.0], [-1.0]],
            b_ub=[x, -1.0],
            bounds=[(None, None)],
        ).fun

    r = autoia(model, [Interval(0.5, 1.5)])
    assert r.machine_rigorous
    assert np.isposinf(r.upper[0])
    cert = _linprog_certs(r)[-1]
    assert cert["value_lower_certified"] is True
    assert cert["value_upper_certified"] is False
    assert cert["certificate_completeness"] == "lower-only"
    assert cert["universal_primal_witness_found"] is False


def test_linprog_unbounded_fails_closed():
    def model(p):
        _ = p[0]
        return linprog([-1.0], bounds=[(0.0, None)]).fun

    with pytest.raises(CertificationError):
        autoia(model, [Interval(0.0, 1.0)])


def test_linprog_res_x_fails_closed():
    def model(p):
        x = p[0]
        res = linprog(
            [-1.0], A_ub=[[1.0]], b_ub=[1.0 + 0.1*x], bounds=[(0.0, 2.0)]
        )
        return res.x[0]

    with pytest.raises(UnsupportedOperationError):
        autoia(model, [Interval(0.0, 1.0)])


def test_linprog_constant_equality_is_supported_via_paired_inequalities():
    def model(p):
        x = p[0]
        # Equality is fixed and the objective is parametric.  Exact value is
        # -max(1,x) on x in [0.5,2].
        return linprog(
            [-1.0, -x], A_eq=[[1.0, 1.0]], b_eq=[1.0],
            bounds=[(0.0, 1.0), (0.0, 1.0)]
        ).fun

    r = autoia(model, [Interval(0.5, 2.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= -2.0
    assert r.upper[0] >= -1.0
    assert r.width[0] < 1.1
    cert = _linprog_certs(r)[-1]
    assert cert["equality_constraint_count"] == 1
    assert cert["equality_semantic"] == "paired-inequalities"


def test_linprog_parametric_equality_can_return_lower_only():
    def model(p):
        x = p[0]
        # Feasible singleton z=x.  No fixed primal witness exists across the box.
        return linprog([1.0], A_eq=[[1.0]], b_eq=[x], bounds=[(0.0, 1.0)]).fun

    r = autoia(model, [Interval(0.0, 1.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= 0.0
    assert np.isposinf(r.upper[0])
    cert = _linprog_certs(r)[-1]
    assert cert["certificate_completeness"] == "lower-only"
    assert cert["equality_constraint_count"] == 1


def test_linprog_top_level_scipy_optimize_proxy():
    def model(p):
        x = p[0]
        return opt.linprog(
            [-1.0], A_ub=[[1.0]], b_ub=[1.0 + 0.1*x], bounds=[(0.0, 2.0)]
        ).fun

    r = autoia(model, [Interval(0.0, 1.0)])
    assert r.machine_rigorous
    # Exact optimum is -(1 + 0.1*x), hence range [-1.1,-1].
    assert r.lower[0] <= -1.1
    assert r.upper[0] >= -1.0
    assert r.width[0] < 0.2


def test_linprog_default_nonnegative_unbounded_above_bounds_supported():
    def model(p):
        x = p[0]
        # SciPy default bounds are 0 <= z < +inf.  Constraint z <= x makes the
        # optimum finite: V(x)=-x for x in [1,2].
        return linprog([-1.0], A_ub=[[1.0]], b_ub=[x]).fun

    r = autoia(model, [Interval(1.0, 2.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= -2.0
    assert r.upper[0] >= -1.0
    assert r.width[0] < 1.1
