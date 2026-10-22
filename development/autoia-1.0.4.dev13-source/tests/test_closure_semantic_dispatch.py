import numpy as np
from scipy.optimize import fsolve, linprog

from autoia import Interval, autoia


def _response_with_fsolve(x):
    def residual(y):
        return np.array([y[0] - x], dtype=object)
    return fsolve(residual, [0.4])[0]


def test_fsolve_inside_function_captured_by_closure_is_prepared():
    fn = _response_with_fsolve

    def model(p):
        return fn(p[0])

    r = autoia(model, [Interval(0.2, 0.6)])
    assert r.machine_rigorous
    assert r.lower[0] <= 0.2
    assert r.upper[0] >= 0.6
    assert any(c.get("operation") == "scipy.optimize.fsolve" for c in r.certificates)


class _ProblemWithNestedMethods:
    def _solve(self, x):
        def residual(y):
            return np.array([y[0] - x], dtype=object)
        return fsolve(residual, [0.4])[0]

    def g(self, x):
        # Nested self-method call is intentional: the captured object must stay
        # behind the same local dispatch boundary after entering g().
        return self._solve(x)


def test_fsolve_inside_method_of_object_captured_by_closure_is_prepared():
    prob = _ProblemWithNestedMethods()

    def model(p):
        return prob.g(p[0])

    r = autoia(model, [Interval(0.2, 0.6)])
    assert r.machine_rigorous
    assert r.lower[0] <= 0.2
    assert r.upper[0] >= 0.6
    assert any(c.get("operation") == "scipy.optimize.fsolve" for c in r.certificates)


def _response_with_linprog(x):
    return linprog(
        [-1.0],
        A_ub=[[1.0]],
        b_ub=[x],
        bounds=[(0.0, 2.0)],
    ).fun


def test_linprog_inside_function_captured_by_closure_is_prepared():
    fn = _response_with_linprog

    def model(p):
        return fn(p[0])

    r = autoia(model, [Interval(0.5, 1.0)])
    assert r.machine_rigorous
    # V(x)=-x.
    assert r.lower[0] <= -1.0
    assert r.upper[0] >= -0.5
    assert any(c.get("operation") == "scipy.optimize.linprog" for c in r.certificates)


class _FunctionAttributeHolder:
    pass


def test_fsolve_inside_function_valued_attribute_of_captured_object_is_prepared():
    holder = _FunctionAttributeHolder()
    holder.g = _response_with_fsolve

    def model(p):
        return holder.g(p[0])

    r = autoia(model, [Interval(0.2, 0.6)])
    assert r.machine_rigorous
    assert r.lower[0] <= 0.2
    assert r.upper[0] >= 0.6
    assert any(c.get("operation") == "scipy.optimize.fsolve" for c in r.certificates)


def test_linprog_inside_function_valued_attribute_of_captured_object_is_prepared():
    holder = _FunctionAttributeHolder()
    holder.g = _response_with_linprog

    def model(p):
        return holder.g(p[0])

    r = autoia(model, [Interval(0.5, 1.0)])
    assert r.machine_rigorous
    assert r.lower[0] <= -1.0
    assert r.upper[0] >= -0.5
    assert any(c.get("operation") == "scipy.optimize.linprog" for c in r.certificates)
