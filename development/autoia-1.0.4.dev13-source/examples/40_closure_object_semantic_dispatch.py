"""Example 40 -- supported SciPy semantics reached through closures and objects.

AutoIA 1.0.4.dev10 extends the local non-mutating dispatch boundary through
ordinary Python closure variables and source-level user objects.  The user does
not need to publish helpers to module globals merely so AutoIA can see them.
"""

import numpy as np
from scipy.optimize import fsolve

from autoia import Interval, autoia


def response(x):
    def residual(y):
        return np.array([y[0] - x], dtype=object)
    return fsolve(residual, [0.4])[0]


def make_function_model():
    fn = response  # captured closure variable

    def model(p):
        return fn(p[0])

    return model


class Problem:
    def _solve(self, x):
        def residual(y):
            return np.array([y[0] - x], dtype=object)
        return fsolve(residual, [0.4])[0]

    def g(self, x):
        # Nested method call deliberately exercises the captured-object proxy.
        return self._solve(x)


def make_object_model():
    prob = Problem()  # captured closure object

    def model(p):
        return prob.g(p[0])

    return model


if __name__ == "__main__":
    box = [Interval.enclosing("0.2", "0.6")]

    r1 = autoia(make_function_model(), box)
    print("captured function")
    print(r1)

    r2 = autoia(make_object_model(), box)
    print("\ncaptured object")
    print(r2)

    assert r1.machine_rigorous and r2.machine_rigorous
    assert r1.lower[0] <= 0.2 <= r1.upper[0]
    assert r1.lower[0] <= 0.6 <= r1.upper[0]
    assert r2.lower[0] <= 0.2 <= r2.upper[0]
    assert r2.lower[0] <= 0.6 <= r2.upper[0]
