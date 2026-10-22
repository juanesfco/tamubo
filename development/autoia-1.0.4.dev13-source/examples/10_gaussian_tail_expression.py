"""10 - Gaussian PDF/CDF expression written as ordinary engineering code.

This example demonstrates generic automatic AutoIA execution through a nonlinear
expression containing a stationary exponential term and normal PDF/CDF calls.
"""
import numpy as np
from autoia import Interval, autoia, standard_normal_pdf, standard_normal_cdf

X_REF = np.array([[0.15], [0.55], [0.9]])
WEIGHTS = np.array([-0.8, 0.35, -0.1])
LENGTH_SCALE = 0.25
SCALE = 1.0
THRESHOLD = -0.2


def gaussian_tail_expression(p):
    x = p[0]
    k = np.empty(3, dtype=object)
    for i in range(3):
        r = (x - X_REF[i, 0]) / LENGTH_SCALE
        k[i] = SCALE * np.exp(-0.5 * r * r)

    mu = np.dot(k, WEIGHTS)
    sigma = 0.3 + 0.1 * np.exp(-x * x)
    a = THRESHOLD - mu
    z = a / sigma
    return a * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)


if __name__ == "__main__":
    result = autoia(gaussian_tail_expression, [Interval(0.30, 0.40)])
    print(result)
