"""25 — RBF-like SPD family and shrinking-box behavior.

This is deliberately *not* Gaussian-process code.  It is a generic structured
SPD matrix family whose entries all depend on the same uncertain parameters.
It demonstrates why AutoIA preserves that common dependence through ``slogdet``
and ``quad_solve_spd`` rather than first replacing every matrix entry by an
independent interval.
"""
import numpy as np
from autoia import Interval, autoia, quad_solve_spd


X = np.array([0.0, 0.45, 1.0])
y_obs = np.array([0.2, -0.3, 0.5])


def matrix_objective(p):
    log_ell, log_lam = p
    ell = np.exp(log_ell)
    lam = np.exp(log_lam)

    K = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            r = (X[i] - X[j]) / ell
            K[i, j] = np.exp(-0.5*r*r)
        K[i, i] = K[i, i] + lam

    sign, logdet = np.linalg.slogdet(K)
    q = quad_solve_spd(K, y_obs)
    combined = 0.5*(logdet + q)
    return np.array([sign, logdet, q, combined], dtype=object)


if __name__ == "__main__":
    center = np.array([np.log(0.33), np.log(0.0725)])
    halfwidths = [0.12, 0.06, 0.03, 0.015]

    print("columns: halfwidth, logdet width, quadratic width, combined width")
    previous = None
    for h in halfwidths:
        box = [
            Interval(float(center[0] - h), float(center[0] + h)),
            Interval(float(center[1] - h), float(center[1] + h)),
        ]
        result = autoia(matrix_objective, box)
        assert result.machine_rigorous
        widths = np.asarray(result.upper - result.lower, dtype=float)
        print(f"{h:7.4f}  {widths[1]:.10f}  {widths[2]:.10f}  {widths[3]:.10f}")
        if previous is not None:
            assert np.all(widths[1:] <= previous[1:] + 1e-12)
        previous = widths

    print("\nAll nested-box certificates are machine rigorous.")
    print("The structured SPD enclosures tighten as the shared parameter box shrinks.")
