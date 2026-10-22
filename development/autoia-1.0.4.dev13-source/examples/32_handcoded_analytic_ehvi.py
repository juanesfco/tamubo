"""32 - Hand-coded analytic two-objective EHVI passed directly to AutoIA.

This example is the multi-objective analogue of Example 30.  It implements a
standard analytic q=1 Expected Hypervolume Improvement (EHVI) calculation for
two independent Gaussian objective posteriors and a fixed Pareto staircase.
AutoIA is not given any EHVI-specific theorem in this file.

Both objectives are maximized.  For a Gaussian Y ~ N(mu,sigma^2), define

    Psi(t;mu,sigma) = E[(Y-t)_+]
                    = (mu-t) Phi(z) + sigma phi(z),
      z             = (mu-t)/sigma.

The non-dominated region above a fixed two-objective Pareto staircase can be
written as a sum of disjoint rectangular cells.  The q=1 analytic EHVI below is
therefore ordinary arithmetic composed with the GP posterior and Psi.
"""

import math
import numpy as np
from autoia import (
    Interval,
    prepare,
    standard_normal_cdf,
    standard_normal_pdf,
)


X_TRAIN = np.array(
    [
        [0.10, 0.10],
        [0.20, 0.80],
        [0.80, 0.20],
        [0.85, 0.85],
        [0.50, 0.50],
    ],
    dtype=float,
)
Y1_TRAIN = np.array([0.25, 0.40, 1.00, 0.75, 0.65], dtype=float)
Y2_TRAIN = np.array([1.00, 0.85, 0.35, 0.65, 0.75], dtype=float)

LENGTHSCALE = np.array([0.32, 0.32], dtype=float)
SIGNAL_VARIANCE = 1.0
NOISE_STD = 0.05
PREDICT_OBSERVATION_NOISE = True

# Fixed maximization Pareto staircase and reference point (0,0).
PARETO = np.array(
    [
        [0.25, 1.00],
        [0.40, 0.85],
        [0.65, 0.75],
        [0.75, 0.65],
        [1.00, 0.35],
    ],
    dtype=float,
)


def _numeric_rbf_matrix(xa, xb):
    d = (xa[:, None, :] - xb[None, :, :]) / LENGTHSCALE
    return SIGNAL_VARIANCE * np.exp(-0.5 * np.sum(d * d, axis=2))


K_TRAIN = _numeric_rbf_matrix(X_TRAIN, X_TRAIN)
K_TRAIN += (NOISE_STD ** 2) * np.eye(len(X_TRAIN))
K_INV = np.linalg.inv(K_TRAIN)
ALPHA1 = K_INV @ Y1_TRAIN
ALPHA2 = K_INV @ Y2_TRAIN


def _kernel_vector(p):
    x0, x1 = p[0], p[1]
    out = np.empty(len(X_TRAIN), dtype=object)
    for i, (a, b) in enumerate(X_TRAIN):
        q = (
            ((x0 - float(a)) / LENGTHSCALE[0]) ** 2
            + ((x1 - float(b)) / LENGTHSCALE[1]) ** 2
        )
        out[i] = SIGNAL_VARIANCE * np.exp(-0.5 * q)
    return out


def _gp_posterior(p, alpha):
    k = _kernel_vector(p)
    mu = np.dot(k, alpha)
    variance = SIGNAL_VARIANCE - np.dot(k, np.dot(K_INV, k))
    if PREDICT_OBSERVATION_NOISE:
        variance = variance + NOISE_STD ** 2
    return mu, np.sqrt(variance)


def _psi(threshold, mu, sigma):
    z = (mu - threshold) / sigma
    return (
        (mu - threshold) * standard_normal_cdf(z)
        + sigma * standard_normal_pdf(z)
    )


def analytic_ehvi(p):
    """Exact q=1 cell-decomposition formula for this fixed two-objective front."""
    mu1, s1 = _gp_posterior(p, ALPHA1)
    mu2, s2 = _gp_posterior(p, ALPHA2)

    p10, p125, p140, p165, p175, p1100 = [
        _psi(t, mu1, s1) for t in (0.00, 0.25, 0.40, 0.65, 0.75, 1.00)
    ]
    p20, p235, p265, p275, p285, p2100 = [
        _psi(t, mu2, s2) for t in (0.00, 0.35, 0.65, 0.75, 0.85, 1.00)
    ]

    return (
        (p10 - p125) * p2100
        + (p125 - p140) * p285
        + (p140 - p165) * p275
        + (p165 - p175) * p265
        + (p175 - p1100) * p235
        + p1100 * p20
    )


# Ordinary numeric reference used only for the printed containment diagnostic.
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _psi_numeric(t, mu, sigma):
    z = (mu - t) / sigma
    return (mu - t) * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) + sigma * _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def _gp_numeric(x, alpha):
    d = (x[None, :] - X_TRAIN) / LENGTHSCALE
    k = SIGNAL_VARIANCE * np.exp(-0.5 * np.sum(d * d, axis=1))
    mu = float(k @ alpha)
    variance = float(SIGNAL_VARIANCE - k @ K_INV @ k)
    if PREDICT_OBSERVATION_NOISE:
        variance += NOISE_STD ** 2
    return mu, math.sqrt(variance)


def _ehvi_numeric(x):
    m1, s1 = _gp_numeric(x, ALPHA1)
    m2, s2 = _gp_numeric(x, ALPHA2)

    p10, p125, p140, p165, p175, p1100 = [
        _psi_numeric(t, m1, s1) for t in (0.00, 0.25, 0.40, 0.65, 0.75, 1.00)
    ]
    p20, p235, p265, p275, p285, p2100 = [
        _psi_numeric(t, m2, s2) for t in (0.00, 0.35, 0.65, 0.75, 0.85, 1.00)
    ]

    return (
        (p10 - p125) * p2100
        + (p125 - p140) * p285
        + (p140 - p165) * p275
        + (p165 - p175) * p265
        + (p175 - p1100) * p235
        + p1100 * p20
    )


def _run_box(label, center, half_width, session):
    c = np.asarray(center, dtype=float)
    h = float(half_width)
    box = [
        Interval.enclosing(str(c[0] - h), str(c[0] + h)),
        Interval.enclosing(str(c[1] - h), str(c[1] + h)),
    ]
    result = session.certify(box)

    xs = np.linspace(c[0] - h, c[0] + h, 31)
    ys = np.linspace(c[1] - h, c[1] + h, 31)
    sampled = np.array([_ehvi_numeric(np.array([x0, x1])) for x0 in xs for x1 in ys])

    proof_lo = float(result.lower[0])
    proof_hi = float(result.upper[0])
    sample_lo = float(sampled.min())
    sample_hi = float(sampled.max())

    print("\n" + label)
    print("  AutoIA EHVI interval    =", [proof_lo, proof_hi])
    print("  dense diagnostic range  =", [sample_lo, sample_hi])
    print("  sampled values contained=", proof_lo <= sample_lo and sample_hi <= proof_hi)

    assert result.machine_rigorous
    assert proof_lo <= sample_lo and sample_hi <= proof_hi


if __name__ == "__main__":
    session = prepare(analytic_ehvi)

    # A local box where the ordinary formula is already tight.
    _run_box("local box", np.array([0.35, 0.35]), 0.02, session)

    # A box where the same exact formula is rigorous but much looser.  Example 33
    # will add user mathematics without changing this ordinary EHVI source.
    _run_box("theorem-motivating box", np.array([0.30, 0.70]), 0.035, session)
