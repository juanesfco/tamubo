"""30 - Hand-coded exact GP followed by analytic Expected Improvement (EI).

This is an advanced *ordinary-code* AutoIA example.  AutoIA does not contain an
"EI primitive" here.  The user writes a fixed Gaussian-process posterior and the
usual analytic EI expression, and AutoIA encloses the whole computation over a
user-specified design interval.

Notation used below
-------------------
x          uncertain scalar design coordinate.
k(x)       covariance vector between x and the fixed training inputs.
mu(x)      posterior mean.
L          fixed lower-triangular Cholesky factor of the training covariance.
v(x)       solution of L v(x) = k(x).
s2(x)      latent posterior variance = signal_variance - v(x)^T v(x).
sigma(x)   posterior standard deviation = sqrt(s2(x)).
best_f     largest observed objective value (this example maximizes).
a(x)       improvement mean = mu(x) - best_f.
Phi, phi   standard-normal CDF and PDF.
EI(x)      a(x) Phi(a/sigma) + sigma phi(a/sigma).

The GP fit itself is represented here by fixed binary64 data.  AutoIA certifies
the subsequent supported calculation conditional on those fixed values.
"""

import math
import numpy as np

from autoia import (
    Interval,
    autoia,
    standard_normal_cdf,
    standard_normal_pdf,
)


# -----------------------------------------------------------------------------
# Fixed fitted GP state.
# -----------------------------------------------------------------------------
X_TRAIN = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=float)
Y_TRAIN = np.sin(2.0 * math.pi * X_TRAIN) + 0.15 * X_TRAIN

LENGTHSCALE = 0.22
SIGNAL_VARIANCE = 1.0
NOISE_STD = 0.08


def _rbf_numeric(x, y):
    d = (x - y) / LENGTHSCALE
    return SIGNAL_VARIANCE * math.exp(-0.5 * d * d)


K_TRAIN = np.empty((len(X_TRAIN), len(X_TRAIN)), dtype=float)
for i, xi in enumerate(X_TRAIN):
    for j, xj in enumerate(X_TRAIN):
        K_TRAIN[i, j] = _rbf_numeric(float(xi), float(xj))
K_TRAIN += (NOISE_STD ** 2) * np.eye(len(X_TRAIN))

CHOLESKY = np.linalg.cholesky(K_TRAIN)
ALPHA = np.linalg.solve(K_TRAIN, Y_TRAIN)
BEST_F = float(np.max(Y_TRAIN))


def _forward_solve(k):
    """Ordinary forward substitution for the fixed triangular matrix L."""
    out = np.empty(len(k), dtype=object)
    for i in range(len(k)):
        rhs = k[i]
        for j in range(i):
            rhs = rhs - float(CHOLESKY[i, j]) * out[j]
        out[i] = rhs / float(CHOLESKY[i, i])
    return out


def gp_posterior(p):
    """Return the hand-coded latent posterior mean and standard deviation."""
    x = p[0]

    k = np.empty(len(X_TRAIN), dtype=object)
    for i, xi in enumerate(X_TRAIN):
        d = (x - float(xi)) / LENGTHSCALE
        k[i] = SIGNAL_VARIANCE * np.exp(-0.5 * d * d)

    mu = np.dot(k, ALPHA)
    v = _forward_solve(k)
    variance = SIGNAL_VARIANCE - np.dot(v, v)
    sigma = np.sqrt(variance)
    return mu, sigma


def expected_improvement(p):
    """Analytic EI for maximization, written as ordinary supported code."""
    mu, sigma = gp_posterior(p)
    a = mu - BEST_F
    z = a / sigma
    return a * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)


# Ordinary floating-point reference used only as a diagnostic, never as proof.
def _numeric_ei(x):
    k = np.array([_rbf_numeric(float(x), float(xi)) for xi in X_TRAIN])
    mu = float(k @ ALPHA)
    v = np.linalg.solve(CHOLESKY, k)
    variance = float(SIGNAL_VARIANCE - v @ v)
    sigma = math.sqrt(max(variance, 0.0))
    a = mu - BEST_F
    if sigma == 0.0:
        return max(a, 0.0)
    z = a / sigma
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return a * Phi + sigma * phi


if __name__ == "__main__":
    # A local box where generic whole-expression propagation is already sharp.
    box = [Interval.enclosing("0.30", "0.36")]
    result = autoia(expected_improvement, box)

    print(result)

    # Dense samples are only a diagnostic for the example.
    grid = np.linspace(0.30, 0.36, 2001)
    sampled = np.array([_numeric_ei(float(x)) for x in grid])
    sampled_lo = float(sampled.min())
    sampled_hi = float(sampled.max())
    proof_lo = float(result.lower[0])
    proof_hi = float(result.upper[0])

    print("dense diagnostic EI range =", [sampled_lo, sampled_hi])
    print("sampled values contained  =", proof_lo <= sampled_lo and sampled_hi <= proof_hi)

    assert result.machine_rigorous
    assert proof_lo <= sampled_lo and sampled_hi <= proof_hi
