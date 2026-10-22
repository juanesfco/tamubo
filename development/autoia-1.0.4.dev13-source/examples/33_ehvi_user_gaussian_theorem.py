"""33 - Tighten analytic EHVI with a user-supplied Gaussian/cell theorem.

This example uses the same two-objective exact-GP problem as Example 32.  The
ordinary cell-decomposition EHVI formula remains unchanged.  The user now adds
mathematical facts that AutoIA cannot be expected to infer from arbitrary code.

For Y ~ N(mu,sigma^2), sigma>0, define

    Psi(t;mu,sigma) = E[(Y-t)_+]
                    = (mu-t) Phi(z) + sigma phi(z),
      z             = (mu-t)/sigma.

User theorem A
--------------
Psi is increasing in both mu and sigma because

    dPsi/dmu    = Phi(z) > 0,
    dPsi/dsigma = phi(z) > 0.

User theorem B
--------------
For a<b define the finite Gaussian band

    D(a,b;mu,sigma) = Psi(a;mu,sigma) - Psi(b;mu,sigma).

D is increasing in mu.  At fixed mu, the sign of dD/dsigma is determined by the
midpoint m=(a+b)/2:

    mu < m  -> D increases with sigma,
    mu > m  -> D decreases with sigma.

For independent objectives, each EHVI cell contribution is a product of
nonnegative one-dimensional factors.  Therefore lower-factor products and
upper-factor products give rigorous cell bounds, and summing the disjoint cells
gives a two-sided EHVI enclosure.

These are application theorems, not AutoIA core semantics.  The theorem status
is explicitly ``user_asserted``.  AutoIA rigorously evaluates the theorem's
numerical operations and runtime assumptions, then the user combines that result
with the generic whole-expression certificate.
"""

import math
import numpy as np
from autoia import (
    Interval,
    autoia,
    standard_normal_cdf,
    standard_normal_pdf,
)
from autoia.application import ApplicationBox, application_theorem


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


def posterior_four(p):
    """One AutoIA trace returns mu1,sigma1,mu2,sigma2 together."""
    mu1, s1 = _gp_posterior(p, ALPHA1)
    mu2, s2 = _gp_posterior(p, ALPHA2)
    return np.array([mu1, s1, mu2, s2], dtype=object)


def _psi(threshold, mu, sigma):
    z = (mu - threshold) / sigma
    return (
        (mu - threshold) * standard_normal_cdf(z)
        + sigma * standard_normal_pdf(z)
    )


def analytic_ehvi(p):
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


def _psi_cert(cert, threshold, mu_point, sigma_point):
    t = cert.point(float(threshold))
    z = (mu_point - t) / sigma_point
    return (
        (mu_point - t) * cert.standard_normal_cdf(z)
        + sigma_point * cert.standard_normal_pdf(z)
    )


def _psi_monotone_pair(cert, threshold, mu_range, sigma_range):
    """User theorem A: Psi increases with both mu and sigma."""
    lower_eval = _psi_cert(
        cert,
        threshold,
        cert.point(mu_range.lower),
        cert.point(sigma_range.lower),
    )
    upper_eval = _psi_cert(
        cert,
        threshold,
        cert.point(mu_range.upper),
        cert.point(sigma_range.upper),
    )
    return lower_eval, upper_eval


def _finite_band_pair(cert, a, b, mu_range, sigma_range):
    """User theorem B: monotonicity of D(a,b;mu,sigma)."""
    cert.require(b > a, "finite Gaussian band has b>a")

    midpoint = 0.5 * (float(a) + float(b))
    mu_lo = float(mu_range.lower)
    mu_hi = float(mu_range.upper)
    sigma_lo = float(sigma_range.lower)
    sigma_hi = float(sigma_range.upper)

    # D is increasing in mu, so the minimum occurs at mu_lo and maximum at
    # mu_hi.  At each of those fixed mu values the midpoint rule chooses the
    # sigma endpoint that minimizes/maximizes D.
    sigma_for_lower = sigma_lo if mu_lo <= midpoint else sigma_hi
    sigma_for_upper = sigma_hi if mu_hi < midpoint else sigma_lo

    lower_eval = (
        _psi_cert(cert, a, cert.point(mu_lo), cert.point(sigma_for_lower))
        - _psi_cert(cert, b, cert.point(mu_lo), cert.point(sigma_for_lower))
    )
    upper_eval = (
        _psi_cert(cert, a, cert.point(mu_hi), cert.point(sigma_for_upper))
        - _psi_cert(cert, b, cert.point(mu_hi), cert.point(sigma_for_upper))
    )
    return lower_eval, upper_eval


def _ehvi_theorem_interval(box, cert):
    posterior = cert.enclose(posterior_four, box)
    mu1, s1, mu2, s2 = posterior[0], posterior[1], posterior[2], posterior[3]

    cert.require(s1.lower > 0.0, "objective-1 posterior sigma is positive")
    cert.require(s2.lower > 0.0, "objective-2 posterior sigma is positive")

    # Objective-1 finite horizontal widths plus the rightmost tail.
    d0 = _finite_band_pair(cert, 0.00, 0.25, mu1, s1)
    d1 = _finite_band_pair(cert, 0.25, 0.40, mu1, s1)
    d2 = _finite_band_pair(cert, 0.40, 0.65, mu1, s1)
    d3 = _finite_band_pair(cert, 0.65, 0.75, mu1, s1)
    d4 = _finite_band_pair(cert, 0.75, 1.00, mu1, s1)
    tail1 = _psi_monotone_pair(cert, 1.00, mu1, s1)

    # Objective-2 vertical heights corresponding to those six cells.
    h0 = _psi_monotone_pair(cert, 1.00, mu2, s2)
    h1 = _psi_monotone_pair(cert, 0.85, mu2, s2)
    h2 = _psi_monotone_pair(cert, 0.75, mu2, s2)
    h3 = _psi_monotone_pair(cert, 0.65, mu2, s2)
    h4 = _psi_monotone_pair(cert, 0.35, mu2, s2)
    h5 = _psi_monotone_pair(cert, 0.00, mu2, s2)

    horizontal = [d0, d1, d2, d3, d4, tail1]
    vertical = [h0, h1, h2, h3, h4, h5]

    # Every cell factor is mathematically nonnegative.  Require that the
    # proof-safe numerical corner evaluations establish this before using simple
    # product monotonicity.
    for i, pair in enumerate(horizontal + vertical):
        cert.require(pair[0].lower >= 0.0, f"EHVI theorem lower factor {i} is nonnegative")

    lower_cells = [horizontal[i][0] * vertical[i][0] for i in range(6)]
    upper_cells = [horizontal[i][1] * vertical[i][1] for i in range(6)]
    return cert.sum(lower_cells), cert.sum(upper_cells)


@application_theorem(
    name="analytic EHVI upper bound from Gaussian cell monotonicity",
    target="EHVI",
    theorem_status="user_asserted",
)
def ehvi_theorem_upper(box, cert):
    return _ehvi_theorem_interval(box, cert)[1]


@application_theorem(
    name="analytic negative-EHVI upper bound from Gaussian cell monotonicity",
    target="-EHVI",
    theorem_status="user_asserted",
)
def negative_ehvi_theorem_upper(box, cert):
    # If L <= EHVI, then -EHVI <= -L.
    return -_ehvi_theorem_interval(box, cert)[0]


# Ordinary numeric reference, diagnostic only.
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


if __name__ == "__main__":
    # This is the same deliberately troublesome box highlighted in Example 32.
    interval_box = [
        Interval.enclosing("0.265", "0.335"),
        Interval.enclosing("0.665", "0.735"),
    ]
    application_box = ApplicationBox.from_intervals(interval_box)

    generic = autoia(analytic_ehvi, interval_box)
    generic_lo = float(generic.lower[0])
    generic_hi = float(generic.upper[0])

    theorem_upper = ehvi_theorem_upper(application_box)
    negative_theorem_upper = negative_ehvi_theorem_upper(application_box)
    theorem_lo = -float(negative_theorem_upper.value)
    theorem_hi = float(theorem_upper.value)

    # EHVI is itself nonnegative.  Intersect all independently valid bounds.
    final_lo = max(0.0, generic_lo, theorem_lo)
    final_hi = min(generic_hi, theorem_hi)

    print("generic AutoIA EHVI       =", [generic_lo, generic_hi])
    print("Gaussian theorem EHVI    =", [theorem_lo, theorem_hi])
    print("EHVI >= 0 theorem lower  =", 0.0)
    print("final theorem portfolio  =", [final_lo, final_hi])
    print("theorem status           =", theorem_upper.theorem_status)
    print("checked assumptions      =", theorem_upper.assumptions_checked)

    xs = np.linspace(0.265, 0.335, 41)
    ys = np.linspace(0.665, 0.735, 41)
    sampled = np.array([_ehvi_numeric(np.array([x0, x1])) for x0 in xs for x1 in ys])
    sampled_lo = float(sampled.min())
    sampled_hi = float(sampled.max())
    print("dense diagnostic range   =", [sampled_lo, sampled_hi])
    print("sampled values contained =", final_lo <= sampled_lo and sampled_hi <= final_hi)

    assert generic.machine_rigorous
    assert theorem_upper.machine_rigorous
    assert negative_theorem_upper.machine_rigorous
    assert final_lo <= sampled_lo and sampled_hi <= final_hi
    assert final_hi < generic_hi
