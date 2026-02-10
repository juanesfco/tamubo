from __future__ import annotations

import cupynumeric as cp

# Normal CDF/PDF constants
SQRT2 = cp.sqrt(2.0)
INV_SQRT_2PI = 1.0 / cp.sqrt(2.0 * cp.pi)


def erf_approx(x):
    """Approximate erf(x) using Abramowitz & Stegun 7.1.26."""

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = cp.sign(x)
    ax = cp.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * cp.exp(-ax * ax)
    return sign * y


def norm_cdf(z):
    return 0.5 * (1.0 + erf_approx(z / SQRT2))


def norm_pdf(z):
    return INV_SQRT_2PI * cp.exp(-0.5 * z * z)


def expected_improvement(mu, sigma, y_min):
    """Compute EI for minimization with vectorized cupynumeric ops."""

    mu = cp.asarray(mu)
    sigma = cp.asarray(sigma)
    y_min = cp.asarray(y_min)

    safe_sigma = cp.where(sigma == 0, 1.0, sigma)
    z = (y_min - mu) / safe_sigma
    ei = (y_min - mu) * norm_cdf(z) + safe_sigma * norm_pdf(z)
    return cp.where(sigma == 0, 0.0, ei)
