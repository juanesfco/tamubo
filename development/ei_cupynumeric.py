import cupynumeric as cp

# Normal CDF/PDF using erf
sqrt2 = cp.sqrt(2.0)
inv_sqrt2pi = 1.0 / cp.sqrt(2.0 * cp.pi)

def erf_approx(x):
    """
    Approximate erf(x) using Abramowitz & Stegun 7.1.26.
    Max error ~1.5e-7.
    """
    # Coefficients
    p  = 0.3275911
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
    return 0.5 * (1.0 + erf_approx(z / sqrt2))

def norm_pdf(z):
    return inv_sqrt2pi * cp.exp(-0.5 * z * z)

def expected_improvement(mu, sigma, y_min):
    """
    Compute Expected Improvement (EI) for a minimization objective.

    Args:
        mu (float or cp.ndarray): Predictive mean(s) at candidate points.
        sigma (float or cp.ndarray): Predictive std dev(s) at candidate points.
        y_min (float or cp.ndarray): Best observed objective value(s). Broadcastable
            to mu/sigma.

    Returns:
        cp.ndarray: Expected improvement values, non-negative, same broadcasted shape
        as mu/sigma. When sigma == 0, EI is 0.
    """
    mu = cp.asarray(mu)
    sigma = cp.asarray(sigma)
    y_min = cp.asarray(y_min)

    safe_sigma = cp.where(sigma == 0, 1.0, sigma)
    Z = (y_min - mu) / safe_sigma  # to minimize
    ei = (y_min - mu) * norm_cdf(Z) + safe_sigma * norm_pdf(Z)  # to minimize
    return cp.where(sigma == 0, 0.0, ei)
