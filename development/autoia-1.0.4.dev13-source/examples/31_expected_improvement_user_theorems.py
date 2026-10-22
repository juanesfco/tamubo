"""31 - Tighten Expected Improvement with user-supplied application theorems.

This example uses the same kind of hand-coded exact GP and analytic EI as
Example 30, but on a wider design box where fully generic propagation is loose.
It demonstrates the intended advanced AutoIA pattern:

    ordinary supported code
      + independently valid user mathematics
      + rigorous proof portfolio.

AutoIA still does not know what a GP or EI *means* at the application-theorem
level.  The decorators below declare user-supplied mathematical implications;
AutoIA rigorously evaluates their numerical interval arithmetic and records the
assumptions, but ``theorem_status='user_asserted'`` does not claim that AutoIA
proved the application theorem itself.

The two user facts illustrated are:

1. EI monotonicity.  For maximization define

       g(a,sigma) = a Phi(a/sigma) + sigma phi(a/sigma).

   For sigma > 0,

       partial g / partial a     = Phi(a/sigma) >= 0,
       partial g / partial sigma = phi(a/sigma) > 0.

   Hence a certified upper bound on posterior mean and standard deviation gives
   an EI upper bound by evaluating g at the upper endpoints.

2. RBF box geometry.  For each training point, minimum and maximum distances
   from an axis-aligned input box are known exactly from the box endpoints.
   Because the RBF kernel decreases with squared distance, those distances give
   an independent certified interval for every kernel component.

The final upper bound is the minimum of independent rigorous upper bounds.  The
final lower bound additionally uses the user-known theorem EI >= 0.
"""

import math
import numpy as np

from autoia import (
    Interval,
    autoia,
    standard_normal_cdf,
    standard_normal_pdf,
)
from autoia.application import (
    ApplicationBox,
    CertificationContext,
    application_theorem,
    minimum_upper_bound,
)


# -----------------------------------------------------------------------------
# Same fixed GP family as Example 30.
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
    out = np.empty(len(k), dtype=object)
    for i in range(len(k)):
        rhs = k[i]
        for j in range(i):
            rhs = rhs - float(CHOLESKY[i, j]) * out[j]
        out[i] = rhs / float(CHOLESKY[i, i])
    return out


def posterior_mean_variance(p):
    """Ordinary model returning mu(x) and latent variance s^2(x)."""
    x = p[0]
    k = np.empty(len(X_TRAIN), dtype=object)
    for i, xi in enumerate(X_TRAIN):
        d = (x - float(xi)) / LENGTHSCALE
        k[i] = SIGNAL_VARIANCE * np.exp(-0.5 * d * d)

    mu = np.dot(k, ALPHA)
    v = _forward_solve(k)
    variance = SIGNAL_VARIANCE - np.dot(v, v)
    return np.array([mu, variance], dtype=object)


def expected_improvement(p):
    mu, variance = posterior_mean_variance(p)
    sigma = np.sqrt(variance)
    a = mu - BEST_F
    z = a / sigma
    return a * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)


def _ei_corner(cert, a_point, sigma_point):
    """Rigorous numerical evaluation of g(a,sigma) at one fixed corner."""
    if sigma_point.lower == 0.0 and sigma_point.upper == 0.0:
        return cert.maximum(a_point, 0.0)
    z = a_point / sigma_point
    return (
        a_point * cert.standard_normal_cdf(z)
        + sigma_point * cert.standard_normal_pdf(z)
    )


@application_theorem(
    name="EI upper bound from posterior enclosure and EI monotonicity",
    target="EI",
    theorem_status="user_asserted",
)
def posterior_monotone_ei_upper(box, cert):
    # One dependency-preserving AutoIA call encloses both posterior quantities.
    posterior = cert.enclose(posterior_mean_variance, box)
    mu = posterior[0]
    variance = posterior[1]

    # Application fact supplied by the user: this is the latent variance of a
    # valid exact GP posterior, so its true value is nonnegative.  For an upper
    # EI bound we only need a rigorous variance upper endpoint.
    cert.require(variance.upper >= 0.0, "posterior variance upper endpoint is nonnegative")

    sigma_upper = cert.sqrt(cert.point(max(0.0, variance.upper)))
    a_upper = cert.point(mu.upper) - BEST_F

    # User theorem: g is nondecreasing in both a and sigma.
    return _ei_corner(cert, a_upper, sigma_upper)


@application_theorem(
    name="RBF box geometry plus EI monotonicity",
    target="EI",
    theorem_status="user_asserted",
)
def radial_rbf_ei_upper(box, cert):
    """Independent RBF geometry theorem, evaluated with public proof tools."""
    x_lo = cert.point(float(box.lower[0]))
    x_hi = cert.point(float(box.upper[0]))
    ell = cert.point(LENGTHSCALE)
    signal = cert.point(SIGNAL_VARIANCE)

    kernel_ranges = []
    for xi_float in X_TRAIN:
        xi = cert.point(float(xi_float))

        # Smallest and largest distance from the 1-D box to xi, divided by ell.
        d_min = cert.maximum(
            cert.maximum(x_lo - xi, xi - x_hi),
            cert.point(0.0),
        ) / ell
        d_max = cert.maximum(abs(x_lo - xi), abs(x_hi - xi)) / ell

        # RBF decreases monotonically with squared distance.
        k_lower_eval = signal * cert.exp(-0.5 * cert.square(d_max))
        k_upper_eval = signal * cert.exp(-0.5 * cert.square(d_min))
        kernel_ranges.append(
            cert.interval(k_lower_eval.lower, k_upper_eval.upper)
        )

    mu = cert.dot(ALPHA, kernel_ranges)

    # Propagate the independent radial kernel intervals through the same stored
    # lower-triangular factor to obtain an independent variance upper bound.
    v = cert.solve_lower_triangular(CHOLESKY, kernel_ranges)
    norm2 = cert.sum(cert.square(vi) for vi in v)
    variance = signal - norm2
    cert.require(variance.upper >= 0.0, "radial posterior variance upper endpoint is nonnegative")

    sigma_upper = cert.sqrt(cert.point(max(0.0, variance.upper)))
    a_upper = cert.point(mu.upper) - BEST_F
    return _ei_corner(cert, a_upper, sigma_upper)


# Ordinary floating-point reference, diagnostic only.
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
    # Wider than Example 30 on purpose: generic EI remains rigorous but loses
    # substantial sharpness, making the theorem portfolio visible.
    interval_box = [Interval.enclosing("0.20", "0.40")]
    application_box = ApplicationBox.from_intervals(interval_box)

    direct = autoia(expected_improvement, interval_box)
    direct_lo = float(direct.lower[0])
    direct_hi = float(direct.upper[0])

    # Turn the direct AutoIA upper endpoint into a public UpperBound object so
    # it can participate in the same proof portfolio as the user theorems.
    cert = CertificationContext()
    direct_again = cert.enclose(expected_improvement, application_box)
    direct_upper = cert.upper_bound(
        direct_again,
        target="EI",
        details={"route": "ordinary whole-expression AutoIA"},
    )

    posterior_upper = posterior_monotone_ei_upper(application_box)
    radial_upper = radial_rbf_ei_upper(application_box)
    best_upper = minimum_upper_bound(direct_upper, posterior_upper, radial_upper)

    # Independent user theorem: EI is an expectation of a nonnegative random
    # improvement, hence EI >= 0.  Intersect this with the generic lower bound.
    final_lower = max(direct_lo, 0.0)
    final_upper = float(best_upper.value)

    print("direct AutoIA EI interval =", [direct_lo, direct_hi])
    print("posterior+monotone upper  =", posterior_upper.value)
    print("RBF radial theorem upper  =", radial_upper.value)
    print("portfolio winning upper   =", best_upper.value)
    print("portfolio winner          =", best_upper.winner.theorem)
    print("EI >= 0 theorem lower     =", 0.0)
    print("final theorem portfolio   =", [final_lower, final_upper])

    grid = np.linspace(0.20, 0.40, 2001)
    sampled = np.array([_numeric_ei(float(x)) for x in grid])
    sampled_lo = float(sampled.min())
    sampled_hi = float(sampled.max())
    print("dense diagnostic EI range =", [sampled_lo, sampled_hi])
    print("sampled values contained  =", final_lower <= sampled_lo and sampled_hi <= final_upper)

    assert direct.machine_rigorous
    assert posterior_upper.machine_rigorous
    assert radial_upper.machine_rigorous
    assert final_lower <= sampled_lo and sampled_hi <= final_upper
    assert final_upper <= direct_hi
