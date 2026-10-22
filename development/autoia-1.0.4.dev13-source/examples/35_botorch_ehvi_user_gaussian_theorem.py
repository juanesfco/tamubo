"""35 - Tighten ordinary BoTorch EHVI with a user-known Gaussian cell theorem.

Example 34 established that AutoIA can transparently execute BoTorch's own
analytic EHVI code.  On the deliberately troublesome box used here, the generic
EHVI composition is rigorous but very loose even though the certified BoTorch
posterior mean and variance ranges are extremely sharp.

This example demonstrates the advanced AutoIA workflow:

    unchanged BoTorch EHVI certificate
                 +
    user-asserted Gaussian/cell theorem
                 =
    rigorous proof portfolio

The theorem does NOT replace the BoTorch model.  It asks AutoIA to enclose the
posterior mean/variance of the same fixed fitted BoTorch model, then uses
application knowledge about Gaussian positive-part expectations to derive an
independent EHVI upper bound.

Notation
--------
For Y ~ N(mu, sigma^2), define

    Psi(t; mu, sigma) = E[(Y-t)_+]
                      = (mu-t) Phi(z) + sigma phi(z),
      z               = (mu-t)/sigma.

Here Phi is the standard-normal CDF and phi is the standard-normal PDF.

User theorem A:

    dPsi/dmu    = Phi(z) > 0,
    dPsi/dsigma = phi(z) > 0.

Thus Psi increases in both mu and sigma.

For a<b define the finite Gaussian band

    D(a,b;mu,sigma) = Psi(a;mu,sigma) - Psi(b;mu,sigma).

User theorem B:

- D increases in mu;
- at fixed mu, D increases with sigma when mu < (a+b)/2 and decreases with
  sigma when mu > (a+b)/2.

For independent objectives, the q=1 two-objective EHVI decomposition is a sum
of products of nonnegative one-dimensional cell factors.  Upper bounds for the
factors therefore give an EHVI upper bound.  EHVI itself is nonnegative, giving
the lower bound zero.

``theorem_status='user_asserted'`` is intentional: AutoIA rigorously evaluates
the theorem's numerical operations and checked assumptions, but it does not
claim to have derived these Gaussian/EHVI facts from arbitrary Python source.
"""

import math

try:
    import torch
    import gpytorch
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.acquisition.multi_objective.analytic import (
        ExpectedHypervolumeImprovement,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError as exc:
    print("SKIP: Example 35 requires torch, gpytorch, and botorch.")
    print("Install those optional packages in the AutoIA environment and rerun.")
    raise SystemExit(0) from exc

from autoia import Interval, autoia
from autoia.application import ApplicationBox, application_theorem


torch.set_default_dtype(torch.double)

TRAIN_X = torch.tensor(
    [
        [0.10, 0.10],
        [0.20, 0.80],
        [0.80, 0.20],
        [0.85, 0.85],
        [0.50, 0.50],
    ],
    dtype=torch.double,
)
TRAIN_Y = torch.tensor(
    [
        [0.25, 1.00],
        [0.40, 0.85],
        [1.00, 0.35],
        [0.75, 0.65],
        [0.65, 0.75],
    ],
    dtype=torch.double,
)
REF_POINT = [0.0, 0.0]

BOX_LO = (0.265, 0.665)
BOX_HI = (0.335, 0.735)


def fit_model():
    model = SingleTaskGP(
        TRAIN_X,
        TRAIN_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=2),
    ).double()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


# For the theorem portfolio, fit ONCE so that the generic EHVI route and the
# posterior-theorem route refer to exactly the same stored binary64 GP state.
MODEL = fit_model()
REF_TENSOR = torch.tensor(REF_POINT, dtype=torch.double)
PARTITIONING = FastNondominatedPartitioning(ref_point=REF_TENSOR, Y=TRAIN_Y)


def engineering_model(p):
    """Ordinary BoTorch analytic EHVI for the fixed fitted model."""
    ehvi = ExpectedHypervolumeImprovement(
        model=MODEL,
        ref_point=REF_POINT,
        partitioning=PARTITIONING,
    )
    x = torch.tensor([[[p[0], p[1]]]], dtype=torch.double)
    return ehvi(x)[0]


def posterior_four(p):
    """Return mu1,var1,mu2,var2 using the same fitted BoTorch model."""
    x = torch.tensor([[[p[0], p[1]]]], dtype=torch.double)
    posterior = MODEL.posterior(x)
    return torch.stack(
        [
            posterior.mean[0, 0, 0],
            posterior.variance[0, 0, 0],
            posterior.mean[0, 0, 1],
            posterior.variance[0, 0, 1],
        ]
    )


def _psi_cert(cert, threshold, mu_point, sigma_point):
    t = cert.point(float(threshold))
    z = (mu_point - t) / sigma_point
    return (
        (mu_point - t) * cert.standard_normal_cdf(z)
        + sigma_point * cert.standard_normal_pdf(z)
    )


def _psi_upper(cert, threshold, mu_range, sigma_range):
    """User theorem A: Psi increases in both mu and sigma."""
    return _psi_cert(
        cert,
        threshold,
        cert.point(mu_range.upper),
        cert.point(sigma_range.upper),
    )


def _finite_band_upper(cert, a, b, mu_range, sigma_range):
    """User theorem B: upper endpoint of D(a,b;mu,sigma)."""
    cert.require(float(b) > float(a), "finite Gaussian band has b>a")

    midpoint = 0.5 * (float(a) + float(b))
    mu_upper = float(mu_range.upper)

    # D is increasing in mu, so its maximum is attained at mu_upper.
    # At that fixed mu the midpoint rule selects the sigma endpoint giving the
    # maximum finite-band expectation.
    if mu_upper < midpoint:
        sigma_for_upper = float(sigma_range.upper)
    else:
        sigma_for_upper = float(sigma_range.lower)

    mu = cert.point(mu_upper)
    sigma = cert.point(sigma_for_upper)
    return _psi_cert(cert, a, mu, sigma) - _psi_cert(cert, b, mu, sigma)


@application_theorem(
    name="BoTorch EHVI upper bound from Gaussian cell monotonicity",
    target="EHVI",
    theorem_status="user_asserted",
)
def botorch_ehvi_theorem_upper(box, cert):
    posterior = cert.enclose(posterior_four, box)
    mu1, var1, mu2, var2 = posterior[0], posterior[1], posterior[2], posterior[3]

    cert.require(var1.lower > 0.0, "objective-1 posterior variance is positive")
    cert.require(var2.lower > 0.0, "objective-2 posterior variance is positive")
    sigma1 = cert.sqrt(var1)
    sigma2 = cert.sqrt(var2)

    # The fixed Pareto front is
    #   (0.25,1.00), (0.40,0.85), (0.65,0.75),
    #   (0.75,0.65), (1.00,0.35)
    # under maximization with reference point (0,0).  Its non-dominated region
    # yields six disjoint q=1 EHVI cells.
    horizontal_upper = [
        _finite_band_upper(cert, 0.00, 0.25, mu1, sigma1),
        _finite_band_upper(cert, 0.25, 0.40, mu1, sigma1),
        _finite_band_upper(cert, 0.40, 0.65, mu1, sigma1),
        _finite_band_upper(cert, 0.65, 0.75, mu1, sigma1),
        _finite_band_upper(cert, 0.75, 1.00, mu1, sigma1),
        _psi_upper(cert, 1.00, mu1, sigma1),
    ]
    vertical_upper = [
        _psi_upper(cert, 1.00, mu2, sigma2),
        _psi_upper(cert, 0.85, mu2, sigma2),
        _psi_upper(cert, 0.75, mu2, sigma2),
        _psi_upper(cert, 0.65, mu2, sigma2),
        _psi_upper(cert, 0.35, mu2, sigma2),
        _psi_upper(cert, 0.00, mu2, sigma2),
    ]

    upper_cells = [
        horizontal_upper[i] * vertical_upper[i] for i in range(6)
    ]
    return cert.sum(upper_cells)


def dense_numeric_range(n_per_dim=41):
    """NON-RIGOROUS diagnostic range using the same fixed fitted model."""
    ehvi = ExpectedHypervolumeImprovement(
        model=MODEL,
        ref_point=REF_POINT,
        partitioning=PARTITIONING,
    )
    xs = torch.linspace(BOX_LO[0], BOX_HI[0], n_per_dim)
    ys = torch.linspace(BOX_LO[1], BOX_HI[1], n_per_dim)
    grid = torch.cartesian_prod(xs, ys).reshape(-1, 1, 2)
    with torch.no_grad():
        values = ehvi(grid)
    return float(values.min()), float(values.max())


if __name__ == "__main__":
    interval_box = [
        Interval.enclosing(str(BOX_LO[0]), str(BOX_HI[0])),
        Interval.enclosing(str(BOX_LO[1]), str(BOX_HI[1])),
    ]
    application_box = ApplicationBox.from_intervals(interval_box)

    generic = autoia(engineering_model, interval_box)
    generic_lo = float(generic.lower[0])
    generic_hi = float(generic.upper[0])

    theorem = botorch_ehvi_theorem_upper(application_box)
    theorem_hi = float(theorem.value)

    # Two independent facts are combined:
    #   1. generic AutoIA encloses the unchanged BoTorch computation;
    #   2. the user theorem independently upper-bounds the same EHVI quantity.
    # EHVI >= 0 supplies the application-known lower bound.
    final_lo = max(0.0, generic_lo)
    final_hi = min(generic_hi, theorem_hi)

    print("generic BoTorch/AutoIA EHVI =", [generic_lo, generic_hi])
    print("Gaussian theorem upper      =", theorem_hi)
    print("EHVI >= 0 theorem lower     =", 0.0)
    print("final theorem portfolio     =", [final_lo, final_hi])
    print("theorem status              =", theorem.theorem_status)
    print("checked assumptions         =", theorem.assumptions_checked)

    sampled_lo, sampled_hi = dense_numeric_range(n_per_dim=41)
    print(
        "dense sampled EHVI range (non-rigorous diagnostic) =",
        [sampled_lo, sampled_hi],
    )
    print("sampled values contained    =", final_lo <= sampled_lo and sampled_hi <= final_hi)

    assert generic.machine_rigorous
    assert theorem.machine_rigorous
    assert final_lo <= sampled_lo and sampled_hi <= final_hi
    assert final_hi < generic_hi
