"""
Example 37 -- composed NumPy elementary-function stress test.

Purpose
-------
Example 36 checks the newly supported functions one at a time.  This example
checks them in combinations that share the same uncertain physical variables.

The dense grid at the end is ONLY a non-rigorous diagnostic.  The AutoIA result
is the certificate.

Requires AutoIA >= 1.0.4.dev6.
"""

import numpy as np

from autoia import Interval, autoia


def engineering_model(p):
    """
    A small nonlinear engineering-style calculation.

    Inputs
    ------
    p[0] = x
    p[1] = q

    The calculation deliberately uses all six newly supported operations:
        np.tanh
        np.sinh
        np.arctan
        np.abs
        np.log1p
        np.cosh

    It also includes one exact mathematical identity residual:

        sinh(z) - tanh(z)*cosh(z) = 0

    for every real z.

    That output is intentionally included as a dependency-quality stress test.
    A rigorous interval containing zero is correct; the width tells us how much
    dependence was lost in the automatic representation.
    """
    x = p[0]
    q = p[1]

    # Shared nonlinear combinations of the same two uncertain inputs.
    z = 0.75 * x - 0.55 * q
    w = 0.40 * x + 0.30 * q

    # A smooth saturating response.
    saturation = np.tanh(z)

    # A hyperbolic transport-like contribution.
    transport = np.sinh(w) / (1.0 + np.cosh(w))

    # A bounded angular correction.
    angle = np.arctan(x / (1.0 + np.abs(q)))

    # A positive loss-like quantity.  Its log1p argument is >= 0, so the
    # mathematical domain x > -1 is automatically satisfied.
    mismatch = np.abs(x - q)
    loss = np.log1p(mismatch)

    # A smooth even energy-like term.
    energy = np.cosh(0.5 * z) - 1.0

    # Mixed engineering-style outputs.
    response_1 = (
        1.20
        + 0.80 * saturation
        + 0.35 * transport
        - 0.20 * angle
        + 0.15 * loss
        + 0.10 * energy
    )

    response_2 = (
        np.log1p(0.25 * np.abs(z))
        + 0.30 * np.arctan(w)
        + 0.08 * np.cosh(z)
        - 0.05 * np.sinh(q)
    )

    # Exact identity for all real z:
    #
    #     tanh(z) = sinh(z)/cosh(z)
    #
    # and cosh(z) > 0, hence
    #
    #     sinh(z) - tanh(z)*cosh(z) = 0.
    #
    # This is not needed by the engineering calculation.  It is present solely
    # to expose dependency loss in automatic range propagation.
    identity_residual = (
        np.sinh(z)
        - np.tanh(z) * np.cosh(z)
    )

    return np.array(
        [response_1, response_2, identity_residual],
        dtype=object,
    )


def point_model(x, q):
    """Ordinary floating-point version used only for dense diagnostics."""
    z = 0.75 * x - 0.55 * q
    w = 0.40 * x + 0.30 * q

    saturation = np.tanh(z)
    transport = np.sinh(w) / (1.0 + np.cosh(w))
    angle = np.arctan(x / (1.0 + np.abs(q)))
    mismatch = np.abs(x - q)
    loss = np.log1p(mismatch)
    energy = np.cosh(0.5 * z) - 1.0

    response_1 = (
        1.20
        + 0.80 * saturation
        + 0.35 * transport
        - 0.20 * angle
        + 0.15 * loss
        + 0.10 * energy
    )

    response_2 = (
        np.log1p(0.25 * np.abs(z))
        + 0.30 * np.arctan(w)
        + 0.08 * np.cosh(z)
        - 0.05 * np.sinh(q)
    )

    identity_residual = np.sinh(z) - np.tanh(z) * np.cosh(z)

    return np.array(
        [response_1, response_2, identity_residual],
        dtype=float,
    )


def dense_diagnostic(nx=301, nq=301):
    """
    Dense floating-point sample of the box.

    NON-RIGOROUS: this does not prove extrema.  It is only useful for judging
    the sharpness of the rigorous AutoIA intervals.
    """
    xs = np.linspace(-0.45, 0.65, nx)
    qs = np.linspace(-0.35, 0.55, nq)

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)

    for x in xs:
        for q in qs:
            y = point_model(x, q)
            lo = np.minimum(lo, y)
            hi = np.maximum(hi, y)

    return lo, hi


if __name__ == "__main__":
    # Both intervals cross zero.  In particular, x-q also crosses zero, so
    # np.abs exercises its genuinely nonsmooth case rather than a sign-definite
    # shortcut.
    box = [
        Interval.enclosing("-0.45", "0.65"),
        Interval.enclosing("-0.35", "0.55"),
    ]

    print("AutoIA composed-function certificate")
    result = autoia(engineering_model, box)
    print(result)

    names = [
        "response_1",
        "response_2",
        "identity_residual (exactly 0 mathematically)",
    ]

    print("\nCertified intervals")
    for i, name in enumerate(names):
        L = float(result.lower[i])
        U = float(result.upper[i])
        print(f"  {name:48s}: [{L:.12g}, {U:.12g}]  width={U-L:.12g}")

    print("\nDense sampled ranges (NON-RIGOROUS diagnostic)")
    dense_lo, dense_hi = dense_diagnostic()
    for i, name in enumerate(names):
        print(
            f"  {name:48s}: "
            f"[{dense_lo[i]:.12g}, {dense_hi[i]:.12g}]  "
            f"width={dense_hi[i]-dense_lo[i]:.12g}"
        )

    print("\nWidth inflation relative to dense sample")
    for i, name in enumerate(names[:2]):
        cert_width = float(result.upper[i] - result.lower[i])
        dense_width = float(dense_hi[i] - dense_lo[i])
        inflation = cert_width / dense_width if dense_width > 0.0 else np.inf
        print(f"  {name:48s}: {inflation:.6g}x")

    print("\nIdentity-residual interpretation")
    print("  Exact mathematical range : [0, 0]")
    print(
        "  AutoIA range             : "
        f"[{float(result.lower[2]):.12g}, {float(result.upper[2]):.12g}]"
    )
    print(
        "  Any nonzero width here is dependency overestimation, not uncertainty "
        "in the true function."
    )

    # Basic rigorous sanity checks.
    assert result.machine_rigorous
    assert float(result.lower[2]) <= 0.0 <= float(result.upper[2])

    for i in range(3):
        assert float(result.lower[i]) <= dense_lo[i]
        assert dense_hi[i] <= float(result.upper[i])

    print("\nAll dense diagnostic samples are contained in the AutoIA certificates.")
