"""Public validated enclosure representations.

Ordinary users normally construct these through :func:`autoia.autoia`.  They
are exposed for inspection, diagnostics, and research extensions.
"""

from .enclosures import (
    ValidatedValue,
    PolynomialEnclosure,
    IntervalEnclosure,
    SharedGeneratorPolynomialEnclosure,
    PiecewiseEnclosure,
)

__all__ = [
    "ValidatedValue",
    "PolynomialEnclosure",
    "IntervalEnclosure",
    "SharedGeneratorPolynomialEnclosure",
    "PiecewiseEnclosure",
]
