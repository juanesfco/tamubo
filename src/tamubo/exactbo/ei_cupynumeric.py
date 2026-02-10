"""Compatibility wrapper for vectorized cupynumeric EI utilities."""

from .vectorized_ei import erf_approx, expected_improvement, norm_cdf, norm_pdf

__all__ = [
    "erf_approx",
    "norm_cdf",
    "norm_pdf",
    "expected_improvement",
]
