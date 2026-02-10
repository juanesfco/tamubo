"""Compatibility wrapper for vectorized cupynumeric bounds."""

from .vectorized_bounds import ei_bounds, mu_bounds, rbf_k_bounds, sigma_bounds

__all__ = [
    "rbf_k_bounds",
    "mu_bounds",
    "sigma_bounds",
    "ei_bounds",
]
