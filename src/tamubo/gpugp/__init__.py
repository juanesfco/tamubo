"""Gaussian process based regression."""

# Author: Juan E Florez-Coronel

__all__ = ["gpr", "gp_posterior"]


def __getattr__(name):
    if name == "gpr":
        from ._gp import gpr
        return gpr
    if name == "gp_posterior":
        from .posterior import gp_posterior
        return gp_posterior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
