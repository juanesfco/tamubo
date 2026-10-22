"""Validated elementary-function Taylor infrastructure."""

from .taylor import (
    SUPPORTED_ELEMENTARY_FUNCTIONS,
    TaylorData,
    taylor_coefficients,
    taylor_remainder_radius,
    build_taylor_data,
)

__all__ = [
    "SUPPORTED_ELEMENTARY_FUNCTIONS",
    "TaylorData",
    "taylor_coefficients",
    "taylor_remainder_radius",
    "build_taylor_data",
]
