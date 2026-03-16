from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .backend import BackendInfo, resolve_backend


@dataclass
class BOResult:
    """Unified BO result payload for benchmark workflows."""

    X: np.ndarray  # (N, d)
    y: np.ndarray | None = None  # (N,)
    backend: BackendInfo | None = None
    log: dict | None = None


def _normalize_inputs(
    X0: np.ndarray,
    bounds: np.ndarray,
    *,
    validation: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    X = np.asarray(X0, dtype=np.float64)
    search_bounds = np.asarray(bounds, dtype=np.float64)

    if validation:
        if X.ndim != 2:
            raise ValueError(f"X0 must be 2D with shape (N0, d), got {X.shape}")
        if search_bounds.ndim != 2 or search_bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {search_bounds.shape}")

    dim = search_bounds.shape[0]
    if validation and X.shape[1] != dim:
        raise ValueError(
            f"X0 second dimension ({X.shape[1]}) must match bounds dimension ({dim})"
        )
    return X, search_bounds, dim


def _unit_cube_bounds(dim: int) -> np.ndarray:
    """Return canonical [0, 1]^d bounds."""
    return np.column_stack((np.zeros(dim, dtype=np.float64), np.ones(dim, dtype=np.float64)))


def _to_unit_cube(
    X: np.ndarray,
    bounds: np.ndarray,
    *,
    validation: bool = True,
) -> np.ndarray:
    """Affine-map points from physical bounds to [0, 1]^d."""
    X_arr = np.asarray(X, dtype=np.float64)
    search_bounds = np.asarray(bounds, dtype=np.float64)
    X_eval = X_arr.reshape(1, -1) if X_arr.ndim == 1 else X_arr

    lower = search_bounds[:, 0]
    upper = search_bounds[:, 1]
    span = upper - lower

    if validation and X_eval.shape[1] != search_bounds.shape[0]:
        raise ValueError(
            f"Point dimension ({X_eval.shape[1]}) must match bounds dimension "
            f"({search_bounds.shape[0]})."
        )
    if np.any(~np.isfinite(search_bounds)):
        raise ValueError("Unit-cube normalization requires finite bounds.")
    if np.any(span <= 0.0):
        raise ValueError("Unit-cube normalization requires strictly increasing bounds.")

    X_unit = (X_eval - lower) / span
    return X_unit.reshape(-1) if X_arr.ndim == 1 else X_unit


def _from_unit_cube(
    X: np.ndarray,
    bounds: np.ndarray,
    *,
    validation: bool = True,
) -> np.ndarray:
    """Affine-map points from [0, 1]^d back to the physical bounds."""
    X_arr = np.asarray(X, dtype=np.float64)
    search_bounds = np.asarray(bounds, dtype=np.float64)
    X_eval = X_arr.reshape(1, -1) if X_arr.ndim == 1 else X_arr

    lower = search_bounds[:, 0]
    upper = search_bounds[:, 1]
    span = upper - lower

    if validation and X_eval.shape[1] != search_bounds.shape[0]:
        raise ValueError(
            f"Point dimension ({X_eval.shape[1]}) must match bounds dimension "
            f"({search_bounds.shape[0]})."
        )
    if np.any(~np.isfinite(search_bounds)):
        raise ValueError("Unit-cube denormalization requires finite bounds.")
    if np.any(span <= 0.0):
        raise ValueError("Unit-cube denormalization requires strictly increasing bounds.")

    X_physical = lower + X_eval * span
    return X_physical.reshape(-1) if X_arr.ndim == 1 else X_physical


def _normalize_problem_to_unit_cube(
    X0: np.ndarray,
    bounds: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    *,
    validation: bool = True,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Build a unit-cube search problem while preserving objective evaluation in physical coordinates.
    """
    X = np.asarray(X0, dtype=np.float64)
    physical_bounds = np.asarray(bounds, dtype=np.float64)
    X_unit = _to_unit_cube(X, physical_bounds, validation=validation)
    unit_bounds = _unit_cube_bounds(physical_bounds.shape[0])

    def f_unit(X_unit_eval: np.ndarray) -> np.ndarray:
        return f(_from_unit_cube(X_unit_eval, physical_bounds, validation=False))

    return X_unit, unit_bounds, f_unit, physical_bounds.copy()


def _evaluate_objective(f: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
    """Evaluate objective and return a flattened array."""
    X = np.asarray(X, dtype=np.float64)
    X_eval = X.reshape(1, -1) if X.ndim == 1 else X
    return np.asarray(f(X_eval), dtype=np.float64).ravel()


def _build_cartesian_grid(
    bounds: np.ndarray,
    grid_resolution: int,
    *,
    validation: bool = True,
) -> np.ndarray:
    resolution = int(grid_resolution)
    if validation and resolution < 2:
        raise ValueError(f"grid_resolution must be >= 2, got {resolution}")

    axes = [
        np.linspace(lower, upper, resolution, dtype=np.float64)
        for lower, upper in np.asarray(bounds, dtype=np.float64)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def _init_log(logMask: bool) -> dict | None:
    return {} if logMask else None


def _as_result(X: np.ndarray, y: np.ndarray, *, backend: str = "numpy", log: dict | None = None) -> BOResult:
    return BOResult(
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        backend=resolve_backend(backend),
        log=log,
    )
