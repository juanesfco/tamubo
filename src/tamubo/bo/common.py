from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tamubo.utils import BackendInfo, resolve_backend


@dataclass
class BOResult:
    """Unified BO result payload for benchmark workflows."""

    X: np.ndarray  # (N, d)
    y: np.ndarray  # (N,)
    backend: BackendInfo
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


def _as_result(X: np.ndarray, y: np.ndarray, log: dict | None) -> BOResult:
    return BOResult(
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        backend=resolve_backend("numpy"),
        log=log,
    )

