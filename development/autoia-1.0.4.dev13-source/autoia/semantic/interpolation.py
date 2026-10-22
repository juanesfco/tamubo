"""Validated semantics for fixed-data rectilinear multilinear interpolation.

The V1 semantic deliberately targets SciPy's ``RegularGridInterpolator`` with
``method='linear'``.  Grid coordinates and table values are fixed ordinary
numeric data; uncertainty enters only through the query coordinates.

A query that crosses grid-cell boundaries is handled by ordinary AutoIA
path-aware comparisons.  On each feasible path the selected cell is fixed and
the interpolant is evaluated as the exact tensor-product multilinear formula in
the dependency-bearing query coordinates.  No early scalar intervalization of
the interpolated value is performed.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ..arrays import contains_validated as array_contains_validated


def _contains_validated(obj: Any) -> bool:
    try:
        return array_contains_validated(obj)
    except Exception:
        return False


def _as_real_finite_array(name: str, value) -> np.ndarray:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise NotImplementedError(
            f"AutoIA validated RegularGridInterpolator currently requires real {name}."
        )
    try:
        out = np.asarray(arr, dtype=float)
    except Exception as exc:
        raise NotImplementedError(
            f"AutoIA validated RegularGridInterpolator requires fixed numeric {name}."
        ) from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"AutoIA validated RegularGridInterpolator requires finite {name}."
        )
    return out


def _normalize_grid_and_values(points, values):
    grids = []
    flips = []
    for axis, p in enumerate(points):
        g = _as_real_finite_array(f"grid axis {axis}", p)
        if g.ndim != 1 or g.size < 2:
            raise ValueError(
                "AutoIA validated RegularGridInterpolator requires each grid axis "
                "to be one-dimensional with at least two points."
            )
        d = np.diff(g)
        if np.all(d > 0.0):
            grids.append(g.copy())
            flips.append(False)
        elif np.all(d < 0.0):
            grids.append(g[::-1].copy())
            flips.append(True)
        else:
            raise ValueError(
                "AutoIA validated RegularGridInterpolator requires each grid axis "
                "to be strictly monotone."
            )

    vals = _as_real_finite_array("table values", values)
    ngrid = len(grids)
    expected = tuple(len(g) for g in grids)
    if vals.ndim < ngrid or tuple(vals.shape[:ngrid]) != expected:
        raise ValueError(
            "AutoIA validated RegularGridInterpolator table shape does not match "
            "the supplied rectilinear grid."
        )
    for axis, flip in enumerate(flips):
        if flip:
            vals = np.flip(vals, axis=axis)
    return tuple(grids), np.array(vals, copy=True)


def _validated_scalar_present(x) -> bool:
    return _contains_validated(x)


def _cell_or_outside(x, grid):
    """Return ``(cell_index, outside)`` using path-aware scalar comparisons.

    ``outside`` is -1 below the grid, +1 above the grid, and 0 in-domain.
    Cell indices refer to the normalized ascending grid.
    """
    if x < grid[0]:
        return 0, -1
    if x > grid[-1]:
        return len(grid) - 2, +1
    # The right endpoint belongs to the last cell.  Using <= makes exact grid
    # boundaries explicit branch surfaces under AutoIA path-aware execution.
    for j in range(len(grid) - 1):
        if x <= grid[j + 1]:
            return j, 0
    return len(grid) - 2, 0  # defensive; x <= grid[-1] above


def _multilinear_point(coords, grids, values, *, bounds_error, fill_value):
    ndim = len(grids)
    cells = []
    outside = False
    for x, g in zip(coords, grids):
        j, status = _cell_or_outside(x, g)
        cells.append(j)
        outside = outside or status != 0

    if outside:
        if bounds_error:
            raise ValueError(
                "RegularGridInterpolator query leaves the fixed grid while "
                "bounds_error=True; the engineering model is not defined on "
                "the full AutoIA input box."
            )
        if fill_value is not None:
            fv = float(fill_value)
            if not np.isfinite(fv):
                raise ValueError(
                    "RegularGridInterpolator reaches an out-of-bounds path with "
                    "a non-finite fill_value; AutoIA certifies finite real outputs only."
                )
            trailing = values.shape[ndim:]
            if not trailing:
                return fv
            return np.full(trailing, fv, dtype=float)
        # fill_value=None means linear extrapolation.  The first/last cell chosen
        # by _cell_or_outside supplies the same affine extension convention.

    t = []
    for x, g, j in zip(coords, grids, cells):
        a = float(g[j])
        b = float(g[j + 1])
        t.append((x - a) / (b - a))

    trailing = values.shape[ndim:]
    out = None if not trailing else np.empty(trailing, dtype=object)
    if trailing:
        out[...] = 0.0
    else:
        out = 0.0

    for bits in product((0, 1), repeat=ndim):
        weight = 1.0
        index = []
        for bit, td, j in zip(bits, t, cells):
            weight = weight * (td if bit else (1.0 - td))
            index.append(j + bit)
        corner = values[tuple(index)]
        out = out + weight * corner
    return out


def _validated_queries(xi, ndim):
    """Normalize common SciPy query forms without discarding semantic objects.

    Returns ``(points, mode, leading_shape)`` where ``points`` is a list of
    coordinate lists.  ``mode='tuple-scalar'`` reproduces SciPy's convenient
    scalar tuple form; other single/batched forms return an array-like result.
    """
    if isinstance(xi, tuple) and len(xi) == ndim:
        if all(np.asarray(v, dtype=object).ndim == 0 for v in xi):
            return [list(xi)], "tuple-scalar", ()
        raise NotImplementedError(
            "AutoIA validated RegularGridInterpolator currently supports tuple "
            "queries only when the tuple contains one scalar coordinate per dimension."
        )

    obj = np.asarray(xi, dtype=object)
    if obj.ndim == 1:
        if obj.shape[0] != ndim:
            raise ValueError(
                "RegularGridInterpolator query must provide one coordinate per grid dimension."
            )
        return [list(obj)], "single-array", (1,)
    if obj.ndim >= 2 and obj.shape[-1] == ndim:
        lead = obj.shape[:-1]
        flat = obj.reshape((-1, ndim))
        return [list(row) for row in flat], "batch", lead
    raise ValueError(
        "AutoIA validated RegularGridInterpolator requires xi with final dimension "
        "equal to the number of grid axes."
    )


def validated_regular_grid_call(*, xi, grids, values, method, bounds_error,
                                fill_value, nu=None):
    """Evaluate a validated fixed-data multilinear interpolator query."""
    if method != "linear":
        raise NotImplementedError(
            "AutoIA validated scipy.interpolate.RegularGridInterpolator currently "
            "supports only method='linear'."
        )
    if nu is not None:
        raise NotImplementedError(
            "AutoIA validated RegularGridInterpolator does not support derivative nu= semantics."
        )
    if not _contains_validated(xi):
        raise RuntimeError("validated_regular_grid_call requires an AutoIA-dependent query.")

    points, mode, lead = _validated_queries(xi, len(grids))
    results = [
        _multilinear_point(
            p, grids, values, bounds_error=bool(bounds_error), fill_value=fill_value
        )
        for p in points
    ]
    trailing = values.shape[len(grids):]

    if mode == "tuple-scalar":
        return results[0]

    if not trailing:
        arr = np.asarray(results, dtype=object)
        return arr.reshape(lead)

    stacked = np.empty((len(results),) + trailing, dtype=object)
    for i, result in enumerate(results):
        stacked[i] = result
    return stacked.reshape(tuple(lead) + trailing)


class RegularGridInterpolatorProxy:
    """Private AutoIA wrapper around one real SciPy RGI instance."""

    __slots__ = (
        "_target", "_grids", "_values", "_method", "_bounds_error", "_fill_value"
    )

    def __init__(self, target):
        self._target = target
        self._grids, self._values = _normalize_grid_and_values(target.grid, target.values)
        self._method = str(target.method)
        self._bounds_error = bool(target.bounds_error)
        self._fill_value = target.fill_value

    def __call__(self, xi, method=None, *, nu=None):
        if not _contains_validated(xi):
            if method is None:
                return self._target(xi, nu=nu)
            return self._target(xi, method=method, nu=nu)
        chosen = self._method if method is None else str(method)
        return validated_regular_grid_call(
            xi=xi,
            grids=self._grids,
            values=self._values,
            method=chosen,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
            nu=nu,
        )

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __dir__(self):
        return dir(self._target)


__all__ = ["RegularGridInterpolatorProxy", "validated_regular_grid_call"]
