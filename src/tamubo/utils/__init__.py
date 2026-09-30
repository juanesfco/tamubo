"""
tamubo.utils public API.

Exports:
- BackendName, SelectedBackend, BackendInfo, has_cupy, resolve_backend,
- get_array_module, to_numpy
- BOResult, _as_result, _build_cartesian_grid, _evaluate_objective, _from_unit_cube,
- _init_log, _normalize_inputs, _normalize_problem_to_unit_cube, _to_unit_cube,
- _unit_cube_bounds
"""
from .backend import (
    BackendInfo,
    BackendName,
    SelectedBackend,
    get_array_module,
    has_cupy,
    resolve_backend,
    to_numpy,
)

from .common import (
    BOResult,
    _as_result,
    _build_cartesian_grid,
    _evaluate_objective,
    _from_unit_cube,
    _init_log,
    _normalize_inputs,
    _normalize_problem_to_unit_cube,
    _to_unit_cube,
    _unit_cube_bounds,
)

__all__ = [
    "BackendName",
    "SelectedBackend",
    "BackendInfo",
    "has_cupy",
    "resolve_backend",
    "get_array_module",
    "to_numpy",
    "BOResult",
    "_as_result",
    "_build_cartesian_grid",
    "_evaluate_objective",
    "_from_unit_cube",
    "_init_log",
    "_normalize_inputs",
    "_normalize_problem_to_unit_cube",
    "_to_unit_cube",
    "_unit_cube_bounds",
]
