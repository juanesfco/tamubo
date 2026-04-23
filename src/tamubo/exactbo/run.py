from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import time as pytime

import numpy as np

from tamubo.acquisition_functions import expected_improvement
from tamubo.gpugp.posterior import gp_posterior
from tamubo.utils import (
    BOResult,
    BackendInfo,
    BackendName,
    _evaluate_objective,
    _from_unit_cube,
    _init_log,
    _normalize_inputs,
    _normalize_problem_to_unit_cube,
)

from ._cupynumeric import cp, require_cupynumeric_backend
from .bounds import ei_bounds, mu_bounds, rbf_k_bounds, sigma_bounds
from .partition import split_boxes


@dataclass(frozen=True)
class SklearnGPState:
    X_train: object
    alpha: object
    L: object
    length_scale: object
    sigma_f_squared: float
    sigma_n_squared: float
    y_train_mean: float
    y_train_std: float
    y_min_scaled: float


def _normalize_epsilon(epsilon: np.ndarray | float, dim: int) -> np.ndarray:
    """Normalize epsilon to a per-dimension NumPy array."""
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=np.float64)
    if eps.shape == (dim,):
        return eps
    raise ValueError(f"epsilon must be scalar or shape ({dim},), got {eps.shape}")


def _centered_latin_hypercube_unit(n_points: int, dim: int) -> np.ndarray:
    """Return a deterministic centered Latin hypercube in ``[0, 1]^dim``."""
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    centers = (np.arange(n_points, dtype=np.float64) + 0.5) / float(n_points)
    perm_ids = np.arange(n_points, dtype=np.int64)
    lhs = np.empty((n_points, dim), dtype=np.float64)

    for j in range(dim):
        step = 2 * j + 1
        while math.gcd(step, n_points) != 1:
            step += 2
        lhs[:, j] = centers[(perm_ids * step + j) % n_points]

    return lhs


def _scalar_int(value) -> int:
    return int(np.asarray(value).item())


def _scalar_float(value) -> float:
    return float(np.asarray(value).item())


def _scalar_bool(value) -> bool:
    return bool(np.asarray(value).item())


def _extract_sklearn_gp_state(X: np.ndarray, gp) -> SklearnGPState:
    kernel = gp.kernel_
    sigma_f_squared = float(kernel.k1.k1.constant_value)
    sigma_n_squared = float(kernel.k2.noise_level)
    length_scale = cp.asarray(kernel.k1.k2.length_scale, dtype=cp.float64)

    return SklearnGPState(
        X_train=cp.asarray(X, dtype=cp.float64),
        alpha=cp.asarray(np.asarray(gp.alpha_, dtype=np.float64).reshape(-1), dtype=cp.float64),
        L=cp.asarray(np.asarray(gp.L_, dtype=np.float64), dtype=cp.float64),
        length_scale=length_scale,
        sigma_f_squared=sigma_f_squared,
        sigma_n_squared=sigma_n_squared,
        y_train_mean=float(np.asarray(gp._y_train_mean, dtype=np.float64).reshape(-1)[0]),
        y_train_std=float(np.asarray(gp._y_train_std, dtype=np.float64).reshape(-1)[0]),
        y_min_scaled=float(np.min(np.asarray(gp.y_train_, dtype=np.float64))),
    )


def _predict_standardized_posterior(points, gp, state: SklearnGPState):
    points = cp.asarray(points, dtype=cp.float64)
    return gp_posterior(
        points,
        X_train=state.X_train,
        alpha=state.alpha,
        L=state.L,
        length_scale=state.length_scale,
        sigma_f_squared=state.sigma_f_squared,
        sigma_n_squared=state.sigma_n_squared,
        y_train_mean=state.y_train_mean,
        y_train_std=state.y_train_std,
        scaled_output=True,
        return_std=True,
        backend="cupynumeric",
        validation=False,
    )


def _compute_ei_upper_bounds(
    bounds_L_target,
    bounds_U_target,
    state: SklearnGPState,
    dim: int,
    *,
    validation: bool,
):
    n_target = _scalar_int(bounds_L_target.shape[0])
    n_train = _scalar_int(state.alpha.shape[0])

    K_lo = cp.empty((n_target, n_train), dtype=cp.float64)
    K_hi = cp.empty((n_target, n_train), dtype=cp.float64)
    for i in range(n_train):
        K_lo[:, i], K_hi[:, i] = rbf_k_bounds(
            bounds_L_target,
            bounds_U_target,
            state.X_train[i],
            n_target,
            dim,
            state.sigma_f_squared,
            state.length_scale,
            backend="cupynumeric",
            validation=validation,
        )

    mu_lo, mu_hi = mu_bounds(
        state.alpha,
        K_lo,
        K_hi,
        n_target,
        n_train,
        y_train_mean=state.y_train_mean,
        y_train_std=state.y_train_std,
        scaled_output=True,
        backend="cupynumeric",
        validation=validation,
    )
    sig_lo, sig_hi = sigma_bounds(
        K_lo,
        K_hi,
        state.L,
        n_target,
        n_train,
        state.sigma_f_squared,
        y_train_std=state.y_train_std,
        scaled_output=True,
        backend="cupynumeric",
        validation=validation,
    )
    _, ei_hi = ei_bounds(
        mu_lo,
        mu_hi,
        sig_lo,
        sig_hi,
        n_target,
        state.y_min_scaled,
        backend="cupynumeric",
        validation=validation,
    )
    return ei_hi


def _sample_boxes_best_ei(boxes_L, boxes_U, gp, state: SklearnGPState, lhs_unit_design):
    n_boxes = _scalar_int(boxes_L.shape[0])
    dim = _scalar_int(boxes_L.shape[1])
    n_samples = _scalar_int(lhs_unit_design.shape[0])

    if n_boxes == 0:
        return (
            cp.empty((0, dim), dtype=cp.float64),
            cp.empty((0,), dtype=cp.float64),
        )

    widths = boxes_U - boxes_L
    sampled_points = boxes_L[:, cp.newaxis, :] + lhs_unit_design[cp.newaxis, :, :] * widths[:, cp.newaxis, :]
    flat_points = sampled_points.reshape((n_boxes * n_samples, dim))

    mu, sigma = _predict_standardized_posterior(flat_points, gp, state)
    mu = cp.asarray(mu, dtype=cp.float64).reshape((n_boxes, n_samples))
    sigma = cp.asarray(sigma, dtype=cp.float64).reshape((n_boxes, n_samples))

    sigma_latent = cp.sqrt(cp.maximum(sigma * sigma - state.sigma_n_squared, 1e-12))
    ei = expected_improvement(
        mu,
        sigma_latent,
        state.y_min_scaled,
        backend="cupynumeric",
    )

    best_idx = cp.argmax(ei, axis=1).reshape((n_boxes, 1))
    best_ei = cp.take_along_axis(ei, best_idx, axis=1).reshape((n_boxes,))
    gather_idx = cp.broadcast_to(best_idx[:, :, cp.newaxis], (n_boxes, 1, dim))
    best_points = cp.take_along_axis(sampled_points, gather_idx, axis=1).reshape((n_boxes, dim))
    return best_points, best_ei


def _build_partition_result(
    best_x,
    ei_best_scaled: float,
    *,
    backend_info: BackendInfo,
    log: dict | None,
    start_time: float | None,
    y_train_std: float,
) -> BOResult:
    if log is not None and start_time is not None:
        log["time"] = pytime.perf_counter() - start_time
        log["ei_max"] = float(ei_best_scaled * y_train_std)
        log["ei_max_scaled"] = float(ei_best_scaled)

    return BOResult(
        X=np.asarray(best_x, dtype=np.float64),
        backend=backend_info,
        log=log,
    )


def exactbo(
    X0: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray | float,
    epsilon_ei: float,
    gp,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
    normalize_to_unit_cube: bool = False,
) -> BOResult:
    """
    Run ExactBO with the cuPyNumeric backend.

    ``backend="auto"`` is still accepted for compatibility, but it must resolve
    to ``"cupynumeric"``.
    """
    backend_info = require_cupynumeric_backend(backend)
    X, search_bounds, dim = _normalize_inputs(X0, bounds, validation=validation)

    iterations = int(max_iters)
    if validation and iterations < 0:
        raise ValueError(f"max_iters must be >= 0, got {iterations}")

    objective = f
    physical_bounds = None
    if normalize_to_unit_cube:
        X, search_bounds, objective, physical_bounds = _normalize_problem_to_unit_cube(
            X,
            search_bounds,
            f,
            validation=validation,
        )

    epsilon_X = _normalize_epsilon(epsilon_X, dim)
    log = _init_log(logMask)
    y = _evaluate_objective(objective, X)

    for iteration in range(iterations):
        X_display = (
            _from_unit_cube(X, physical_bounds, validation=False)
            if physical_bounds is not None
            else X
        )
        y = _evaluate_objective(objective, X)

        if verbose:
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"Current training data: \nX: {X_display}, \ny: {y}")

        if logMask:
            log[f"i{iteration}"] = {"X": X_display.copy(), "y": y.copy()}

        gp.fit(X, y)
        if verbose:
            print(f"GP kernel after fitting: {gp.kernel_}")

        partitioning_result = exactbo_partitioning(
            X,
            search_bounds,
            epsilon_X,
            epsilon_ei,
            gp,
            max_partitions,
            backend=backend_info.selected,
            validation=validation,
            verbose=verbose,
            logMask=logMask,
        )

        Xn = np.asarray(partitioning_result.X, dtype=np.float64).reshape(-1)
        yn = _evaluate_objective(objective, Xn)

        Xn_display = (
            _from_unit_cube(Xn, physical_bounds, validation=False)
            if physical_bounds is not None
            else Xn
        )

        if verbose:
            print(f"Evaluated new point: {Xn_display} -> {yn}")

        if logMask:
            log[f"i{iteration}"].update(partitioning_result.log)
            log[f"i{iteration}"].update({"Xn": Xn_display.copy(), "yn": yn.copy()})

        X = np.vstack((X, Xn))
        y = np.hstack((y, yn))

    X_result = (
        _from_unit_cube(X, physical_bounds, validation=False)
        if physical_bounds is not None
        else X
    )
    return BOResult(X_result, y, backend_info, log)


def exactbo_partitioning(
    X: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray,
    epsilon_ei: float,
    gp,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> BOResult:
    """Run one ExactBO partitioning step on a fitted sklearn-like GP."""
    backend_info = require_cupynumeric_backend(backend)
    X, search_bounds, dim = _normalize_inputs(X, bounds, validation=validation)

    partitions = int(max_partitions)
    if partitions <= 0:
        raise ValueError(f"max_partitions must be > 0, got {partitions}")

    state = _extract_sklearn_gp_state(X, gp)
    bounds_L = cp.asarray(search_bounds[:, 0][cp.newaxis, :], dtype=cp.float64)
    bounds_U = cp.asarray(search_bounds[:, 1][cp.newaxis, :], dtype=cp.float64)
    domain_width = bounds_U[0] - bounds_L[0]
    epsilon_X = cp.asarray(epsilon_X, dtype=cp.float64)

    lhs_points_per_box = int(2**dim)
    lhs_unit_design = cp.asarray(
        _centered_latin_hypercube_unit(lhs_points_per_box, dim),
        dtype=cp.float64,
    )

    stride = 2 * dim + 1
    partition = 0
    n_total = 1
    n_target_start = 1
    idx_best_global = 0
    idx_best_global_next = 0
    best_x = None
    best_ei_scaled = 0.0

    log = _init_log(logMask)
    start_time = pytime.perf_counter() if logMask else None

    while partition < partitions:
        n_boxes = _scalar_int(bounds_L.shape[0])

        if partition > 0:
            n_target_start = n_target * stride
            idx_best_global_start = idx_best_global_next * stride
            preserved_analyze_idx = cp.arange(
                idx_best_global_start,
                idx_best_global_start + stride,
                dtype=cp.int64,
            )
        else:
            preserved_analyze_idx = cp.asarray([0], dtype=cp.int64)

        bounds_L_target = bounds_L[:n_target_start]
        bounds_U_target = bounds_U[:n_target_start]

        if verbose:
            print(
                f"Partition {partition}/{partitions - 1}, "
                f"Boxes: {n_total}, Target boxes to analyze: {n_target_start}."
            )

        ei_hi = _compute_ei_upper_bounds(
            bounds_L_target,
            bounds_U_target,
            state,
            dim,
            validation=validation,
        )
        idx_max_ei_hi = _scalar_int(cp.argmax(ei_hi))
        max_ei_hi = _scalar_float(ei_hi[idx_max_ei_hi])

        analyze_box_mask = ei_hi >= (max_ei_hi - epsilon_ei)
        analyze_box_mask[preserved_analyze_idx] = True
        analyze_local_idx = cp.where(analyze_box_mask)[0]
        n_analyze = _scalar_int(analyze_local_idx.shape[0])

        analyze_best_points, ei_analyze = _sample_boxes_best_ei(
            bounds_L_target[analyze_local_idx],
            bounds_U_target[analyze_local_idx],
            gp,
            state,
            lhs_unit_design,
        )
        idx_ei_max_analyze = _scalar_int(cp.argmax(ei_analyze))
        ei_max_analyze = _scalar_float(ei_analyze[idx_ei_max_analyze])
        idx_ei_max_analyze_local = _scalar_int(analyze_local_idx[idx_ei_max_analyze])
        best_x_analyze = cp.asarray(analyze_best_points[idx_ei_max_analyze], dtype=cp.float64)
        best_x = best_x_analyze
        best_ei_scaled = ei_max_analyze

        w_max_ei_analyzed = bounds_U_target[idx_ei_max_analyze_local] - bounds_L_target[idx_ei_max_analyze_local]
        active_boxes_mask = ei_hi > (ei_max_analyze + epsilon_ei)
        n_active = _scalar_int(cp.sum(active_boxes_mask))

        if n_active == 0 and _scalar_bool(cp.all(w_max_ei_analyzed < epsilon_X)):
            return _build_partition_result(
                best_x_analyze,
                ei_max_analyze,
                backend_info=backend_info,
                log=log,
                start_time=start_time,
                y_train_std=state.y_train_std,
            )

        active_boxes_mask[idx_ei_max_analyze_local] = True
        active_local_idx = cp.where(active_boxes_mask)[0]
        n_active = _scalar_int(active_local_idx.shape[0])

        active_best_points, ei_active = _sample_boxes_best_ei(
            bounds_L_target[active_local_idx],
            bounds_U_target[active_local_idx],
            gp,
            state,
            lhs_unit_design,
        )
        idx_best = _scalar_int(cp.argmax(ei_active))
        ei_max_active = _scalar_float(ei_active[idx_best])
        idx_best_local = _scalar_int(active_local_idx[idx_best])
        best_x_active = cp.asarray(active_best_points[idx_best], dtype=cp.float64)
        best_x = best_x_active
        best_ei_scaled = ei_max_active

        w_max_ei_active = bounds_U_target[idx_best_local] - bounds_L_target[idx_best_local]
        target_boxes_mask = ei_hi > (ei_max_active + epsilon_ei)
        n_target = _scalar_int(cp.sum(target_boxes_mask))

        if n_target == 0 and _scalar_bool(cp.all(w_max_ei_active < epsilon_X)):
            return _build_partition_result(
                best_x_active,
                ei_max_active,
                backend_info=backend_info,
                log=log,
                start_time=start_time,
                y_train_std=state.y_train_std,
            )

        idx_best_global = idx_best_local
        target_boxes_mask[idx_best_global] = True
        n_target = _scalar_int(cp.sum(target_boxes_mask))
        idx_best_global_next = _scalar_int(cp.sum(target_boxes_mask[:idx_best_global]))

        if verbose:
            target_width = cp.max(bounds_U_target[target_boxes_mask] - bounds_L_target[target_boxes_mask], axis=0)
            print(
                f"  Analyzed: {n_analyze}, Active: {n_active}, Target: {n_target}, "
                f"Max EI_hi: {max_ei_hi:.6f}, Max EI Active: {ei_max_active:.6f}, "
                f"Max Width: {target_width}."
            )

        partition += 1
        n_total += n_target * (2 * dim)

        if partition < partitions:
            bounds_L, bounds_U = split_boxes(
                bounds_L_target,
                bounds_U_target,
                target_boxes_mask,
                domain_width,
                n_target_start,
                dim,
                keep_inactive=False,
                backend="cupynumeric",
                validation=validation,
            )

    return _build_partition_result(
        best_x,
        best_ei_scaled,
        backend_info=backend_info,
        log=log,
        start_time=start_time,
        y_train_std=state.y_train_std,
    )
