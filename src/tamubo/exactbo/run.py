from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import time as pytime

import numpy as np
from legate.core import get_legate_runtime

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


_DEFAULT_SAMPLE_BATCH_BYTES = 64 * 1024**2
_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_ROW_COPY_MAX_SELECTED = 4096
_CPU_MAX_WIDTH_ROWS = 10_000


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


def _chunk_size_for_bytes(bytes_per_item: int, max_bytes: int) -> int:
    if max_bytes <= 0:
        raise ValueError(f"sample_batch_bytes must be positive, got {max_bytes}")
    return max(1, int(max_bytes) // max(1, int(bytes_per_item)))


def _execute_pending_tasks() -> None:
    get_legate_runtime().issue_execution_fence(block=True)


def _split_batch_size(dim: int, stride: int, batch_bytes: int) -> int:
    bytes_per_box = (4 * stride * dim + 6 * dim) * _FLOAT64_BYTES
    return _chunk_size_for_bytes(bytes_per_box, batch_bytes)


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
    batch_bytes: int,
    validation: bool,
):
    n_target = _scalar_int(bounds_L_target.shape[0])
    if n_target == 0:
        return cp.empty((0,), dtype=cp.float64)

    n_train = _scalar_int(state.alpha.shape[0])
    bytes_per_box = (4 * dim + 2 * n_train + 12) * _FLOAT64_BYTES
    chunk_size = _chunk_size_for_bytes(bytes_per_box, batch_bytes)
    ei_hi = cp.empty((n_target,), dtype=cp.float64)

    for start in range(0, n_target, chunk_size):
        stop = min(start + chunk_size, n_target)
        chunk_L = bounds_L_target[start:stop].copy()
        chunk_U = bounds_U_target[start:stop].copy()
        ei_hi[start:stop] = _compute_ei_upper_bounds_chunk(
            chunk_L,
            chunk_U,
            state,
            dim,
            validation=validation,
        )
        _execute_pending_tasks()

    return ei_hi


def _compute_ei_upper_bounds_chunk(
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


def _sample_points_for_boxes(boxes_L, boxes_U, lhs_unit_design):
    """Build a flat sample buffer without materializing a 3D box/sample tensor."""
    n_boxes = _scalar_int(boxes_L.shape[0])
    dim = _scalar_int(boxes_L.shape[1])
    n_samples = _scalar_int(lhs_unit_design.shape[0])

    flat_points = cp.empty((n_samples * n_boxes, dim), dtype=cp.float64)
    widths = boxes_U - boxes_L

    for sample_idx in range(n_samples):
        rows = slice(sample_idx * n_boxes, (sample_idx + 1) * n_boxes)
        sample_points = flat_points[rows]
        cp.multiply(widths, lhs_unit_design[sample_idx], out=sample_points)
        sample_points += boxes_L

    return flat_points


def _best_sampled_box_in_chunk(
    boxes_L,
    boxes_U,
    gp,
    state: SklearnGPState,
    lhs_unit_design,
    box_mask=None,
):
    n_boxes = _scalar_int(boxes_L.shape[0])
    n_samples = _scalar_int(lhs_unit_design.shape[0])

    flat_points = _sample_points_for_boxes(boxes_L, boxes_U, lhs_unit_design)
    mu, sigma = _predict_standardized_posterior(flat_points, gp, state)
    mu = cp.asarray(mu, dtype=cp.float64).reshape((n_samples, n_boxes))
    sigma = cp.asarray(sigma, dtype=cp.float64).reshape((n_samples, n_boxes))

    sigma_latent = cp.empty_like(sigma)
    cp.multiply(sigma, sigma, out=sigma_latent)
    sigma_latent -= state.sigma_n_squared
    cp.maximum(sigma_latent, 1e-12, out=sigma_latent)
    cp.sqrt(sigma_latent, out=sigma_latent)

    ei = expected_improvement(
        mu,
        sigma_latent,
        state.y_min_scaled,
        backend="cupynumeric",
    )

    ei_by_box = ei.T
    best_sample_by_box = cp.argmax(ei_by_box, axis=1).reshape((n_boxes, 1))
    best_ei_by_box = cp.take_along_axis(
        ei_by_box,
        best_sample_by_box,
        axis=1,
    ).reshape((n_boxes,))
    if box_mask is not None:
        best_ei_by_box = cp.where(box_mask, best_ei_by_box, -cp.inf)

    best_box_pos = _scalar_int(cp.argmax(best_ei_by_box))
    best_sample_pos = _scalar_int(best_sample_by_box[best_box_pos, 0])
    best_row = best_sample_pos * n_boxes + best_box_pos

    return (
        cp.asarray(flat_points[best_row], dtype=cp.float64).copy(),
        _scalar_float(best_ei_by_box[best_box_pos]),
        best_box_pos,
    )


def _copy_selected_box_bounds(bounds_L_window, bounds_U_window, local_indices, n_selected: int, dim: int):
    if n_selected == _scalar_int(bounds_L_window.shape[0]):
        return bounds_L_window.copy(), bounds_U_window.copy()

    selected_L = cp.empty((n_selected, dim), dtype=cp.float64)
    selected_U = cp.empty((n_selected, dim), dtype=cp.float64)
    local_indices_np = np.asarray(local_indices, dtype=np.int64)

    out_start = 0
    while out_start < n_selected:
        src_start = int(local_indices_np[out_start])
        out_stop = out_start + 1
        src_stop = src_start + 1

        while out_stop < n_selected and int(local_indices_np[out_stop]) == src_stop:
            out_stop += 1
            src_stop += 1

        selected_L[out_start:out_stop] = bounds_L_window[src_start:src_stop]
        selected_U[out_start:out_stop] = bounds_U_window[src_start:src_stop]
        out_start = out_stop

    return selected_L, selected_U


def _sample_masked_boxes_best_ei(
    bounds_L,
    bounds_U,
    box_mask,
    gp,
    state: SklearnGPState,
    lhs_unit_design,
    *,
    sample_batch_bytes: int,
):
    box_mask = cp.asarray(box_mask, dtype=bool)
    n_total = _scalar_int(bounds_L.shape[0])
    dim = _scalar_int(bounds_L.shape[1])
    n_samples = _scalar_int(lhs_unit_design.shape[0])

    n_boxes = _scalar_int(cp.sum(box_mask))
    if n_boxes == 0:
        return cp.empty((dim,), dtype=cp.float64), float("-inf"), -1

    bytes_per_box = (n_samples + 2) * dim * _FLOAT64_BYTES
    chunk_size = _chunk_size_for_bytes(bytes_per_box, sample_batch_bytes)

    best_point = None
    best_ei = float("-inf")
    best_box_idx = -1

    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        mask_window = box_mask[start:stop]
        n_selected = _scalar_int(cp.sum(mask_window))
        if n_selected == 0:
            continue

        bounds_L_window = bounds_L[start:stop]
        bounds_U_window = bounds_U[start:stop]

        if n_selected > _ROW_COPY_MAX_SELECTED:
            chunk_point, chunk_ei, chunk_box_pos = _best_sampled_box_in_chunk(
                bounds_L_window,
                bounds_U_window,
                gp,
                state,
                lhs_unit_design,
                box_mask=mask_window,
            )
            candidate_box_idx = start + chunk_box_pos
        else:
            local_indices = cp.where(mask_window)[0]
            chunk_L, chunk_U = _copy_selected_box_bounds(
                bounds_L_window,
                bounds_U_window,
                local_indices,
                n_selected,
                dim,
            )

            chunk_point, chunk_ei, chunk_box_pos = _best_sampled_box_in_chunk(
                chunk_L,
                chunk_U,
                gp,
                state,
                lhs_unit_design,
            )
            candidate_box_idx = start + _scalar_int(local_indices[chunk_box_pos])

        if chunk_ei > best_ei:
            best_point = chunk_point
            best_ei = chunk_ei
            best_box_idx = candidate_box_idx
        _execute_pending_tasks()

    return best_point, best_ei, best_box_idx


def _max_width_for_masked_boxes(
    bounds_L,
    bounds_U,
    box_mask,
    *,
    sample_batch_bytes: int,
):
    box_mask = cp.asarray(box_mask, dtype=bool)
    n_total = _scalar_int(bounds_L.shape[0])
    dim = _scalar_int(bounds_L.shape[1])

    n_boxes = _scalar_int(cp.sum(box_mask))
    if n_boxes == 0:
        return cp.zeros((dim,), dtype=cp.float64)

    if n_total <= _CPU_MAX_WIDTH_ROWS:
        mask_np = np.asarray(box_mask, dtype=bool)
        widths_np = np.asarray(bounds_U, dtype=np.float64) - np.asarray(bounds_L, dtype=np.float64)
        return np.max(widths_np[mask_np], axis=0)

    bytes_per_box = 2 * dim * _FLOAT64_BYTES
    chunk_size = _chunk_size_for_bytes(bytes_per_box, sample_batch_bytes)
    max_width = cp.full((dim,), -cp.inf, dtype=cp.float64)

    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        mask_window = box_mask[start:stop]
        n_selected = _scalar_int(cp.sum(mask_window))
        if n_selected == 0:
            continue

        if n_selected <= _ROW_COPY_MAX_SELECTED:
            local_indices = cp.where(mask_window)[0]
            chunk_L, chunk_U = _copy_selected_box_bounds(
                bounds_L[start:stop],
                bounds_U[start:stop],
                local_indices,
                n_selected,
                dim,
            )
            chunk_width = chunk_U - chunk_L
        else:
            chunk_width = bounds_U[start:stop] - bounds_L[start:stop]
            if n_selected != stop - start:
                chunk_width = cp.where(mask_window[:, cp.newaxis], chunk_width, -cp.inf)
        max_width = cp.maximum(max_width, cp.max(chunk_width, axis=0))
        _execute_pending_tasks()

    return max_width


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
    sample_batch_bytes: int = _DEFAULT_SAMPLE_BATCH_BYTES,
) -> BOResult:
    """
    Run ExactBO with the cuPyNumeric backend.

    ``backend="auto"`` is still accepted for compatibility, but it must resolve
    to ``"cupynumeric"``.

    ``sample_batch_bytes`` bounds the main per-chunk working buffers used for
    EI-bound evaluation, box splitting, and LHS sample scoring. Smaller values
    reduce peak memory at the cost of more chunks.
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
            sample_batch_bytes=sample_batch_bytes,
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
    sample_batch_bytes: int = _DEFAULT_SAMPLE_BATCH_BYTES,
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
            batch_bytes=sample_batch_bytes,
            validation=validation,
        )
        idx_max_ei_hi = _scalar_int(cp.argmax(ei_hi))
        max_ei_hi = _scalar_float(ei_hi[idx_max_ei_hi])

        analyze_box_mask = ei_hi >= (max_ei_hi - epsilon_ei)
        analyze_box_mask[preserved_analyze_idx] = True
        n_analyze = _scalar_int(cp.sum(analyze_box_mask))

        (
            best_x_analyze,
            ei_max_analyze,
            idx_ei_max_analyze_local,
        ) = _sample_masked_boxes_best_ei(
            bounds_L_target,
            bounds_U_target,
            analyze_box_mask,
            gp,
            state,
            lhs_unit_design,
            sample_batch_bytes=sample_batch_bytes,
        )
        best_x = best_x_analyze
        best_ei_scaled = ei_max_analyze

        w_max_ei_analyzed = (
            bounds_U_target[idx_ei_max_analyze_local]
            - bounds_L_target[idx_ei_max_analyze_local]
        )
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
        n_active = _scalar_int(cp.sum(active_boxes_mask))

        (
            best_x_active,
            ei_max_active,
            idx_best_local,
        ) = _sample_masked_boxes_best_ei(
            bounds_L_target,
            bounds_U_target,
            active_boxes_mask,
            gp,
            state,
            lhs_unit_design,
            sample_batch_bytes=sample_batch_bytes,
        )
        best_x = best_x_active
        best_ei_scaled = ei_max_active

        w_max_ei_active = (
            bounds_U_target[idx_best_local] - bounds_L_target[idx_best_local]
        )
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
            target_width = _max_width_for_masked_boxes(
                bounds_L_target,
                bounds_U_target,
                target_boxes_mask,
                sample_batch_bytes=sample_batch_bytes,
            )
            print(
                f"  Analyzed: {n_analyze}, Active: {n_active}, Target: {n_target}, "
                f"Max EI_hi: {max_ei_hi:.6f}, Max EI Active: {ei_max_active:.6f}, "
                f"Max Width: {target_width}."
            )

        partition += 1
        n_total += n_target * (2 * dim)

        if partition < partitions:
            next_bounds_L, next_bounds_U = split_boxes(
                bounds_L_target,
                bounds_U_target,
                target_boxes_mask,
                domain_width,
                n_target_start,
                dim,
                keep_inactive=False,
                backend="cupynumeric",
                validation=validation,
                batch_size=_split_batch_size(dim, stride, sample_batch_bytes),
            )
            del bounds_L, bounds_U, bounds_L_target, bounds_U_target
            del analyze_box_mask, active_boxes_mask
            del target_boxes_mask, ei_hi
            bounds_L, bounds_U = next_bounds_L, next_bounds_U
            _execute_pending_tasks()

    return _build_partition_result(
        best_x,
        best_ei_scaled,
        backend_info=backend_info,
        log=log,
        start_time=start_time,
        y_train_std=state.y_train_std,
    )
