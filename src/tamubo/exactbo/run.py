from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass
from typing import Callable
import time as pytime

import numpy as np

from tamubo.utils import BackendName, resolve_backend, BOResult, _evaluate_objective, _init_log
from tamubo.acquisition_functions import expected_improvement
from tamubo.gpugp.posterior import gp_posterior
from .bounds import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
from .partition import split_boxes

def _normalize_epsilon(epsilon: np.ndarray | float, dim: int) -> np.ndarray:
    """Normalize epsilon to a per-dimension array."""
    eps = np.asarray(epsilon, dtype=float)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=float)
    if eps.shape == (dim,):
        return eps
    raise ValueError(f"epsilon must be scalar or shape ({dim},), got {eps.shape}")

def _array_module(backend: BackendName = "auto"):
    """Return the resolved array module (`numpy` or `cupynumeric`)."""
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return np
    # Import cupynumeric only when it is the selected backend.
    return import_module("cupynumeric")

def _get_timer(xp):
    if xp.__name__ == "numpy":
        return pytime.perf_counter
    else:
        legatetime = getattr(import_module("legate.timing"), "time")
        return lambda: legatetime()


def _force_materialization(backend: BackendName) -> None:
    """
    Force completion of deferred backend work.

    For cuPyNumeric/Legate this first issues a mapping fence to flush the
    scheduler window and limit overlap of downstream mappings, then issues a
    blocking execution fence so all previously launched work finishes before
    control returns to Python.
    """
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return
    get_legate_runtime = getattr(import_module("legate.core"), "get_legate_runtime")
    runtime = get_legate_runtime()
    runtime.issue_mapping_fence()
    runtime.issue_execution_fence(block=True)


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
    predict_batch_size: int | None = None,
    bounds_batch_size: int | None = None,
    max_target_boxes: int | None = None,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> BOResult:
    """
    Run ExactBO with backend selection and CPU fallback.

    Parameters
    ----------
    X0 : ndarray, shape (N0, d)
        Initial evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dim.
    epsilon_X : float or ndarray, shape (d,)
        Partition termination threshold(s) for the input space.
    epsilon_ei : float
        Threshold for the expected improvement values.
    gp : sklearn-like regressor
        Surrogate model with .fit/.predict plus sklearn GP attributes.
    f : callable
        Objective function.
    max_iters : int
        Outer BO iterations.
    max_partitions : int
        Max partition loops per BO iteration.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Execution backend.
    predict_batch_size : int, optional
        Max number of query points per GP posterior prediction call during
        partitioning. If None, an automatic memory-aware value is used.
    bounds_batch_size : int, optional
        Max number of target boxes processed per bounds chunk during
        partitioning. If None, an automatic memory-aware value is used.
    max_target_boxes : int, optional
        Hard cap for the number of target boxes kept per partition.
        If None, no cap is applied.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).
    verbose : bool, default=False
        Print loop-level progress.
    logMask : bool, default=False
        Enable logging of intermediate results.

    Returns
    -------
    BOResult
        Final design points, objective values, backend resolution info and log.
    """
    
    X = np.asarray(X0, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)

    if validation:
        if X.ndim != 2:
            raise ValueError(f"X0 must be 2D with shape (N0, d), got {X.shape}")
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")

    dim = bounds.shape[0]
    if validation:
        if X.shape[1] != dim:
            raise ValueError(
                f"X0 second dimension ({X.shape[1]}) must match bounds dimension ({dim})"
            )

    epsilon_X = _normalize_epsilon(epsilon_X, dim)
    backend_info = resolve_backend(backend)

    log = _init_log(logMask)

    for iteration in range(max_iters):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iters}")
        # Evaluate function at current data points
        y = _evaluate_objective(f, X)  # (N,)
        if verbose:
            print(f"Current training data: \nX: {X}, \ny: {y}")
        if logMask:
            log[f"i{iteration}"] = {"X": X.copy(), "y": y.copy()}

        # Fit Gaussian Process
        gp.fit(X, y)
        if verbose:
            print(f"GP kernel after fitting: {gp.kernel_}")

        # Run partitioning to find next point and evaluate it
        partitioning_result = exactbo_partitioning(
            X,
            bounds,
            epsilon_X,
            epsilon_ei,
            gp,
            max_partitions,
            backend=backend_info.selected,
            predict_batch_size=predict_batch_size,
            bounds_batch_size=bounds_batch_size,
            max_target_boxes=max_target_boxes,
            validation=validation,
            verbose=verbose,
            logMask=logMask,
        )
        Xn = np.asarray(partitioning_result.X, dtype=np.float64).ravel()  # (d,)
        yn = _evaluate_objective(f, Xn)  # (1,)
        if verbose:
            print(f"Evaluated new point: {Xn} -> {yn}")
        if logMask:
            log[f"i{iteration}"].update(partitioning_result.log)
            log[f"i{iteration}"].update({"Xn": Xn.copy(), "yn": yn.copy()})

        # Update data
        X = np.vstack((X, Xn))  # (N+1,d)
        y = np.hstack((y, yn))  # (N+1,)
        _force_materialization(backend)
        
    
    return BOResult(X, y, backend_info, log)


def exactbo_partitioning(
    X: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray,
    epsilon_ei: float,
    gp,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    predict_batch_size: int | None = None,
    bounds_batch_size: int | None = None,
    max_target_boxes: int | None = None,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> BOResult:
    """
    Run ExactBO Partitioning with backend selection.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dim.
    epsilon_X : float or ndarray, shape (d,)
        Partition termination threshold(s) for the input space.
    epsilon_ei : float
        Threshold for the expected improvement values.
    gp : sklearn-like regressor
        Surrogate model with .fit/.predict plus sklearn GP attributes.
    iteration : int
        Current BO iteration.
    max_partitions : int
        Max partition loops per BO iteration.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    predict_batch_size : int, optional
        Max number of query points per GP posterior prediction call.
        If None, an automatic memory-aware value is used.
    bounds_batch_size : int, optional
        Max number of target boxes processed per bounds chunk.
        If None, an automatic memory-aware value is used.
    max_target_boxes : int, optional
        Hard cap for the number of target boxes kept per partition.
        If None, no cap is applied.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).
    verbose : bool, default=False
        Print loop-level progress.
    logMask : bool, default=False
        Enable logging of intermediate results.

    Returns
    -------
    BOResult
        Next design point, backend resolution info and log.
    """
    xp = _array_module(backend)
    
    # Copy of X for partitioning, converted to backend array
    Xc = xp.asarray(X, dtype=xp.float64)

    # Initialize boxes
    ## One row per box (initially one box)
    ## One column per dimension
    bounds_L = xp.asarray([bounds[:, 0]], dtype=xp.float64)  # (n,d)
    bounds_U = xp.asarray([bounds[:, 1]], dtype=xp.float64)  # (n,d)

    # GP hyperparameters
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params["k1__k1__constant_value"]
    sigma_n_2 = gp_kernel_params["k2__noise_level"]
    length_scale = xp.asarray(gp_kernel_params["k1__k2__length_scale"], dtype=xp.float64)
    alpha = xp.asarray(gp.alpha_, dtype=xp.float64).reshape(-1)
    y_train_std = float(np.asarray(gp._y_train_std, dtype=np.float64).ravel()[0])
    y_train_mean = float(np.asarray(gp._y_train_mean, dtype=np.float64).ravel()[0])
    L = xp.asarray(gp.L_, dtype=xp.float64)  # (N,N)
    y_min = xp.min(gp.y_train_)
    y_min_unscaled = y_min * y_train_std + y_train_mean

    # Partition parameters
    N = Xc.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    if predict_batch_size is None:
        predict_batch_size = max(1, int((128 * 1024**2) // max(16 * N, 16)))
    else:
        predict_batch_size = int(predict_batch_size)
        if predict_batch_size <= 0:
            raise ValueError("predict_batch_size must be a positive integer.")
    if bounds_batch_size is None:
        bounds_batch_size = max(1, int((192 * 1024**2) // max(16 * N, 16)))
    else:
        bounds_batch_size = int(bounds_batch_size)
        if bounds_batch_size <= 0:
            raise ValueError("bounds_batch_size must be a positive integer.")
    if max_target_boxes is not None:
        max_target_boxes = int(max_target_boxes)
        if max_target_boxes <= 0:
            raise ValueError("max_target_boxes must be a positive integer.")
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    epsilon_X = xp.asarray(epsilon_X, dtype=xp.float64)
    partition = 0
    w_max = w.copy()
    target_boxes_mask = xp.ones((1,), dtype=bool)
    n_target_start = 1
    idx_best_global = 0
    n_total = 1

    # Initialize log
    log = _init_log(logMask)

    def _predict_with_std(points):
        points = xp.asarray(points, dtype=xp.float64)
        n_points = int(points.shape[0])
        if n_points <= predict_batch_size:
            if xp.__name__ == "numpy":
                mu_chunk, sigma_chunk = gp.predict(np.asarray(points), return_std=True)
            else:
                mu_chunk, sigma_chunk = gp_posterior(
                    points,
                    X_train=Xc,
                    alpha=alpha,
                    L=L,
                    length_scale=length_scale,
                    sigma_f_squared=sigma_f_2,
                    sigma_n_squared=sigma_n_2,
                    y_train_mean=y_train_mean,
                    y_train_std=y_train_std,
                    return_std=True,
                    backend=backend,
                    validation=False,
                )
            return (
                xp.asarray(mu_chunk, dtype=xp.float64),
                xp.asarray(sigma_chunk, dtype=xp.float64),
            )

        mu = xp.empty((n_points,), dtype=xp.float64)
        sigma = xp.empty((n_points,), dtype=xp.float64)
        for start in range(0, n_points, predict_batch_size):
            end = min(start + predict_batch_size, n_points)
            chunk = points[start:end]
            if xp.__name__ == "numpy":
                mu_chunk, sigma_chunk = gp.predict(np.asarray(chunk), return_std=True)
            else:
                mu_chunk, sigma_chunk = gp_posterior(
                    chunk,
                    X_train=Xc,
                    alpha=alpha,
                    L=L,
                    length_scale=length_scale,
                    sigma_f_squared=sigma_f_2,
                    sigma_n_squared=sigma_n_2,
                    y_train_mean=y_train_mean,
                    y_train_std=y_train_std,
                    return_std=True,
                    backend=backend,
                    validation=False,
                )
            mu[start:end] = xp.asarray(mu_chunk, dtype=xp.float64)
            sigma[start:end] = xp.asarray(sigma_chunk, dtype=xp.float64)
            _force_materialization(backend)
        return mu, sigma

    def _ei_hi_bounds_chunked(bounds_L_target, bounds_U_target):
        n_target = int(bounds_L_target.shape[0])
        ei_hi = xp.empty((n_target,), dtype=xp.float64)
        for start in range(0, n_target, bounds_batch_size):
            end = min(start + bounds_batch_size, n_target)
            chunk_n = end - start
            chunk_L = bounds_L_target[start:end]
            chunk_U = bounds_U_target[start:end]

            K_lo = xp.empty((chunk_n, N), dtype=xp.float64)
            K_hi = xp.empty((chunk_n, N), dtype=xp.float64)
            for i in range(N):
                xi = Xc[i]
                K_lo[:, i], K_hi[:, i] = rbf_k_bounds(
                    chunk_L,
                    chunk_U,
                    xi,
                    chunk_n,
                    d,
                    sigma_f_2,
                    length_scale,
                    backend=backend,
                    validation=validation,
                )

            mu_lo, mu_hi = mu_bounds(
                alpha,
                K_lo,
                K_hi,
                chunk_n,
                N,
                y_train_mean=y_train_mean,
                y_train_std=y_train_std,
                backend=backend,
                validation=validation,
            )
            sig_lo, sig_hi = sigma_bounds(
                K_lo,
                K_hi,
                L,
                chunk_n,
                N,
                sigma_f_2,
                y_train_std=y_train_std,
                backend=backend,
                validation=validation,
            )
            _, ei_hi_chunk = ei_bounds(
                mu_lo,
                mu_hi,
                sig_lo,
                sig_hi,
                chunk_n,
                y_min_unscaled,
                backend=backend,
                validation=validation,
            )
            ei_hi[start:end] = ei_hi_chunk
            del K_lo, K_hi, mu_lo, mu_hi, sig_lo, sig_hi, ei_hi_chunk
            # cuPyNumeric/Legate can defer chunk work aggressively. Fence here so
            # the helper does not accumulate a large pending graph across dozens
            # of chunks before the next reduction/mask operation forces it.
            _force_materialization(backend)
        return ei_hi

    def _topk_primary_secondary(primary, secondary, k: int):
        """
        Return indices of the top-k entries using `primary` as the main score and
        `secondary` as a tie-breaker, without sorting the full input.
        """
        m = int(primary.shape[0])
        if k >= m:
            return xp.arange(m, dtype=xp.int64)

        shortlist = min(m, max(2 * k, k + 4096))
        if hasattr(xp, "argpartition") and shortlist < m:
            shortlist_idx = xp.argpartition(primary, m - shortlist)[m - shortlist:]
        else:
            shortlist_idx = xp.argsort(primary)[-shortlist:]

        short_primary = primary[shortlist_idx]
        short_secondary = secondary[shortlist_idx]
        s = int(shortlist_idx.shape[0])

        rank_secondary = xp.empty((s,), dtype=xp.int64)
        order_secondary = xp.argsort(short_secondary)
        rank_secondary[order_secondary] = xp.arange(s, dtype=xp.int64)

        rank_primary = xp.empty((s,), dtype=xp.int64)
        order_primary = xp.argsort(short_primary)
        rank_primary[order_primary] = xp.arange(s, dtype=xp.int64)

        combined = rank_primary * (s + 1) + rank_secondary
        keep_order = xp.argsort(combined)[-k:]
        keep_idx = shortlist_idx[keep_order]
        del short_primary, short_secondary, rank_secondary, order_secondary, rank_primary, order_primary, combined, keep_order
        return keep_idx
    
    if logMask:
        now = _get_timer(xp)
        t0 = now()
    
    while partition < max_partitions:
        if verbose:
            print(
                f"Partition {partition}/{max_partitions-1},"
            )
        # Total number of boxes
        n = bounds_L.shape[0]
        
        # For the first partition, we analyze just the original box. For subsequent partitions, 
        # we only focus on the new target boxes resulting from the previous partition.
        if partition > 0:
            # Starting target box count (new 2d+1 boxes per each of the n_targets boxes from the previous partition)
            n_target_start = n_target*(2*d+1)
            # Starting target box mask (only analyze the new target boxes from the previous partition)
            target_boxes_mask = xp.zeros((n,), dtype=bool)
            target_boxes_mask[:n_target_start] = True
            # Calculate where is the best global box in the new partition (it should be among the new target boxes)
            idx_best_global = int(idx_best_global_next*(2*d+1)+2*d)
        
        # Target boxes are always at the start of the arrays after each split.
        # Use slicing (views) to avoid advanced-index copies of large arrays.
        bounds_L_target = bounds_L[:n_target_start]
        bounds_U_target = bounds_U[:n_target_start]

        #if verbose:
        #    print(f"  Start target boxes: {n}, to analyze: {bounds_L_target.shape[0]}, Best global box index: {idx_best_global}.")
        # Compute EI upper bounds in chunks to cap peak GPU memory.
        ei_hi = _ei_hi_bounds_chunked(bounds_L_target, bounds_U_target)
        #if verbose:
        #    print(f"  Computed EI upper bounds for {ei_hi.shape[0]} target boxes.")

        # Find the box with the highest upper EI bound
        idx_max_ei_hi = int(xp.argmax(ei_hi))
        #if verbose:
        #    print(f"  Max EI_hi at box index {idx_max_ei_hi}.")
        max_ei_hi = float(ei_hi[idx_max_ei_hi])

        # Analyze boxes where the upper EI bound is within epsilon_ei of the maximum upper EI bound
        analyze_box_mask = ei_hi >= (max_ei_hi - epsilon_ei)  # (n_target_start,)
        # Make sure to analyze at least the box with highest EI from past partition
        analyze_box_mask[idx_best_global] = True
        analyze_local_idx = xp.where(analyze_box_mask)[0]  # (n_analyze,)
        n_analyze = int(analyze_local_idx.shape[0])
        # Calculate EI at the center of the boxes (use latent sigma for EI calculation)
        ei_hi_analyze_centers = (
            bounds_L_target[analyze_local_idx] + bounds_U_target[analyze_local_idx]
        ) / 2.0  # (n_analyze,d)
        #if verbose:
        #    print(f"  Computing GP predictions for {ei_hi_analyze_centers.shape[0]} analyzed boxes.")
        mu_analyze, sigma_analyze = _predict_with_std(ei_hi_analyze_centers)  # (n_analyze,) both
        #if verbose:
        #    print(f"  Computed GP predictions for {mu_analyze.shape[0]} analyzed boxes.")
        mu_analyze = xp.asarray(mu_analyze, dtype=xp.float64)
        sigma_analyze = xp.asarray(sigma_analyze, dtype=xp.float64)
        sigma_analyze_lat = xp.sqrt(xp.clip(sigma_analyze**2 - sigma_n_2 * y_train_std**2, 1e-12, None))  # Avoid zero std for EI calculation
        ei_analyze = expected_improvement(mu_analyze, sigma_analyze_lat, y_min_unscaled, backend=backend)  # (n_analyze,)
        #if verbose:
        #    print(f"  Computed EI for {ei_analyze.shape[0]} analyzed boxes.")
        # Find the box with the highest analyzed EI
        idx_ei_max_analyze = int(xp.argmax(ei_analyze))
        ei_max_analyze = float(ei_analyze[idx_ei_max_analyze])
        idx_ei_max_analyze_local = int(analyze_local_idx[idx_ei_max_analyze])
        # Find best point among the analyzed boxes and the width of the box with the highest analyzed EI
        best_x_analyze = ei_hi_analyze_centers[idx_ei_max_analyze]
        w_max_ei_analyzed = (
            bounds_U_target[idx_ei_max_analyze_local] - bounds_L_target[idx_ei_max_analyze_local]
        )  # (d,)
        del mu_analyze, sigma_analyze, sigma_analyze_lat, ei_analyze, ei_hi_analyze_centers

        # Active boxes are the ones where ei_hi is higher than ei_max_analyze plus epsilon_ei,
        active_boxes_mask = ei_hi > (ei_max_analyze + epsilon_ei)  # (n_target_start,)
        n_active = int(xp.sum(active_boxes_mask))
        
        # No active boxes and max EI box is smaller than epsilon_X, return the best point found
        if n_active == 0 and xp.all(w_max_ei_analyzed < epsilon_X):
            if verbose:
                print(
                    f"  Boxes: {n_total}, Analyzed: {n_analyze}, Active: 0,\n"
                    f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f},\n"
                    f"  Max EI Analyzed Box Width: {w_max_ei_analyzed}, Terminating partitioning."
                )
            # Uncomment this for 2D animations
            #if logMask:
            #    log[f"p{partition}"] = {
            #        "bounds_L": np.asarray(bounds_L),
            #        "bounds_U": np.asarray(bounds_U),
            #        "target_boxes_mask": np.zeros((n,), dtype=bool),
            #    }
            if logMask:
                t1 = now()
                if xp.__name__ == "numpy":
                    log['time'] = t1 - t0
                else:
                    log['time'] = (t1 - t0) / 1e6  # Convert microseconds to seconds for Legate
                log['ei_max'] = float(ei_max_analyze)
            return BOResult(X=np.asarray(best_x_analyze), log=log)
        else:
            # Ensure the box with the highest analyzed EI is also active.
            active_boxes_mask[idx_ei_max_analyze_local] = True
            n_active = int(xp.sum(active_boxes_mask))

        # Check the active boxes.
        active_local_idx = xp.where(active_boxes_mask)[0] # (n_active,)
        # Calculate EI at the center of the boxes (use latent sigma for EI calculation)
        center_active = (
            bounds_L_target[active_local_idx] + bounds_U_target[active_local_idx]
        ) / 2.0  # (n_active,d)
        mu_active, sigma_active = _predict_with_std(center_active)  # (n_active,) both
        
        mu_active = xp.asarray(mu_active, dtype=xp.float64)
        sigma_active = xp.asarray(sigma_active, dtype=xp.float64)
        #if verbose:
        #    print(f"  Computed GP predictions for {mu_active.shape[0]} active boxes.")
        sigma_active_lat = xp.sqrt(xp.clip(sigma_active**2 - sigma_n_2 * y_train_std**2, 1e-12, None))  # Avoid zero std for EI calculation
        ei_active = expected_improvement(mu_active, sigma_active_lat, y_min_unscaled, backend=backend)  # (n_active,)
        #if verbose:
        #    print(f"  Computed EI for {ei_active.shape[0]} active boxes.")
        # Find the box with the highest EI among the active boxes
        idx_best = int(xp.argmax(ei_active))
        ei_max_active = float(ei_active[idx_best])
        idx_best_local = int(active_local_idx[idx_best])
        # Find best point among the active boxes and the width of the box with the highest active EI
        best_x_active = center_active[idx_best]
        w_max_ei_active = bounds_U_target[idx_best_local] - bounds_L_target[idx_best_local]  # (d,)
        del mu_active, sigma_active, sigma_active_lat, center_active

        # Target boxes are the ones where ei_hi is more than epsilon_ei plus ei_max.
        target_boxes_mask[:n_target_start] = ei_hi > (ei_max_active + epsilon_ei)
        n_target = int(xp.sum(target_boxes_mask))

        # No target boxes and max EI box is smaller than epsilon_X, return the best point found
        if n_target == 0 and xp.all(w_max_ei_active < epsilon_X):
            if verbose:
                print(
                    f"  Boxes: {n_total}, Analyzed: {n_analyze}, Active: {n_active}, Target: 0,\n"
                    f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f}, Max EI Active: {ei_max_active:.6f},\n"
                    f"  Max EI Active Box Width: {w_max_ei_active}, Terminating partitioning."
                )
            # Uncomment this for 2D animations
            #if logMask:
            #    log[f"p{partition}"] = {
            #        "bounds_L": np.asarray(bounds_L),
            #        "bounds_U": np.asarray(bounds_U),
            #        "target_boxes_mask": np.zeros((n,), dtype=bool),
            #    }
            if logMask:
                t1 = now()
                if xp.__name__ == "numpy":
                    log['time'] = t1 - t0
                else:
                    log['time'] = (t1 - t0) / 1e6  # Convert microseconds to seconds for Legate
                log['ei_max'] = float(ei_max_active)
            return BOResult(X=np.asarray(best_x_active), log=log)
        else:
            # Ensure the box with the highest active EI is also a target box.
            idx_best_global = int(idx_best_local)
            target_boxes_mask[idx_best_global] = True
            n_target = int(xp.sum(target_boxes_mask))

        # Optional approximation guard to avoid combinatorial target growth.
        if max_target_boxes is not None and n_target > max_target_boxes:
            #if verbose:
            #    print(
            #        f"  Pruning target boxes from {n_target} to {max_target_boxes} to limit combinatorial growth."
            #    )
            keep = min(max_target_boxes, n_target)
            # Target boxes are a subset of active boxes, so use active ordering
            # directly and avoid the expensive searchsorted/multi-sort path.
            target_in_active_mask = ei_hi[active_local_idx] > (ei_max_active + epsilon_ei)
            target_in_active_mask[idx_best] = True
            target_local_idx = active_local_idx[target_in_active_mask]
            target_ei_active = ei_active[target_in_active_mask]
            target_ei_hi = ei_hi[target_local_idx]

            keep_pos = _topk_primary_secondary(target_ei_active, target_ei_hi, keep)
            keep_local = target_local_idx[keep_pos]
            target_boxes_mask[:n_target_start] = False
            target_boxes_mask[keep_local] = True
            target_boxes_mask[idx_best_global] = True
            n_target = int(xp.sum(target_boxes_mask))
            if verbose:
                print(
                    f"  Pruned target boxes from {int(target_local_idx.shape[0])} to {n_target} "
                    f"(max_target_boxes={max_target_boxes}, score=ei_active+ei_hi tie-break)."
                )
            del target_in_active_mask, target_local_idx, target_ei_active, target_ei_hi, keep_pos, keep_local

        # Calculate position of the best point in next partition
        idx_best_global_next = int(xp.sum(target_boxes_mask[:idx_best_global]))
        del ei_hi, ei_active, analyze_box_mask, active_boxes_mask, analyze_local_idx, active_local_idx
        
        # Update maximum width of active boxes
        w_max = xp.max(bounds_U[target_boxes_mask] - bounds_L[target_boxes_mask], axis=0)

        if verbose:
            print(
                f"  Boxes: {n_total}, Analyzed: {n_analyze}, Active: {n_active}, Target: {n_target},\n"
                f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f}, Max EI Active: {ei_max_active:.6f},\n" 
                f"  Max Width: {w_max}."
            )

        # Uncomment this for 2D animations
        #if logMask:
        #    log[f"p{partition}"] = {
        #        "bounds_L": np.asarray(bounds_L),
        #        "bounds_U": np.asarray(bounds_U),
        #        "target_boxes_mask": np.asarray(target_boxes_mask),
        #    }
        
        # Update partition count
        partition += 1

        # Calculate number of boxes for next iteration
        n_total = n_total + n_target * (2 * d)

        # Split active boxes (don't if its the last partition)
        if partition < max_partitions:
            bounds_L, bounds_U = split_boxes(
                bounds_L,
                bounds_U,
                target_boxes_mask,
                w,
                n,
                d,
                keep_inactive=False,
                backend=backend,
                validation=validation,
            )
            #if verbose:
            #    print("  Materializing split output before next EI-upper-bound pass.")
            _force_materialization(backend)

    # Store final log info
    if logMask:
        t1 = now()
        if xp.__name__ == "numpy":
            log['time'] = t1 - t0
        else:
            log['time'] = (t1 - t0) / 1e6  # Convert microseconds to seconds for Legate
        log['ei_max'] = float(ei_max_active)

    return BOResult(X=np.asarray(best_x_active), log=log)
