from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tamubo.utils import BackendName, resolve_backend, BOResult, _evaluate_objective, _init_log
from tamubo.acquisition_functions import expected_improvement
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
        partitioning_result = exactbo_partitioning(X, bounds, epsilon_X, epsilon_ei, gp, max_partitions, backend=backend_info.selected, validation=validation, verbose=verbose, logMask=logMask)
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
    alpha = xp.asarray(gp.alpha_, dtype=xp.float64)
    y_train_std = gp._y_train_std
    y_train_mean = gp._y_train_mean
    L = xp.asarray(gp.L_, dtype=xp.float64)  # (N,N)
    y_min = xp.min(gp.y_train_)
    y_min_unscaled = y_min * y_train_std + y_train_mean

    # Partition parameters
    N = Xc.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    epsilon_X = xp.asarray(epsilon_X, dtype=xp.float64)
    partition = 0
    w_max = w.copy()
    target_boxes_mask = xp.ones((1,), dtype=bool)
    n_target_start = 1
    idx_best_global = 0

    # Initialize log
    log = _init_log(logMask)
    
    while partition < max_partitions:
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
        
        # Get bounds of the target boxes for this partition
        target_idx = xp.where(target_boxes_mask)[0] # (n_target_start,)
        bounds_L_target = bounds_L[target_idx]
        bounds_U_target = bounds_U[target_idx]

        # Bounds
        ## Kernel
        K_lo = xp.zeros((n_target_start, N), dtype=xp.float64)
        K_hi = xp.zeros((n_target_start, N), dtype=xp.float64)
        for i in range(N):
            xi = Xc[i]
            K_lo[:, i], K_hi[:, i] = rbf_k_bounds(bounds_L_target,bounds_U_target,xi,n_target_start,d,sigma_f_2,length_scale,backend=backend,validation=validation)  # (n_target_start,) both
        ## Mean
        mu_lo, mu_hi = mu_bounds(alpha, K_lo, K_hi, n_target_start, N, y_train_mean=y_train_mean, y_train_std=y_train_std, backend=backend, validation=validation)  # (n_target_start,) both
        ## Sigma
        sig_lo, sig_hi = sigma_bounds(K_lo, K_hi, L, n_target_start, N, sigma_f_2, y_train_std=y_train_std, backend=backend, validation=validation)  # (n_target_start,) both
        ## EI
        _, ei_hi = ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, n_target_start, y_min_unscaled, backend=backend, validation=validation)  # (n_target_start,) both

        # Find the box with the highest upper EI bound
        idx_max_ei_hi = int(xp.argmax(ei_hi))
        max_ei_hi = float(ei_hi[idx_max_ei_hi])

        # Analyze boxes where the upper EI bound is within epsilon_ei of the maximum upper EI bound
        analyze_box_mask = ei_hi >= (max_ei_hi - epsilon_ei)  # (n_target_start,)
        # Make sure to analyze at least the box with highest EI from past partition
        analyze_box_mask[idx_best_global] = True
        analyze_local_idx = xp.where(analyze_box_mask)[0]  # (n_analyze,)
        n_analyze = int(analyze_local_idx.shape[0])
        ei_hi_analyze_bounds_L = bounds_L_target[analyze_local_idx]  # (n_analyze,d)
        ei_hi_analyze_bounds_U = bounds_U_target[analyze_local_idx]  # (n_analyze,d)
        # Calculate EI at the center of the boxes (use latent sigma for EI calculation)
        ei_hi_analyze_centers = (ei_hi_analyze_bounds_L + ei_hi_analyze_bounds_U) / 2.0  # (n_analyze,d)
        mu_analyze, sigma_analyze = gp.predict(np.asarray(ei_hi_analyze_centers), return_std=True)  # (n_analyze,) both
        sigma_analyze_lat = np.sqrt(np.clip(sigma_analyze**2 - sigma_n_2*y_train_std**2, 1e-12, None))  # Avoid zero std for EI calculation
        ei_analyze = expected_improvement(xp.asarray(mu_analyze), xp.asarray(sigma_analyze_lat), y_min_unscaled, backend=backend)  # (n_analyze,)
        # Find the box with the highest analyzed EI
        idx_ei_max_analyze = int(xp.argmax(ei_analyze))
        ei_max_analyze = float(ei_analyze[idx_ei_max_analyze])
        idx_ei_max_analyze_local = int(analyze_local_idx[idx_ei_max_analyze])
        # Find best point among the analyzed boxes and the width of the box with the highest analyzed EI
        best_x_analyze = ei_hi_analyze_centers[idx_ei_max_analyze]
        w_max_ei_analyzed = ei_hi_analyze_bounds_U[idx_ei_max_analyze] - ei_hi_analyze_bounds_L[idx_ei_max_analyze]  # (d,)

        # Active boxes are the ones where ei_hi is higher than ei_max_analyze plus epsilon_ei,
        active_boxes_mask = ei_hi > (ei_max_analyze + epsilon_ei)  # (n_target_start,)
        n_active = int(xp.sum(active_boxes_mask))
        
        # No active boxes and max EI box is smaller than epsilon_X, return the best point found
        if n_active == 0 and xp.all(w_max_ei_analyzed < epsilon_X):
            if verbose:
                print(
                    f"Partition {partition}/{max_partitions-1},\n"
                    f"  Boxes: {n}, Analyzed: {n_analyze}, Active: 0,\n"
                    f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f},\n"
                    f"  Max EI Analyzed Box Width: {w_max_ei_analyzed}, Terminating partitioning."
                )
            # Uncomment this for 2D animations
            if logMask:
                log[f"p{partition}"] = {
                    "bounds_L": np.asarray(bounds_L),
                    "bounds_U": np.asarray(bounds_U),
                    "target_boxes_mask": np.zeros((n,), dtype=bool),
                }
            if logMask:
                log['ei_max'] = float(ei_max_analyze)
            return BOResult(X=np.asarray(best_x_analyze), log=log)
        else:
            # Ensure the box with the highest analyzed EI is also active.
            active_boxes_mask[idx_ei_max_analyze_local] = True
            n_active = int(xp.sum(active_boxes_mask))

        # Check the active boxes.
        active_local_idx = xp.where(active_boxes_mask)[0] # (n_active,)
        bound_U_active = bounds_U_target[active_local_idx]  # (n_active,d)
        bound_L_active = bounds_L_target[active_local_idx]  # (n_active,d)
        # Calculate EI at the center of the boxes (use latent sigma for EI calculation)
        center_active = (bound_L_active + bound_U_active) / 2.0  # (n_active,d)
        mu_active, sigma_active = gp.predict(np.array(center_active), return_std=True)  # (n_active,) both
        sigma_active_lat = np.sqrt(np.clip(sigma_active**2 - sigma_n_2*y_train_std**2, 1e-12, None))  # Avoid zero std for EI calculation
        ei_active = expected_improvement(xp.asarray(mu_active), xp.asarray(sigma_active_lat), y_min_unscaled, backend=backend)  # (n_active,)
        # Find the box with the highest EI among the active boxes
        idx_best = int(xp.argmax(ei_active))
        ei_max_active = float(ei_active[idx_best])
        idx_best_local = int(active_local_idx[idx_best])
        # Find best point among the active boxes and the width of the box with the highest active EI
        best_x_active = center_active[idx_best]
        w_max_ei_active = bound_U_active[idx_best] - bound_L_active[idx_best]  # (d,)

        # Target boxes are the ones where ei_hi is more than epsilon_ei plus ei_max.
        target_boxes_mask[target_idx] = ei_hi > (ei_max_active + epsilon_ei)  
        n_target = int(xp.sum(target_boxes_mask))

        # No target boxes and max EI box is smaller than epsilon_X, return the best point found
        if n_target == 0 and xp.all(w_max_ei_active < epsilon_X):
            if verbose:
                print(
                    f"Partition {partition}/{max_partitions-1}, \n"
                    f"  Boxes: {n}, Analyzed: {n_analyze}, Active: {n_active}, Target: 0,\n"
                    f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f}, Max EI Active: {ei_max_active:.6f},\n"
                    f"  Max EI Active Box Width: {w_max_ei_active}, Terminating partitioning."
                )
            # Uncomment this for 2D animations
            if logMask:
                log[f"p{partition}"] = {
                    "bounds_L": np.asarray(bounds_L),
                    "bounds_U": np.asarray(bounds_U),
                    "target_boxes_mask": np.zeros((n,), dtype=bool),
                }
            if logMask:
                log['ei_max'] = float(ei_max_active)
            return BOResult(X=np.asarray(best_x_active), log=log)
        else:
            # Ensure the box with the highest active EI is also a target box.
            idx_best_global = int(target_idx[idx_best_local])
            target_boxes_mask[idx_best_global] = True
            n_target = int(xp.sum(target_boxes_mask))

        # Calculate position of the best point in next partition
        idx_best_global_next = int(xp.sum(target_boxes_mask[:idx_best_global]))
        
        # Update maximum width of active boxes
        w_max = xp.max(bounds_U[target_boxes_mask] - bounds_L[target_boxes_mask], axis=0)

        if verbose:
            print(
                f"Partition {partition}/{max_partitions-1}, \n"
                f"  Boxes: {n}, Analyzed: {n_analyze}, Active: {n_active}, Target: {n_target},\n"
                f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f}, Max EI Active: {ei_max_active:.6f},\n" 
                f"  Max Width: {w_max}."
            )

        # Uncomment this for 2D animations
        if logMask:
            log[f"p{partition}"] = {
                "bounds_L": np.asarray(bounds_L),
                "bounds_U": np.asarray(bounds_U),
                "target_boxes_mask": np.asarray(target_boxes_mask),
            }
        
        # Update partition count
        partition += 1

        # Split active boxes (don't if its the last partition)
        if partition < max_partitions:
            bounds_L, bounds_U = split_boxes(bounds_L, bounds_U, target_boxes_mask, w, n, d, backend=backend, validation=validation)

    # Store final log info
    if logMask:
        log['ei_max'] = float(ei_max_active)

    return BOResult(X=np.asarray(best_x_active), log=log)
