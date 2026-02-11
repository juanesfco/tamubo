from __future__ import annotations

import cupynumeric as cp
import numpy as np

from .vectorized_partition import split_boxes
from .vectorized_bounds import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
from .vectorized_ei import expected_improvement


def _evaluate_objective(f, x):
    """Evaluate objective and always return a 1D numpy array."""
    x = np.asarray(x)
    x_eval = x.reshape(1, -1) if x.ndim == 1 else x
    return np.asarray(f(x_eval)).ravel()


def partition_loop_cupynumeric(X_data, bounds, epsilon, gp, max_partitions, *, verbose=False):
    """
    Find the next candidate point by iteratively partitioning the search box.

    The routine builds EI lower/upper bounds on each hyperbox, keeps only active
    boxes whose upper EI can still beat the current best EI, and splits active
    boxes until `max_partitions` is reached or the active-box widths are below
    `epsilon`. The returned point is the center of the active box with highest
    exact EI.

    Args:
        X_data: Training inputs with shape (N, d), used by the fitted GP.
        bounds: Search-space bounds with shape (d, 2), [lower, upper] per dim.
        epsilon: Per-dimension stopping tolerance for active-box widths.
        gp: Fitted scikit-learn GaussianProcessRegressor-like model.
        max_partitions: Maximum number of partition/split iterations.
        verbose: If True, print partition diagnostics.

    Returns:
        best_x: Candidate point with shape (d,) selected for the next evaluation.
    """
    # Keep host-side state in numpy; move only kernel inputs to CuPyNumeric.
    X_data = np.asarray(X_data, dtype=np.float64)

    # X to CuPyNumeric
    X_cp = cp.asarray(X_data, dtype=cp.float64)  # (N,d)

    # Initialize boxes
    ## One row per box (initially one box)
    ## One column per dimension
    bounds_L = cp.asarray([bounds[:, 0]], dtype=cp.float64)  # (n,d)
    bounds_U = cp.asarray([bounds[:, 1]], dtype=cp.float64)  # (n,d)

    # GP hyperparameters
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params["k1__k1__constant_value"]
    length_scale = gp_kernel_params["k1__k2__length_scale"]
    alpha = cp.asarray(gp.alpha_, dtype=cp.float64)
    y_train_std = gp._y_train_std
    y_train_mean = gp._y_train_mean
    L = cp.asarray(gp.L_, dtype=cp.float64)  # (N,N)
    y_min = cp.min(gp.y_train_)
    y_min_unscaled = y_min * y_train_std + y_train_mean

    # Partition parameters
    N = X_data.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    epsilon = cp.asarray(epsilon, dtype=cp.float64)
    if epsilon.ndim == 0:
        epsilon = cp.full((d,), epsilon, dtype=cp.float64)
    elif epsilon.shape != (d,):
        raise ValueError(f"epsilon must be scalar or shape ({d},), got {tuple(epsilon.shape)}")
    partition = 0
    w_max = w.copy()
    ei_max = 0
    active_boxes_mask = cp.ones((1,), dtype=bool)

    while partition < max_partitions and cp.any(w_max > epsilon):
        # Total number of boxes
        n = bounds_L.shape[0]

        # Bounds
        ## Kernel
        K_lo = cp.zeros((n, N), dtype=cp.float64)
        K_hi = cp.zeros((n, N), dtype=cp.float64)
        for i in range(N):
            xi = X_cp[i]
            K_lo[:, i], K_hi[:, i] = rbf_k_bounds(
                bounds_L.ravel(),
                bounds_U.ravel(),
                xi,
                n,
                d,
                sigma_f_2,
                length_scale,
                False,
            )
        ## Mean
        mu_lo, mu_hi = mu_bounds(alpha, K_lo, K_hi, n, N, y_train_mean, y_train_std, False)  # (n,) both
        ## Sigma
        sig_lo, sig_hi = sigma_bounds(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std, False)  # (n,) both
        ## EI
        ei_lo, ei_hi = ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, y_min, y_train_mean, y_train_std, False)  # (n,) both

        # Compute actual EI in the center of the hyperbox with highest upper EI bound
        idx_max_ei_hi = cp.argmax(ei_hi)
        max_ei_hi_box_L = bounds_L[idx_max_ei_hi, :]  # (d,)
        max_ei_hi_box_U = bounds_U[idx_max_ei_hi, :]  # (d,)
        max_ei_hi_box_center = (max_ei_hi_box_L + max_ei_hi_box_U) / 2.0  # (d,)
        mu_pred, sigma_pred = gp.predict(np.array(max_ei_hi_box_center).reshape(1, -1), return_std=True)
        ei_center = expected_improvement(mu_pred[0], sigma_pred[0], y_min_unscaled)
        ei_max = max(ei_max, float(np.asarray(ei_center).ravel()[0]))

        # Active boxes are the ones where ei_hi is higher than ei_max
        active_boxes_mask = ei_hi > ei_max  # (n,)
        if not bool(cp.any(active_boxes_mask)):
            idx_max = cp.argmax(ei_hi)
            active_boxes_mask = cp.zeros(n, dtype=bool)
            active_boxes_mask[idx_max] = True

        # Update maximum width of active boxes
        w_max = cp.max(bounds_U[active_boxes_mask] - bounds_L[active_boxes_mask], axis=0)

        # Update partition count
        partition += 1

        # Split active boxes (don't if its the last partition)
        if partition < max_partitions and bool(cp.any(w_max > epsilon)):
            bounds_L, bounds_U = split_boxes(bounds_L, bounds_U, active_boxes_mask, w, n, d)

        # Print status
        if verbose:
            print(
                f" Partition {partition}/{max_partitions}, Boxes: {n}, "
                f"Active: {cp.sum(active_boxes_mask)}, Max EI: {ei_max:.6f}, Max Width: {w_max}"
            )

    # Check EI in the center of the active boxes and return the best point
    bound_U_active = bounds_U[active_boxes_mask]  # (m,d)
    bound_L_active = bounds_L[active_boxes_mask]  # (m,d)
    center_active = (bound_L_active + bound_U_active) / 2.0  # (m,d)
    mu_active, sigma_active = gp.predict(np.array(center_active), return_std=True)  # (m,) both
    ei_active = expected_improvement(cp.asarray(mu_active), cp.asarray(sigma_active), y_min_unscaled)  # (m,)
    idx_best = cp.argmax(ei_active)
    best_x = center_active[idx_best]

    return np.asarray(best_x)


def exactbo_loop_cupynumeric(X0, bounds, epsilon, gp, f, max_iters, max_partitions, *, verbose=False):
    """
    Run the outer ExactBO optimization loop.

    At each iteration, this function evaluates the objective on current data,
    fits the GP, calls `partition_loop_cupynumeric` to select the next point,
    evaluates that point, and appends it to the dataset.

    Args:
        X0: Initial design points with shape (N0, d).
        bounds: Search-space bounds with shape (d, 2), [lower, upper] per dim.
        epsilon: Partition stopping tolerance passed to `partition_loop_cupynumeric`.
        gp: Gaussian process regressor instance used for surrogate fitting.
        f: Objective function that maps inputs to scalar values.
        max_iters: Number of BO outer iterations.
        max_partitions: Max partition steps per BO iteration.
        verbose: If True, print iteration-level diagnostics.

    Returns:
        X_data: Final evaluated inputs with shape (N0 + max_iters, d).
        y_data: Final objective values aligned with `X_data`.
    """
    # Initialize data
    X_data = np.asarray(X0, dtype=np.float64).copy()  # (N,d), initially N=N0

    for iteration in range(max_iters):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iters}")
        # Evaluate function at current data points
        y_data = _evaluate_objective(f, X_data)  # (N,)
        if verbose:
            print(f"Current training data: X: {X_data}, y: {y_data}")

        # Fit Gaussian Process
        gp.fit(X_data, y_data)
        if verbose:
            print("GP fitted")

        # Run partitioning to find next point and evaluate it
        X_new = np.asarray(
            partition_loop_cupynumeric(X_data, bounds, epsilon, gp, max_partitions, verbose=verbose),
            dtype=np.float64,
        ).ravel()  # (d,)
        y_new = _evaluate_objective(f, X_new)  # (1,)
        if verbose:
            print(f"Evaluated new point: {X_new} -> {y_new}")

        # Update data
        X_data = np.vstack((X_data, X_new))  # (N+1,d)
        y_data = np.hstack((y_data, y_new))  # (N+1,)

    return X_data, y_data
