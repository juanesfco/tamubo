from __future__ import annotations

import cupynumeric as cp
import numpy as np

from .vectorized_bounds import ei_bounds, mu_bounds, rbf_k_bounds, sigma_bounds
from .vectorized_ei import expected_improvement
from .vectorized_partition import split_boxes


def _evaluate_objective(f, x):
    """Evaluate objective and always return a 1D numpy array."""

    x = np.asarray(x)
    x_eval = x.reshape(1, -1) if x.ndim == 1 else x
    return np.asarray(f(x_eval)).ravel()


def partition_loop_cupynumeric(
    x_data,
    bounds,
    epsilon,
    gp,
    max_partitions,
    *,
    verbose=False,
):
    """Partition loop using cupynumeric vectorized kernels."""

    x_cp = cp.asarray(x_data)

    bounds_l = cp.asarray([bounds[:, 0]], dtype=cp.float64)
    bounds_u = cp.asarray([bounds[:, 1]], dtype=cp.float64)

    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params["k1__k1__constant_value"]
    length_scale = gp_kernel_params["k1__k2__length_scale"]
    alpha = cp.asarray(gp.alpha_)
    y_train_std = gp._y_train_std
    y_train_mean = gp._y_train_mean
    chol_l = cp.asarray(gp.L_)
    y_min = cp.min(gp.y_train_)
    y_min_unscaled = y_min * y_train_std + y_train_mean

    n_train = x_data.shape[0]
    d = bounds.shape[0]
    domain_width = bounds_u[0] - bounds_l[0]
    epsilon = cp.asarray(epsilon, dtype=cp.float64)
    if epsilon.ndim == 0:
        epsilon = cp.full((d,), epsilon, dtype=cp.float64)
    elif epsilon.shape != (d,):
        raise ValueError(f"epsilon must be scalar or shape ({d},), got {tuple(epsilon.shape)}")

    partition = 0
    w_max = domain_width.copy()
    ei_max = 0
    active_boxes_mask = cp.ones((1,), dtype=bool)

    while partition < max_partitions and cp.any(w_max > epsilon):
        n = bounds_l.shape[0]

        k_lo = cp.zeros((n, n_train))
        k_hi = cp.zeros((n, n_train))
        for i in range(n_train):
            xi = x_cp[i]
            k_lo[:, i], k_hi[:, i] = rbf_k_bounds(
                bounds_l.ravel(),
                bounds_u.ravel(),
                xi,
                n,
                d,
                sigma_f_2,
                length_scale,
                validation=False,
            )

        mu_lo, mu_hi = mu_bounds(
            alpha,
            k_lo,
            k_hi,
            n,
            n_train,
            y_train_mean,
            y_train_std,
            validate=False,
        )
        sig_lo, sig_hi = sigma_bounds(
            k_lo,
            k_hi,
            chol_l,
            n,
            n_train,
            sigma_f_2,
            y_train_std,
            validate=False,
        )
        _, ei_hi = ei_bounds(
            mu_lo,
            mu_hi,
            sig_lo,
            sig_hi,
            y_min,
            y_train_mean,
            y_train_std,
            validate=False,
        )

        idx_max_ei_hi = cp.argmax(ei_hi)
        max_ei_hi_box_l = bounds_l[idx_max_ei_hi, :]
        max_ei_hi_box_u = bounds_u[idx_max_ei_hi, :]
        max_ei_hi_box_center = (max_ei_hi_box_l + max_ei_hi_box_u) / 2.0

        mu_pred, sigma_pred = gp.predict(np.array(max_ei_hi_box_center).reshape(1, -1), return_std=True)
        ei_center = expected_improvement(mu_pred[0], sigma_pred[0], y_min_unscaled)
        ei_max = max(ei_max, float(np.asarray(ei_center).ravel()[0]))

        active_boxes_mask = ei_hi > ei_max
        if not bool(cp.any(active_boxes_mask)):
            idx_max = cp.argmax(ei_hi)
            active_boxes_mask = cp.zeros(n, dtype=bool)
            active_boxes_mask[idx_max] = True

        w_max = cp.max(bounds_u[active_boxes_mask] - bounds_l[active_boxes_mask], axis=0)
        partition += 1

        if partition < max_partitions and bool(cp.any(w_max > epsilon)):
            bounds_l, bounds_u = split_boxes(bounds_l, bounds_u, active_boxes_mask, domain_width, n, d)

        if verbose:
            print(
                f" Partition {partition}/{max_partitions}, Boxes: {n}, "
                f"Active: {cp.sum(active_boxes_mask)}, Max EI: {ei_max:.6f}, Max Width: {w_max}"
            )

    bound_u_active = bounds_u[active_boxes_mask]
    bound_l_active = bounds_l[active_boxes_mask]
    center_active = (bound_l_active + bound_u_active) / 2.0
    mu_active, sigma_active = gp.predict(np.array(center_active), return_std=True)
    ei_active = expected_improvement(cp.asarray(mu_active), cp.asarray(sigma_active), y_min_unscaled)
    idx_best = cp.argmax(ei_active)
    best_x = center_active[idx_best]

    return np.asarray(best_x)


def exactbo_loop_cupynumeric(
    x0,
    bounds,
    epsilon,
    gp,
    f,
    max_iters,
    max_partitions,
    *,
    verbose=False,
):
    """Outer ExactBO loop using cupynumeric partition search."""

    x_data = np.asarray(x0).copy()

    for iteration in range(max_iters):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iters}")

        y_data = _evaluate_objective(f, x_data)
        gp.fit(x_data, y_data)

        x_new = partition_loop_cupynumeric(
            x_data,
            bounds,
            epsilon,
            gp,
            max_partitions,
            verbose=verbose,
        )

        y_new = _evaluate_objective(f, x_new)
        x_data = np.vstack((x_data, x_new))
        y_data = np.hstack((y_data, y_new))

        if verbose:
            print(f"Evaluated new point: {x_new} -> {y_new}")

    return x_data, y_data
