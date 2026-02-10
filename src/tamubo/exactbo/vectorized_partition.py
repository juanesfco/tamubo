from __future__ import annotations

import cupynumeric as cp


def split_boxes(bounds_l, bounds_u, active_boxes_mask, domain_width, n, d):
    """
    Split active hyperboxes following the DIRECT partition rule in d dimensions.

    Parameters
    ----------
    bounds_l, bounds_u : cupynumeric.ndarray, shape (n, d)
        Lower/upper bounds for each box.
    active_boxes_mask : cupynumeric.ndarray, shape (n,)
        Boolean mask selecting active boxes.
    domain_width : cupynumeric.ndarray or scalar
        Domain widths used for normalized split ordering.
    n : int
        Number of input hyperboxes.
    d : int
        Box dimensionality.
    """

    active_boxes_mask = cp.asarray(active_boxes_mask).astype(bool)
    if active_boxes_mask.shape[0] != n:
        raise ValueError("active_boxes_mask must have shape (n,)")

    bounds_l = bounds_l.astype(cp.float64, copy=False)
    bounds_u = bounds_u.astype(cp.float64, copy=False)
    domain_width = cp.asarray(domain_width, dtype=cp.float64)

    active_bounds_l = bounds_l[active_boxes_mask]
    active_bounds_u = bounds_u[active_boxes_mask]
    inactive_bounds_l = bounds_l[~active_boxes_mask]
    inactive_bounds_u = bounds_u[~active_boxes_mask]

    nt = active_bounds_l.shape[0]
    if nt == 0:
        return bounds_l, bounds_u

    bounds_l_out = cp.repeat(active_bounds_l, repeats=2 * d + 1, axis=0)
    bounds_u_out = cp.repeat(active_bounds_u, repeats=2 * d + 1, axis=0)

    boxes_mid = cp.repeat((active_bounds_l + active_bounds_u) / 2, repeats=2 * d + 1, axis=0)
    boxes_width = cp.repeat(active_bounds_u - active_bounds_l, repeats=2 * d + 1, axis=0)
    boxes_width_prop = boxes_width / domain_width
    boxes_split_order = cp.argsort(boxes_width_prop, axis=1)

    for dd in range(d):
        mask_l = cp.ones(2 * d + 1, dtype=bool)
        mask_l[: 2 * (d - dd - 1)] = False
        mask_l[2 * (d - dd - 1)] = False
        mask_l_out = cp.tile(mask_l, nt)

        mask_r = cp.ones(2 * d + 1, dtype=bool)
        mask_r[: 2 * (d - dd - 1)] = False
        mask_r[2 * (d - dd - 1) + 1] = False
        mask_r_out = cp.tile(mask_r, nt)

        mask_order = boxes_split_order == dd
        c = boxes_mid[mask_order]
        w = boxes_width[mask_order] / 4
        l = c - 0.5 * w
        r = c + 0.5 * w

        mask_subs = cp.ones(2 * (dd + 1), dtype=bool)
        mask_subs[0] = False
        mask_subs_out = cp.tile(mask_subs, nt)

        rows_mask_order, cols_mask_order = cp.where(mask_order)

        rows_l = rows_mask_order[mask_l_out]
        cols_l = cols_mask_order[mask_l_out]
        vals_l = l[mask_l_out]
        rows_r = rows_mask_order[mask_r_out]
        cols_r = cols_mask_order[mask_r_out]
        vals_r = r[mask_r_out]

        bounds_l_out[rows_l[mask_subs_out], cols_l[mask_subs_out]] = vals_l[mask_subs_out]
        bounds_l_out[rows_l[~mask_subs_out], cols_l[~mask_subs_out]] = vals_r[~mask_subs_out]
        bounds_u_out[rows_r[~mask_subs_out], cols_r[~mask_subs_out]] = vals_l[~mask_subs_out]
        bounds_u_out[rows_r[mask_subs_out], cols_r[mask_subs_out]] = vals_r[mask_subs_out]

    if inactive_bounds_l.shape[0] == 0:
        return bounds_l_out, bounds_u_out

    bounds_l_out = cp.concatenate([bounds_l_out, inactive_bounds_l], axis=0)
    bounds_u_out = cp.concatenate([bounds_u_out, inactive_bounds_u], axis=0)
    return bounds_l_out, bounds_u_out
