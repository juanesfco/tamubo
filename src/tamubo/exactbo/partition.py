from __future__ import annotations

from tamubo.utils import BackendName

from ._cupynumeric import cp, require_cupynumeric_backend


def split_boxes(
    bounds_L,
    bounds_U,
    active_boxes_mask,
    domain_width,
    n: int,
    d: int,
    *,
    keep_inactive: bool = True,
    backend: BackendName = "auto",
    validation: bool = True,
) -> tuple:
    """
    Split active hyperboxes following the DIRECT partition rule in ``d`` dimensions.

    Active boxes are emitted first, in contiguous ``2*d + 1`` blocks:
    ``[lower_0, upper_0, lower_1, upper_1, ..., center]``.
    """
    require_cupynumeric_backend(backend)

    bounds_L = cp.asarray(bounds_L, dtype=cp.float64)
    bounds_U = cp.asarray(bounds_U, dtype=cp.float64)
    active_boxes_mask = cp.asarray(active_boxes_mask, dtype=bool)
    domain_width = cp.asarray(domain_width, dtype=cp.float64)

    if validation:
        if bounds_L.shape != (n, d) or bounds_U.shape != (n, d):
            raise ValueError("bounds_L and bounds_U must have shape (n, d).")
        if active_boxes_mask.shape != (n,):
            raise ValueError("active_boxes_mask must have shape (n,).")

    active_bounds_L = bounds_L[active_boxes_mask]
    active_bounds_U = bounds_U[active_boxes_mask]
    n_active = int(active_bounds_L.shape[0])

    if n_active == 0:
        if keep_inactive:
            return bounds_L.copy(), bounds_U.copy()
        return (
            cp.empty((0, d), dtype=cp.float64),
            cp.empty((0, d), dtype=cp.float64),
        )

    stride = 2 * d + 1
    split_bounds_L = cp.repeat(active_bounds_L[:, cp.newaxis, :], repeats=stride, axis=1)
    split_bounds_U = cp.repeat(active_bounds_U[:, cp.newaxis, :], repeats=stride, axis=1)

    active_width = active_bounds_U - active_bounds_L
    split_order = cp.argsort(active_width / domain_width, axis=1)[:, ::-1]
    dim_ids = cp.arange(d, dtype=cp.int64)[cp.newaxis, :]

    for rank in range(d):
        cols = split_order[:, rank : rank + 1]
        third_width = cp.take_along_axis(active_width, cols, axis=1) / 3.0
        lower_third = cp.take_along_axis(active_bounds_L, cols, axis=1) + third_width
        upper_third = cp.take_along_axis(active_bounds_U, cols, axis=1) - third_width
        dim_mask = cols == dim_ids

        lower_row = 2 * rank
        upper_row = lower_row + 1

        split_bounds_U[:, lower_row, :] = cp.where(
            dim_mask,
            lower_third,
            split_bounds_U[:, lower_row, :],
        )
        split_bounds_L[:, upper_row, :] = cp.where(
            dim_mask,
            upper_third,
            split_bounds_L[:, upper_row, :],
        )

        if upper_row + 1 < stride:
            tail_mask = dim_mask[:, cp.newaxis, :]
            split_bounds_L[:, upper_row + 1 :, :] = cp.where(
                tail_mask,
                lower_third[:, cp.newaxis, :],
                split_bounds_L[:, upper_row + 1 :, :],
            )
            split_bounds_U[:, upper_row + 1 :, :] = cp.where(
                tail_mask,
                upper_third[:, cp.newaxis, :],
                split_bounds_U[:, upper_row + 1 :, :],
            )

    active_flat_L = split_bounds_L.reshape((n_active * stride, d))
    active_flat_U = split_bounds_U.reshape((n_active * stride, d))

    if not keep_inactive:
        return active_flat_L, active_flat_U

    inactive_bounds_L = bounds_L[~active_boxes_mask]
    inactive_bounds_U = bounds_U[~active_boxes_mask]
    if int(inactive_bounds_L.shape[0]) == 0:
        return active_flat_L, active_flat_U

    return (
        cp.concatenate((active_flat_L, inactive_bounds_L), axis=0),
        cp.concatenate((active_flat_U, inactive_bounds_U), axis=0),
    )
