from __future__ import annotations

import numpy as np
from legate.core import get_legate_runtime

from tamubo.utils import BackendName

from ._cupynumeric import cp, require_cupynumeric_backend


_ROW_COPY_MAX_SELECTED = 4096


def _scalar_int(value) -> int:
    return int(np.asarray(value).item())


def _execute_pending_tasks() -> None:
    get_legate_runtime().issue_execution_fence(block=True)


def _copy_selected_bounds(bounds_L_window, bounds_U_window, local_indices, n_selected: int, d: int):
    if n_selected == _scalar_int(bounds_L_window.shape[0]):
        return bounds_L_window.copy(), bounds_U_window.copy()

    selected_L = cp.empty((n_selected, d), dtype=cp.float64)
    selected_U = cp.empty((n_selected, d), dtype=cp.float64)
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


def _copy_split_runs(
    out_L,
    out_U,
    out_start: int,
    split_L,
    split_U,
    local_indices,
    n_selected: int,
    stride: int,
) -> int:
    local_indices_np = np.asarray(local_indices, dtype=np.int64)
    selected_start = 0
    out_cursor = out_start

    while selected_start < n_selected:
        src_start = int(local_indices_np[selected_start])
        selected_stop = selected_start + 1
        src_stop = src_start + 1

        while selected_stop < n_selected and int(local_indices_np[selected_stop]) == src_stop:
            selected_stop += 1
            src_stop += 1

        run_boxes = src_stop - src_start
        out_stop = out_cursor + run_boxes * stride
        out_L[out_cursor:out_stop] = split_L[src_start * stride : src_stop * stride]
        out_U[out_cursor:out_stop] = split_U[src_start * stride : src_stop * stride]
        out_cursor = out_stop
        selected_start = selected_stop

    return out_cursor


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
    batch_size: int | None = None,
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

    n_active = _scalar_int(cp.sum(active_boxes_mask))

    if n_active == 0:
        if keep_inactive:
            return bounds_L.copy(), bounds_U.copy()
        return (
            cp.empty((0, d), dtype=cp.float64),
            cp.empty((0, d), dtype=cp.float64),
        )

    stride = 2 * d + 1
    if batch_size is not None:
        return _split_boxes_batched(
            bounds_L,
            bounds_U,
            active_boxes_mask,
            domain_width,
            n,
            d,
            stride,
            n_active=n_active,
            keep_inactive=keep_inactive,
            batch_size=batch_size,
        )

    active_idx = cp.where(active_boxes_mask)[0]
    active_bounds_L = bounds_L[active_idx]
    active_bounds_U = bounds_U[active_idx]
    split_bounds_L = cp.repeat(active_bounds_L[:, cp.newaxis, :], repeats=stride, axis=1)
    split_bounds_U = cp.repeat(active_bounds_U[:, cp.newaxis, :], repeats=stride, axis=1)

    active_width = active_bounds_U - active_bounds_L
    split_order = cp.argsort(-(active_width / domain_width), axis=1)
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


def _split_boxes_batched(
    bounds_L,
    bounds_U,
    active_boxes_mask,
    domain_width,
    n: int,
    d: int,
    stride: int,
    *,
    n_active: int,
    keep_inactive: bool,
    batch_size: int,
) -> tuple:
    batch_size = max(1, int(batch_size))
    n_active_out = n_active * stride

    if keep_inactive:
        n_out = n_active_out + (n - n_active)
    else:
        n_out = n_active_out

    out_L = cp.empty((n_out, d), dtype=cp.float64)
    out_U = cp.empty((n_out, d), dtype=cp.float64)

    active_out = 0
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        mask_window = active_boxes_mask[start:stop]
        n_selected = _scalar_int(cp.sum(mask_window))
        if n_selected == 0:
            continue

        bounds_L_window = bounds_L[start:stop]
        bounds_U_window = bounds_U[start:stop]
        local_indices = cp.where(mask_window)[0]

        if n_selected > _ROW_COPY_MAX_SELECTED:
            split_L, split_U = _split_active_boxes_flat(
                bounds_L_window,
                bounds_U_window,
                domain_width,
                d,
                stride,
            )
            active_out = _copy_split_runs(
                out_L,
                out_U,
                active_out,
                split_L,
                split_U,
                local_indices,
                n_selected,
                stride,
            )
        else:
            chunk_bounds_L, chunk_bounds_U = _copy_selected_bounds(
                bounds_L_window,
                bounds_U_window,
                local_indices,
                n_selected,
                d,
            )
            chunk_L, chunk_U = _split_active_boxes_flat(
                chunk_bounds_L,
                chunk_bounds_U,
                domain_width,
                d,
                stride,
            )
            out_start = active_out
            out_stop = out_start + n_selected * stride
            out_L[out_start:out_stop] = chunk_L
            out_U[out_start:out_stop] = chunk_U
            active_out = out_stop
        _execute_pending_tasks()

    if not keep_inactive:
        return out_L, out_U

    inactive_offset = n_active_out
    inactive_out = inactive_offset

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        mask_window = ~active_boxes_mask[start:stop]
        n_selected = _scalar_int(cp.sum(mask_window))
        if n_selected == 0:
            continue

        bounds_L_window = bounds_L[start:stop]
        bounds_U_window = bounds_U[start:stop]
        local_indices = cp.where(mask_window)[0]
        chunk_L, chunk_U = _copy_selected_bounds(
            bounds_L_window,
            bounds_U_window,
            local_indices,
            n_selected,
            d,
        )
        out_stop = inactive_out + n_selected
        out_L[inactive_out:out_stop] = chunk_L
        out_U[inactive_out:out_stop] = chunk_U
        inactive_out = out_stop
        _execute_pending_tasks()

    return out_L, out_U


def _split_active_boxes_flat(active_bounds_L, active_bounds_U, domain_width, d: int, stride: int):
    n_active = int(active_bounds_L.shape[0])
    split_bounds_L = cp.repeat(active_bounds_L[:, cp.newaxis, :], repeats=stride, axis=1)
    split_bounds_U = cp.repeat(active_bounds_U[:, cp.newaxis, :], repeats=stride, axis=1)

    active_width = active_bounds_U - active_bounds_L
    split_order = cp.argsort(-(active_width / domain_width), axis=1)
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

    return (
        split_bounds_L.reshape((n_active * stride, d)),
        split_bounds_U.reshape((n_active * stride, d)),
    )
