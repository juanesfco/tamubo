from __future__ import annotations

import numpy as np

from tamubo.utils import BackendName, get_array_module as _array_module

# Relative tolerance for "as wide as the widest dimension": trisected widths that
# should tie differ in the last bits after repeated splitting.
_TIE_RTOL = 1e-9
# GPU memory for the per-chunk slot buffers (lower and upper bounds of all 2d + 1
# possible children); the output itself is allocated once at its exact size.
_SPLIT_CHUNK_BYTES = 4 * 1024**3


def split_boxes(
    bounds_L,
    bounds_U,
    active_boxes_mask,
    domain_width,
    n: int,
    d: int,
    *,
    keep_inactive: bool = True,
    split_ratio: float = 0.0,
    min_width=None,
    return_counts: bool = False,
    backend: BackendName = "auto",
    validation: bool = True,
) -> tuple:
    """
    Split active hyperboxes following the DIRECT partition rule in d dimensions.

    Each split dimension is trisected, widest (in units of ``domain_width``)
    first: the two outer thirds become new boxes and the middle third is split
    along the next dimension; the final middle box comes last. A box split along
    m dimensions yields 2m + 1 children, contiguous in the output.

    Parameters
    ----------
    bounds_L : np.ndarray or cupy.ndarray, shape (n, d)
        Lower bounds of the hyperboxes.
    bounds_U : np.ndarray or cupy.ndarray, shape (n, d)
        Upper bounds of the hyperboxes.
    active_boxes_mask : np.ndarray or cupy.ndarray, shape (n,)
        Boolean mask indicating which boxes to split.
    domain_width : np.ndarray or cupy.ndarray or scalar, shape (d,) or ()
        Per-dimension scale: widths are compared as ``width / domain_width``,
        both to order the splits and to choose which dimensions to split. Pass
        the domain width, or the GP length scales to split in kernel units.
    n : int
        Number of input hyperboxes.
    d : int
        Dimension of the hyperboxes.
    keep_inactive : bool, default=True
        If True, inactive boxes are appended unchanged to the output.
        If False, only split active boxes are returned.
    split_ratio : float in [0, 1], default=0.0
        0 splits every dimension (2d + 1 children per box). Otherwise a box is
        split only along the dimensions whose scaled width is at least
        ``split_ratio`` times its largest scaled width; 1 is DIRECT's
        longest-side rule.
    min_width : array-like of shape (d,), optional
        Used only when ``split_ratio > 0``: dimensions already narrower than
        this are not split (unless every dimension is), so no dimension is
        refined below the resolution the search needs.
    return_counts : bool, default=False
        Also return the number of children of each active box, in order.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).

    Returns
    -------
    bounds_L_out, bounds_U_out : np.ndarray or cupy.ndarray, shape (n_out, d)
        Bounds of the output boxes: children of the active boxes (in order),
        then the unchanged inactive boxes if ``keep_inactive``.
    counts : np.ndarray or cupy.ndarray of int64, shape (nt,)
        Children per active box; only if ``return_counts``.
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    active_boxes_mask = xp.asarray(active_boxes_mask).astype(bool)
    bounds_L = bounds_L.astype(xp.float64, copy=False)
    bounds_U = bounds_U.astype(xp.float64, copy=False)
    domain_width = xp.asarray(domain_width, dtype=xp.float64)
    if min_width is not None:
        min_width = xp.asarray(min_width, dtype=xp.float64)

    # Validate input shapes if requested.
    if validation:
        assert bounds_L.shape == (n, d), f"bounds_L must have shape (n, d), got {bounds_L.shape}"
        assert bounds_U.shape == (n, d), f"bounds_U must have shape (n, d), got {bounds_U.shape}"
        assert active_boxes_mask.shape == (n,), f"active_boxes_mask must have shape (n,), got {active_boxes_mask.shape}"
        assert 0.0 <= split_ratio <= 1.0, f"split_ratio must be in [0, 1], got {split_ratio}"

    # Check if using numpy or cupy
    if xp is np:
        # Serial computation for numpy (more efficient for small n).
        out = _split_boxes_numpy(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d,
                                 keep_inactive=keep_inactive, split_ratio=split_ratio, min_width=min_width)
    else:
        # Vectorized computation for cupy (more efficient for large n).
        out = _split_boxes_vectorized(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, xp,
                                      keep_inactive=keep_inactive, split_ratio=split_ratio, min_width=min_width)
    return out if return_counts else out[:2]


def split_child_counts(bounds_L, bounds_U, domain_width, *, split_ratio=0.0, min_width=None, backend="auto"):
    """
    Number of children ``split_boxes`` gives each box (2m + 1 when split along m
    dimensions), without building them. Arguments as in ``split_boxes``.
    """
    xp = _array_module(backend)
    n, d = int(bounds_L.shape[0]), int(bounds_L.shape[1])
    domain_width = xp.asarray(domain_width, dtype=xp.float64)
    if min_width is not None:
        min_width = xp.asarray(min_width, dtype=xp.float64)
    counts = xp.empty((n,), dtype=xp.int64)
    # In chunks of boxes: the (n, d) temporaries would otherwise rival the boxes themselves.
    chunk = max(1, _SPLIT_CHUNK_BYTES // (4 * d * 8))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        width = xp.asarray(bounds_U[start:end], dtype=xp.float64) - xp.asarray(bounds_L[start:end], dtype=xp.float64)
        split_dims = _dims_to_split(width, width / domain_width, split_ratio, min_width, xp)
        counts[start:end] = 2 * xp.sum(split_dims, axis=1, dtype=xp.int64) + 1
        del width, split_dims
    return counts


def _dims_to_split(width, scaled_width, split_ratio, min_width, xp):
    """
    (n, d) mask of the dimensions to trisect in each box: every dimension when
    split_ratio is 0; otherwise those at least min_width wide whose scaled width
    is within split_ratio of the widest such (all dimensions qualify when every
    one is already narrower than min_width).
    """
    if split_ratio <= 0.0:
        return xp.ones(width.shape, dtype=bool)
    candidate = xp.ones(width.shape, dtype=bool) if min_width is None else width >= min_width
    candidate = candidate | ~xp.any(candidate, axis=1, keepdims=True)
    widest = xp.max(xp.where(candidate, scaled_width, -xp.inf), axis=1, keepdims=True)
    return candidate & (scaled_width >= split_ratio * widest * (1.0 - _TIE_RTOL))


def _split_boxes_numpy(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, *, keep_inactive=True,
                       split_ratio=0.0, min_width=None):
    bounds_L_out = []
    bounds_U_out = []
    counts = []
    # Iterate over each box and split if active, otherwise keep unchanged.
    for i in range(n):
        # If the box is active, split it according to the DIRECT rule.
        if active_boxes_mask[i]:
            # Extract the bounds for the i-th box
            box_L = bounds_L[i] # (d,)
            box_U = bounds_U[i] # (d,)
            # Compute the width and width proportion for each dimension
            box_width = box_U - box_L # (d,)
            box_width_prop = box_width / domain_width # (d,)
            # Dimensions this box is split along.
            split_dims = _dims_to_split(box_width[None, :], box_width_prop[None, :], split_ratio, min_width, np)[0]
            # Determine the order of dimensions to split based on width proportion.
            # DIRECT splits along the largest dimension first so we invert argsort order.
            split_order = np.argsort(box_width_prop)[::-1] # (d,)

            # Initialize the center box which will be updated in-place during the splitting process
            cbox_L = box_L.copy()
            cbox_U = box_U.copy()

            # Split the box along the dimensions in the determined order,
            # creating 2 new boxes for each split (one on each side of the center box).
            for dd in range(d):
                # Split dimension in the order of width proportion (largest first).
                dim_to_split = split_order[dd]
                if not split_dims[dim_to_split]:
                    continue
                # Extract the lower and upper bounds for the dimension to split
                l = box_L[dim_to_split]
                u = box_U[dim_to_split]
                # Extract the width of the box along the dimension to split
                w = box_width[dim_to_split]

                # Compute the split points (1/3 and 2/3) along the dimension to split
                lb = l + w/3
                ub = u - w/3

                # Create the lower box
                lbox_L = cbox_L.copy()
                lbox_U = cbox_U.copy()
                # Upper bound of the lower box is the lower split point
                lbox_U[dim_to_split] = lb

                # Create the upper box
                ubox_L = cbox_L.copy()
                ubox_U = cbox_U.copy()
                # Lower bound of the upper box is the upper split point
                ubox_L[dim_to_split] = ub

                # Update the center box for the next split
                cbox_L[dim_to_split] = lb
                cbox_U[dim_to_split] = ub

                # Append the new boxes to the output lists
                bounds_L_out.append(lbox_L)
                bounds_U_out.append(lbox_U)
                bounds_L_out.append(ubox_L)
                bounds_U_out.append(ubox_U)

            # Finally, append the center box after all splits
            bounds_L_out.append(cbox_L)
            bounds_U_out.append(cbox_U)
            counts.append(2 * int(np.count_nonzero(split_dims)) + 1)

        # If the box is inactive, keep it unchanged.
        else:
            if keep_inactive:
                bounds_L_out.append(bounds_L[i])
                bounds_U_out.append(bounds_U[i])

    # Convert the output lists to arrays and return
    return (np.array(bounds_L_out).reshape(-1, d), np.array(bounds_U_out).reshape(-1, d),
            np.array(counts, dtype=np.int64))


def _split_boxes_vectorized(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, cp, *, keep_inactive=True,
                            split_ratio=0.0, min_width=None):
    # Get the bounds of the active boxes.
    active_bounds_L = bounds_L[active_boxes_mask] # (nt, d)
    active_bounds_U = bounds_U[active_boxes_mask] # (nt, d)

    # Number of active boxes.
    nt = active_bounds_L.shape[0]
    # If there are no active boxes, return the original bounds.
    if nt == 0:
        return bounds_L, bounds_U, cp.zeros((0,), dtype=cp.int64)

    # Each active box has room for 2 * d + 1 children; slots of dimensions it is
    # not split along are dropped.
    stride = 2 * d + 1
    # Number of inactive boxes that will be appended to the output.
    n_inactive = n - nt

    # Compute widths and normalized widths.
    active_width = active_bounds_U - active_bounds_L # (nt, d)
    active_width_prop = active_width / domain_width # (nt, d)

    # Determine the order of dimensions to split for each box based on width proportion.
    # DIRECT splits along the largest dimension first. In our implementation below we
    # will follow this in the big picture but we will start by updating the bounds
    # for the smallest dimensions first.
    split_order = cp.argsort(active_width_prop, axis=1) # (nt, d)
    # Whether each box is split along its dd-th smallest dimension.
    split_sorted = cp.take_along_axis(
        _dims_to_split(active_width, active_width_prop, split_ratio, min_width, cp), split_order, axis=1
    ) # (nt, d)

    # Slot pair p holds the (p+1)-th widest dimension's outer thirds; the last slot
    # is the center box, always kept.
    slot_kept = cp.concatenate(
        [cp.repeat(split_sorted[:, ::-1], 2, axis=1), cp.ones((nt, 1), dtype=bool)], axis=1
    ) # (nt, stride)
    counts = cp.sum(slot_kept, axis=1, dtype=cp.int64) # (nt,)
    n_children = int(cp.sum(counts))
    n_out = n_children + (n_inactive if keep_inactive else 0)
    out_L = cp.empty((n_out, d), dtype=cp.float64)
    out_U = cp.empty((n_out, d), dtype=cp.float64)

    # Build the children in chunks of parents: (chunk, stride, d) slot buffers for every
    # possible child would otherwise dominate peak memory when few slots are kept.
    chunk = max(1, _SPLIT_CHUNK_BYTES // (2 * stride * d * 8))
    filled = 0
    for start in range(0, nt, chunk):
        end = min(start + chunk, nt)
        slots_L, slots_U = _split_slots(
            active_bounds_L[start:end], active_bounds_U[start:end], active_width[start:end],
            split_order[start:end], split_sorted[start:end], d, cp,
        )
        kept = slot_kept[start:end]
        if bool(cp.all(kept)):
            chunk_L = slots_L.reshape((-1, d))
            chunk_U = slots_U.reshape((-1, d))
        else:
            chunk_L = slots_L[kept] # parents in order
            chunk_U = slots_U[kept]
        out_L[filled:filled + chunk_L.shape[0]] = chunk_L
        out_U[filled:filled + chunk_U.shape[0]] = chunk_U
        filled += chunk_L.shape[0]
        del slots_L, slots_U, chunk_L, chunk_U

    if keep_inactive and n_inactive:
        out_L[n_children:] = bounds_L[~active_boxes_mask]
        out_U[n_children:] = bounds_U[~active_boxes_mask]
    return out_L, out_U, counts


def _split_slots(active_bounds_L, active_bounds_U, active_width, split_order, split_sorted, d, cp):
    """(m, 2d + 1, d) bounds of every possible child of m boxes; unused slots hold leftovers."""
    stride = 2 * d + 1
    nt = active_bounds_L.shape[0]
    # Keep active-box output as (nt, stride, d) so updates stay array-based and avoid
    # pairwise advanced indexing with [rows, cols], which is fragile in deferred mode.
    bounds_L_out = cp.repeat(active_bounds_L[:, cp.newaxis, :], repeats=stride, axis=1) # (nt, stride, d)
    bounds_U_out = cp.repeat(active_bounds_U[:, cp.newaxis, :], repeats=stride, axis=1) # (nt, stride, d)

    dim_ids = cp.arange(d, dtype=cp.int64)[cp.newaxis, :] # (1, d)

    # Loop over the dimensions.
    for dd in range(d):
        # For each box, determine the column to update based on the split order.
        cols = split_order[:, dd:dd+1] # (nt, 1)
        # One-third width in the selected split dimension per active box.
        third_w = cp.take_along_axis(active_width, cols, axis=1) / 3.0 # (nt, 1)
        # Lower and upper split points for the selected split dimension.
        lb = cp.take_along_axis(active_bounds_L, cols, axis=1) + third_w # (nt, 1)
        ub = cp.take_along_axis(active_bounds_U, cols, axis=1) - third_w # (nt, 1)
        # One-hot mask selecting the split dimension, for the boxes split along it.
        dim_mask = (cols == dim_ids) & split_sorted[:, dd:dd+1] # (nt, d)

        # Position of the lower-box in this dimension within the rows stride.
        pos = 2 * (d - dd - 1)

        # Lower-boxes upper bounds for this dimension are the center-box lower bounds.
        bounds_U_out[:, pos, :] = cp.where(dim_mask, lb, bounds_U_out[:, pos, :])
        # Upper-boxes lower bounds for this dimension are the center-box upper bounds.
        bounds_L_out[:, pos + 1, :] = cp.where(dim_mask, ub, bounds_L_out[:, pos + 1, :])

        # Remaining rows in each stride inherit center-box bounds for this dimension.
        if pos + 2 < stride:
            trailing_mask = dim_mask[:, cp.newaxis, :] # (nt, 1, d)
            bounds_L_out[:, pos + 2 :, :] = cp.where(
                trailing_mask,
                lb[:, cp.newaxis, :],
                bounds_L_out[:, pos + 2 :, :],
            )
            bounds_U_out[:, pos + 2 :, :] = cp.where(
                trailing_mask,
                ub[:, cp.newaxis, :],
                bounds_U_out[:, pos + 2 :, :],
            )

    return bounds_L_out, bounds_U_out
