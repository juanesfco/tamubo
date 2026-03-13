from __future__ import annotations

from importlib import import_module

import numpy as np

from tamubo.utils import BackendName, resolve_backend


def _array_module(backend: BackendName = "auto"):
    """Return the resolved array module (`numpy` or `cupynumeric`)."""
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return np
    # Import cupynumeric only when it is the selected backend.
    return import_module("cupynumeric")


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
    Split active hyperboxes following the DIRECT partition rule in d dimensions.

    Parameters
    ----------
    bounds_L : np.ndarray or cupynumeric.ndarray, shape (n, d)
        Lower bounds of the hyperboxes.
    bounds_U : np.ndarray or cupynumeric.ndarray, shape (n, d)
        Upper bounds of the hyperboxes.
    active_boxes_mask : np.ndarray or cupynumeric.ndarray, shape (n,)
        Boolean mask indicating which boxes to split.
    domain_width : np.ndarray or cupynumeric.ndarray or scalar, shape (d,) or ()
        Domain widths used to normalize box widths for split ordering.
    n : int
        Number of input hyperboxes.
    d : int
        Dimension of the hyperboxes.
    keep_inactive : bool, default=True
        If True, inactive boxes are appended unchanged to the output.
        If False, only split active boxes are returned.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).

    Returns
    -------
    bounds_L_out : np.ndarray or cupynumeric.ndarray, shape (nt*(2*d+1) + (n-nt), d)
        Lower bounds of the output boxes (split actives + unchanged inactives).
    bounds_U_out : np.ndarray or cupynumeric.ndarray, shape (nt*(2*d+1) + (n-nt), d)
        Upper bounds of the output boxes (split actives + unchanged inactives).
    """
    # Convert inputs to the appropriate array type based on the backend.   
    xp = _array_module(backend)
    active_boxes_mask = xp.asarray(active_boxes_mask).astype(bool)
    bounds_L = bounds_L.astype(xp.float64, copy=False)
    bounds_U = bounds_U.astype(xp.float64, copy=False)
    domain_width = xp.asarray(domain_width, dtype=xp.float64)

    # Validate input shapes if requested.
    if validation:
        assert bounds_L.shape == (n, d), f"bounds_L must have shape (n, d), got {bounds_L.shape}"
        assert bounds_U.shape == (n, d), f"bounds_U must have shape (n, d), got {bounds_U.shape}"
        assert active_boxes_mask.shape == (n,), f"active_boxes_mask must have shape (n,), got {active_boxes_mask.shape}"

    # Check if using numpy or cupynumeric
    if xp is np:
        # Serial computation for numpy (more efficient for small n).
        return _split_boxes_numpy(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, keep_inactive=keep_inactive)
    else:
        # Vectorized computation for cupynumeric (more efficient for large n).
        return _split_boxes_cupynumeric(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, keep_inactive=keep_inactive)


def _split_boxes_numpy(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, *, keep_inactive=True):
    bounds_L_out = []
    bounds_U_out = []
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
        
        # If the box is inactive, keep it unchanged.
        else:
            if keep_inactive:
                bounds_L_out.append(bounds_L[i])
                bounds_U_out.append(bounds_U[i])

    # Convert the output lists to arrays and return
    return np.array(bounds_L_out), np.array(bounds_U_out)


def _split_boxes_cupynumeric(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d, *, keep_inactive=True):
    import cupynumeric as cp

    # Get the bounds of the active boxes.
    active_bounds_L = bounds_L[active_boxes_mask] # (nt, d)
    active_bounds_U = bounds_U[active_boxes_mask] # (nt, d)

    # Number of active boxes.
    nt = active_bounds_L.shape[0]
    # If there are no active boxes, return the original bounds.
    if nt == 0:
        return bounds_L, bounds_U

    # Each active box is split into 2 * d + 1 boxes.
    stride = 2 * d + 1
    # Number of inactive boxes that will be appended to the output.
    n_inactive = n - nt

    # Keep active-box output as (nt, stride, d) so updates stay array-based and avoid
    # pairwise advanced indexing with [rows, cols], which is fragile in deferred mode.
    bounds_L_out = cp.repeat(active_bounds_L[:, cp.newaxis, :], repeats=stride, axis=1) # (nt, stride, d)
    bounds_U_out = cp.repeat(active_bounds_U[:, cp.newaxis, :], repeats=stride, axis=1) # (nt, stride, d)

    # Compute widths and normalized widths.
    active_width = active_bounds_U - active_bounds_L # (nt, d)
    active_width_prop = active_width / domain_width # (nt, d)

    # Determine the order of dimensions to split for each box based on width proportion. 
    # DIRECT splits along the largest dimension first. In our implementation below we 
    # will follow this in the big picture but we will start by updating the bounds 
    # for the smallest dimensions first.
    split_order = cp.argsort(active_width_prop, axis=1) # (nt, d)

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
        # One-hot mask selecting the split dimension for each active box.
        dim_mask = cols == dim_ids # (nt, d)

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

    active_flat_L = bounds_L_out.reshape((nt * stride, d))
    active_flat_U = bounds_U_out.reshape((nt * stride, d))

    if n_inactive == 0 or not keep_inactive:
        return active_flat_L, active_flat_U

    # Avoid concatenate() to reduce peak memory (concatenate creates a second
    # full output copy before references are released).
    total = nt * stride + n_inactive
    out_L = cp.empty((total, d), dtype=cp.float64)
    out_U = cp.empty((total, d), dtype=cp.float64)
    out_L[: nt * stride] = active_flat_L
    out_U[: nt * stride] = active_flat_U
    out_L[nt * stride :] = bounds_L[~active_boxes_mask]
    out_U[nt * stride :] = bounds_U[~active_boxes_mask]
    return out_L, out_U
