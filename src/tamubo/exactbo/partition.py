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
        return _split_boxes_numpy(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d)
    else:
        # Vectorized computation for cupynumeric (more efficient for large n).
        return _split_boxes_cupynumeric(bounds_L, bounds_U, active_boxes_mask, domain_width, d)


def _split_boxes_numpy(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d):
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
            bounds_L_out.append(bounds_L[i])
            bounds_U_out.append(bounds_U[i])

    # Convert the output lists to arrays and return
    return np.array(bounds_L_out), np.array(bounds_U_out)


def _split_boxes_cupynumeric(bounds_L, bounds_U, active_boxes_mask, domain_width, d):
    import cupynumeric as cp

    # Split active boxes from inactive boxes.
    active_bounds_L = bounds_L[active_boxes_mask] # (nt, d)
    active_bounds_U = bounds_U[active_boxes_mask] # (nt, d)
    inactive_bounds_L = bounds_L[~active_boxes_mask] # ((n-nt), d)
    inactive_bounds_U = bounds_U[~active_boxes_mask] # ((n-nt), d)

    # Number of active boxes
    nt = active_bounds_L.shape[0]
    # If there are no active boxes, return the original bounds unchanged.
    if nt == 0:
        return bounds_L, bounds_U

    # Prepare the output arrays for the split boxes. Each active box will 
    # generate 2*d new boxes plus 1 center box.
    bounds_L_out = cp.repeat(active_bounds_L, repeats=2 * d + 1, axis=0) # (nt*(2*d+1), d)
    bounds_U_out = cp.repeat(active_bounds_U, repeats=2 * d + 1, axis=0) # (nt*(2*d+1), d)
    # Compute the width of each active box along each dimension and 
    # the proportion of the width relative to the domain width.
    boxes_width = cp.repeat(cp.subtract(active_bounds_U, active_bounds_L), repeats=2 * d + 1, axis=0) # (nt*(2*d+1), d)
    boxes_width_prop = cp.divide(boxes_width, domain_width) # (nt*(2*d+1), d)
    # Determine the order of dimensions to split for each box based on width proportion. 
    # DIRECT splits along the largest dimension first. In our implementation below we will follow this
    # in the big picture but we will start by updating the bounds for the smallest dimensions first.
    boxes_split_order = cp.argsort(boxes_width_prop, axis=1) # (nt*(2*d+1), d)

    # Perform the splitting for each active box according to the determined order.
    for dd in range(d):
        # Column mask for dimension to split in the current iteration.
        # We will modify bounds along smallest dimension first.
        mask_order = boxes_split_order == dd # (nt*(2*d+1), d)

        # Row and column indices that correspond to the current dimension being split, 
        # used for indexing the output bounds.
        rows_mask_order, cols_mask_order = cp.where(mask_order) # (nt*(2*d+1),) both

        # Extract the bounds and widths along the current dimension.
        l = bounds_L_out[mask_order] # (nt*(2*d+1),)
        u = bounds_U_out[mask_order] # (nt*(2*d+1),)
        w = boxes_width[mask_order] # (nt*(2*d+1),)

        # Compute the split points (1/3 and 2/3) along the dimension to split.
        lb = l + w/3 # (nt*(2*d+1),)
        ub = u - w/3 # (nt*(2*d+1),)

        # This masks identify which of the 2*d+1 boxes (per original active box) bounds we will modify.
        mask_L = cp.ones(2 * d + 1, dtype=bool) # (2*d+1,)
        mask_U = cp.ones(2 * d + 1, dtype=bool) # (2*d+1,)
        # The smallest dimensions will have the smallest amount of bounds modified. And the opposite too.
        # Center box bounds will always be modified.
        mask_L[: 2 * (d - dd - 1)] = False
        mask_L[2 * (d - dd - 1)] = False
        mask_U[: 2 * (d - dd - 1)] = False
        mask_U[2 * (d - dd - 1) + 1] = False
        # Tile masks to match the shape of the output bounds for all active boxes.
        mask_L_out = cp.tile(mask_L, nt) # (nt*(2*d+1),) -> sum(mask_L_out) = nt*2*(dd+1)
        mask_U_out = cp.tile(mask_U, nt) # (nt*(2*d+1),) -> sum(mask_U_out) = nt*2*(dd+1)

        # Rows and columns that will actually be modified in the output bounds
        rows_l = rows_mask_order[mask_L_out]
        rows_u = rows_mask_order[mask_U_out]
        cols_l = cols_mask_order[mask_L_out]
        cols_u = cols_mask_order[mask_U_out]
        # Values to update the bounds with
        vals_l = lb[mask_L_out]
        vals_u = ub[mask_U_out]

        # The size of this mask matches with the number of bounds to modify: sum(mask_L) = sum(mask_U).  
        mask_subs = cp.ones(2 * (dd + 1), dtype=bool) # (2*(dd+1),)
        # The first box (not the center box) will be modified differently.
        mask_subs[0] = False 
        # Tile the mask to match the actual number of bounds: sum(mask_L_out) = sum(mask_U_out) = nt*2*(dd+1).
        mask_subs_out = cp.tile(mask_subs, nt) # (nt*2*(dd+1),)

        # Most of the bounds will be updated with the corresponding split point (L->lb or U->ub) 
        # but some one of them (the upper box) will be updated with the opposite split point (L->ub or U->lb).
        bounds_L_out[rows_l[mask_subs_out], cols_l[mask_subs_out]] = vals_l[mask_subs_out]
        bounds_L_out[rows_l[~mask_subs_out], cols_l[~mask_subs_out]] = vals_u[~mask_subs_out]
        bounds_U_out[rows_u[~mask_subs_out], cols_u[~mask_subs_out]] = vals_l[~mask_subs_out]
        bounds_U_out[rows_u[mask_subs_out], cols_u[mask_subs_out]] = vals_u[mask_subs_out]

    if inactive_bounds_L.shape[0] == 0:
        return bounds_L_out, bounds_U_out

    # Concatenate the split active boxes with the unchanged inactive boxes to form the final output.
    bounds_L_out = cp.concatenate([bounds_L_out, inactive_bounds_L], axis=0)
    bounds_U_out = cp.concatenate([bounds_U_out, inactive_bounds_U], axis=0)
    return bounds_L_out, bounds_U_out