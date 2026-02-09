# Print starting
#print("Starting - Partition Script")

import tamubo.exactbo as ebo
import cupynumeric as cp
import numpy as np
import pickle
from legate.timing import time

# Print modules loaded
#print("Modules Loaded")

def split_boxes(bounds_L, bounds_U, active_boxes_mask, domain_width, n, d):
    """
    Split active hyperboxes following the DIRECT partition rule in d dimensions.

    Parameters
    ----------
    bounds_L : cupynumeric.ndarray, shape (n, d)
        Lower bounds of the hyperboxes.
    bounds_U : cupynumeric.ndarray, shape (n, d)
        Upper bounds of the hyperboxes.
    active_boxes_mask : cupynumeric.ndarray, shape (n,)
        Boolean mask indicating which boxes to split.
    domain_width : cupynumeric.ndarray or scalar, shape (d,) or ()
        Domain widths used to normalize box widths for split ordering.
    n : int
        Number of input hyperboxes.
    d : int
        Dimension of the hyperboxes.

    Returns
    -------
    bounds_L_out : cupynumeric.ndarray, shape (nt*(2*d+1) + (n-nt), d)
        Lower bounds of the output boxes (split actives + unchanged inactives).
    bounds_U_out : cupynumeric.ndarray, shape (nt*(2*d+1) + (n-nt), d)
        Upper bounds of the output boxes (split actives + unchanged inactives).
    """
    active_boxes_mask = cp.asarray(active_boxes_mask).astype(bool)
    if active_boxes_mask.shape[0] != n:
        raise ValueError("active_boxes_mask must have shape (n,)")

    bounds_L = bounds_L.astype(cp.float64, copy=False)
    bounds_U = bounds_U.astype(cp.float64, copy=False)
    domain_width = cp.asarray(domain_width, dtype=cp.float64)

    active_bounds_L = bounds_L[active_boxes_mask]
    active_bounds_U = bounds_U[active_boxes_mask]
    inactive_bounds_L = bounds_L[~active_boxes_mask]
    inactive_bounds_U = bounds_U[~active_boxes_mask]

    nt = active_bounds_L.shape[0]
    if nt == 0:
        return bounds_L, bounds_U

    bounds_L_out = cp.repeat(active_bounds_L, repeats=2*d+1, axis=0)
    bounds_U_out = cp.repeat(active_bounds_U, repeats=2*d+1, axis=0)
    boxes_mid = cp.repeat((active_bounds_L + active_bounds_U) / 2, repeats=2*d+1, axis=0)
    boxes_width = cp.repeat(cp.subtract(active_bounds_U, active_bounds_L), repeats=2*d+1, axis=0)
    boxes_width_prop = cp.divide(boxes_width, domain_width)
    boxes_split_order = cp.argsort(boxes_width_prop, axis=1)

    for dd in range(d):
        mask_L = cp.ones(2*d+1)
        mask_L[:2*(d-dd-1)] = 0
        mask_L[2*(d-dd-1)] = 0
        mask_L_out = cp.tile(mask_L, nt) == 1

        mask_R = cp.ones(2*d+1)
        mask_R[:2*(d-dd-1)] = 0
        mask_R[2*(d-dd-1)+1] = 0
        mask_R_out = cp.tile(mask_R, nt) == 1

        mask_order = boxes_split_order == dd
        c = boxes_mid[mask_order]
        w = boxes_width[mask_order] / 4
        l = cp.subtract(c, 0.5 * w)
        r = cp.add(c, 0.5 * w)

        mask_subs = cp.ones(2*(dd+1))
        mask_subs[0] = 0
        mask_subs_out = cp.tile(mask_subs, nt) == 1

        rows_mask_order, cols_mask_order = cp.where(mask_order)

        bounds_L_out[rows_mask_order[mask_L_out][mask_subs_out], cols_mask_order[mask_L_out][mask_subs_out]] = l[mask_L_out][mask_subs_out]
        bounds_L_out[rows_mask_order[mask_L_out][~mask_subs_out], cols_mask_order[mask_L_out][~mask_subs_out]] = r[mask_R_out][~mask_subs_out]
        bounds_U_out[rows_mask_order[mask_R_out][~mask_subs_out], cols_mask_order[mask_R_out][~mask_subs_out]] = l[mask_L_out][~mask_subs_out]
        bounds_U_out[rows_mask_order[mask_R_out][mask_subs_out], cols_mask_order[mask_R_out][mask_subs_out]] = r[mask_R_out][mask_subs_out]
    
    if inactive_bounds_L.shape[0] == 0:
        return bounds_L_out, bounds_U_out

    bounds_L_out = cp.concatenate([bounds_L_out, inactive_bounds_L], axis=0)
    bounds_U_out = cp.concatenate([bounds_U_out, inactive_bounds_U], axis=0)
    return bounds_L_out, bounds_U_out

def main():
    # Define bounds
    bounds = [[0,2],[0,2]]

    # Define initial box
    init_box = ebo.Box(bounds,True)

    # Define boxes object
    boxes = ebo.Boxes([init_box])

    # Number of boxes
    n = len(boxes)
    d = init_box.dim
    w = cp.array(init_box.width)

    # Boxes bounds to cupynumeric
    boxes_bounds = boxes.bounds.reshape(n*d,2)
    boxes_bounds_L = cp.array(boxes_bounds[:,0].reshape(n,d))
    boxes_bounds_R = cp.array(boxes_bounds[:,1].reshape(n,d))

    # Start loop to partition a couple of times
    ns = []
    times = []
    Ls = []
    Rs = []
    for _ in range(8):
        n = boxes_bounds_L.shape[0]
        active_boxes_mask = cp.ones(n,dtype=bool)
        start_time = time()
        new_boxes_bounds_L, new_boxes_bounds_R = split_boxes(boxes_bounds_L, boxes_bounds_R, active_boxes_mask, w, n, d)
        end_time = time()
        
        ns.append(n)
        times.append((end_time-start_time)/1e6)
        Ls.append(new_boxes_bounds_L.copy())
        Rs.append(new_boxes_bounds_R.copy())

        boxes_bounds_L = new_boxes_bounds_L
        boxes_bounds_R = new_boxes_bounds_R

    # Results dictionary
    results = {
        'ns': ns,
        'times': times,
        'Ls': Ls,
        'Rs': Rs
    }

    # Save results as pickle
    with open("examples/Data/results_partition_cupynumeric.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()

# Print done
#print("Done - Partition Script")
