# Print starting
print("Starting - Partition Script")

import tamubo.exactbo as ebo
import cupynumeric as cp
import numpy as np
import pickle
from legate.timing import time

# Print modules loaded
print("Modules Loaded")

# Define boxes splitting algorithm
def split_boxes(boxes_bounds_L, boxes_bounds_R, domain_width, n, d):
    boxes_bounds_L_out = cp.repeat(boxes_bounds_L, repeats=2*d+1, axis=0)
    boxes_bounds_R_out = cp.repeat(boxes_bounds_R, repeats=2*d+1, axis=0)
    boxes_mid = cp.repeat((boxes_bounds_L + boxes_bounds_R)/2, repeats=2*d+1, axis=0)
    boxes_width = cp.repeat(cp.subtract(boxes_bounds_R,boxes_bounds_L), repeats=2*d+1, axis=0)
    boxes_width_prop = cp.divide(boxes_width, domain_width)
    boxes_split_order = cp.argsort(boxes_width_prop, axis=1)

    for dd in range(d):
        mask_L = cp.ones(2*d+1)
        mask_L[:2*(d-dd-1)] = 0
        mask_L[2*(d-dd-1)] = 0
        mask_L_out = cp.tile(mask_L,n) == 1

        mask_R = cp.ones(2*d+1)
        mask_R[:2*(d-dd-1)] = 0
        mask_R[2*(d-dd-1)+1] = 0
        mask_R_out = cp.tile(mask_R,n) == 1
        
        mask_order = boxes_split_order == dd
        c = boxes_mid[mask_order]
        w = boxes_width[mask_order]/4
        l = cp.subtract(c, 0.5*w)
        r = cp.add(c, 0.5*w)

        mask_subs = cp.ones(2*(dd+1))
        mask_subs[0] = 0
        mask_subs_out = cp.tile(mask_subs,n) == 1

        rows_mask_order, cols_mask_order = cp.where(mask_order)

        boxes_bounds_L_out[rows_mask_order[mask_L_out][mask_subs_out],cols_mask_order[mask_L_out][mask_subs_out]] = l[mask_L_out][mask_subs_out]
        boxes_bounds_L_out[rows_mask_order[mask_L_out][~mask_subs_out],cols_mask_order[mask_L_out][~mask_subs_out]] = r[mask_R_out][~mask_subs_out]
        boxes_bounds_R_out[rows_mask_order[mask_R_out][~mask_subs_out],cols_mask_order[mask_R_out][~mask_subs_out]] = l[mask_L_out][~mask_subs_out]
        boxes_bounds_R_out[rows_mask_order[mask_R_out][mask_subs_out],cols_mask_order[mask_R_out][mask_subs_out]] = r[mask_R_out][mask_subs_out]
        
    return(boxes_bounds_L_out, boxes_bounds_R_out)

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
        start_time = time()
        new_boxes_bounds_L, new_boxes_bounds_R = split_boxes(boxes_bounds_L, boxes_bounds_R, w, n, d)
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
print("Done - Partition Script")