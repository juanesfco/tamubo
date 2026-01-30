# Print starting
print("Starting")

import tamubo.exactbo as ebo
import numpy as np
import pickle
from legate.timing import time

# Print modules loaded
print("Modules Loaded")

# Define bounds
bounds = [[0,2],[0,2]]

# Define initial box
init_box = ebo.Box(bounds,True)

# Define boxes object
boxes_old = ebo.Boxes([init_box])

# Number of boxes
d = init_box.dim
w = np.array(init_box.width)

# Start loop to partition a couple of times
Ns = []
times = []
Ls = []
Rs = []
for p in range(8):
    n = len(boxes_old)
    boxes = ebo.Boxes()
    start_time = time()
    for box in boxes_old:
        boxes.extend(ebo.split_box(box, 'centered', w))
    end_time = time()

    n_p = len(boxes)
    boxes_bounds = boxes.bounds.reshape(n_p*d,2)
    boxes_bounds_L = np.array(boxes_bounds[:,0].reshape(n_p,d))
    boxes_bounds_R = np.array(boxes_bounds[:,1].reshape(n_p,d))
    
    Ns.append(n)
    times.append((end_time-start_time)/1e6)
    Ls.append(boxes_bounds_L.copy())
    Rs.append(boxes_bounds_R.copy())

    boxes_old = boxes

# Results dictionary
results = {
    'Ns': Ns,
    'times': times,
    'Ls': Ls,
    'Rs': Rs
}

# Save results as pickle
with open("examples/Data/results_partition_numpy.pkl", "wb") as f:
    pickle.dump(results, f)

# Print done
print("Done")