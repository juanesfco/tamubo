# Print done
print("Starting")

import sys
import pickle
import tamubo.exactbo as ebo
import cupynumeric as cp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from legate.timing import time
from partition_cupynumeric import split_boxes

# Print done
print("Modules Loaded")

# Define model
kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

# Define bounds
bounds = [[0,1],[0,1]]

# Define initial box
init_box = ebo.Box(bounds,True)

# Function to minimize
# f(x,y)= \alpha(x^2 + y^2) - \sum_{i=1}^3 A_i \exp \left( -\frac{(x - Cx_i)^2 + (y - Cy_i)^2}{B_i} \right) + D
def f(X):
    x, y = X[:,0], X[:,1]

    # Parameters
    alpha = 0.1
    A  = np.array([4, 3, 2])
    B  = np.array([0.08, 0.05, 0.02])    # betas
    C  = np.array([[0.9, 0.3 ],      # centers (x1,y1)
                   [0.1 , 0.8],
                   [0.6 , 0.7 ]])
    D = 2
    
    # Compute function value
    val = alpha*(x**2 + y**2)
    for Ai, Bi, (xi, yi) in zip(A, B, C):
        r2 = (x - xi)**2 + (y - yi)**2
        val -= Ai * np.exp(-r2 / Bi)
    val += D
    return val

# Create initial points and evaluate
X0 = np.array([[0,0],[1,1],[0.5,0.5],[0,1],[1,0]])
y0 = f(X0)

N = X0.shape[0]

# Train model
gp.fit(X0,y0.ravel())

# Define boxes object
boxes = ebo.Boxes([init_box])

# Number of boxes
n = len(boxes)
d = init_box.dim
w = cp.array(init_box.width)

# Boxes bounds
boxes_bounds = boxes.bounds.reshape(n*d,2)
boxes_bounds_L = boxes_bounds[:,0].reshape(n,d)
boxes_bounds_R = boxes_bounds[:,1].reshape(n,d)

# Loop over different amount of boxes
ns = []
times = []
Ks = []
for p in range(8):
    n = boxes_bounds_L.shape[0]

    # Find kernel bounds (timing it)
    start_time = time()
    #K = np.zeros((n,2*N))
    for i_box in range(n):
    #    L = boxes_bounds_L[i_box]
    #    R = boxes_bounds_R[i_box]
    #    for i_x in range(N):
    #        x = X0[i_x]
    #        KL, KR = ebo.rbf_k_bounds(L,R,x,gp)
    #        K[i_box,2*i_x] = KL
    #        K[i_box,2*i_x+1] = KR
        box = boxes[i_box]
        mu, _ = ebo.mu_bounds(box,gp)

    end_time = time()

    ns.append(n)
    times.append((end_time-start_time)/1e6)
    Ks.append(K)

    boxes_bounds_L, boxes_bounds_R = split_boxes(cp.array(boxes_bounds_L), cp.array(boxes_bounds_R), w, n, d)
    #boxes_bounds_L, boxes_bounds_R = np.array(boxes_bounds_L), np.array(boxes_bounds_R)

# Results dictionary
results = {
    'ns': ns,
    'times': times,
    'Ks': Ks
}

# Save results as pickle
with open("examples/Data/results_bounds_numpy.pkl", "wb") as f:
    pickle.dump(results, f)

# Print done
print("Done")