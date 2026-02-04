# Run with: CUPYNUMERIC_REPORT_COVERAGE=1 legate --cpus 1 --gpus 1 --show-config run_exactbo_cupynumeric.py <N>
# Import libraries
print("Starting")

import sys
import numpy as np
from legate.timing import time
from loops_cupynumeric import exactbo_loop
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

print("Libraries loaded")

# Define model
## Kernel: Constant * RBF + WhiteKernel (noise)
kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

# Define bounds for 2D problem
## Each row is a dimension, 
## first column is lower bound, 
## second column is upper bound
bounds = np.array([[0.0,1.0],
                   [0.0,1.0]])

# Define precision
N = int(sys.argv[1]) # Points per dimension
epsilon = 1/(N-1)
print(f"Using precision: {epsilon}")

# Function to minimize
## f(x,y)= \alpha*(x^2 + y^2) - \sum_{i=1}^3 A_i \exp \left( -\frac{(x - Cx_i)^2 + (y - Cy_i)^2}{B_i} \right) + D
def f(X):
    if len(X.shape) == 1:
        X = X.reshape(1,-1)

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

# Create initial points
X0 = np.array([[0.25,0.25],
               [0.25,0.75],
               [0.75,0.25],
               [0.75,0.75]])

# Optimization parameters
max_iters = 5
max_partitions = 20

# Run optimization
res = exactbo_loop(X0, bounds, epsilon, gp, f, max_iters, max_partitions)

Xres = res[0]
yres = res[1]

idx = np.argmin(yres)
x_best = Xres[idx]
y_best = yres[idx]

print("Optimization finished")
print(f"Best point: {x_best}")
print(f"Best value: {y_best}")