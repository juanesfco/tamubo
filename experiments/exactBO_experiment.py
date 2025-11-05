# Print done
print("Starting")

import sys
import tamubo.exactbo as ebo
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import pickle

# Print done
print("Modules Loaded")

# Define model
kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

# Define bounds
bounds = [[0,1],[0,1]]

# Define precision
N = int(sys.argv[1]) # Points per dimension
precision = 1/(N-1)

# Create object
eboloop = ebo.ExactBOLoop(gp, bounds, precision, log=True)

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

# Set in object
eboloop.set_oracle(f)

# Create initial points and evaluate
X0 = np.array([[0,0],[1,1],[0.5,0.5],[0,1],[1,0]])
y0 = f(X0)

# Run optimization
split_type = sys.argv[2]
res = eboloop.run(X0, y0, 10, split_type=split_type)

# Save logs
ebolog = eboloop.log

# Save logs
with open(f"experiments/exactBO_logs/exactBO_log_N{N}_{split_type}.pkl", "wb") as open_file:
    pickle.dump(ebolog, open_file)

# Print done
print("Done")