# Run with: python true_minimum.py <N>
print("Starting")

import sys
import numpy as np

print("Libraries loaded")

# Define bounds for 2D problem
bounds = np.array([[0.0, 1.0],
                   [0.0, 1.0]])

# Grid resolution (points per dimension)
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 201
print(f"Using grid: {N}x{N} points")

# Function to minimize
## f(x,y)= \alpha*(x^2 + y^2) - \sum_{i=1}^3 A_i \exp \left( -\frac{(x - Cx_i)^2 + (y - Cy_i)^2}{B_i} \right) + D
def f(X):
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    x, y = X[:, 0], X[:, 1]

    # Parameters
    alpha = 0.1
    A = np.array([4, 3, 2])
    B = np.array([0.08, 0.05, 0.02])  # betas
    C = np.array([[0.9, 0.3],      # centers (x1,y1)
                  [0.1, 0.8],
                  [0.6, 0.7]])
    D = 2

    # Compute function value
    val = alpha * (x**2 + y**2)
    for Ai, Bi, (xi, yi) in zip(A, B, C):
        r2 = (x - xi)**2 + (y - yi)**2
        val -= Ai * np.exp(-r2 / Bi)
    val += D
    return val

def build_grid(bounds, n_per_dim):
    axes = [np.linspace(low, high, n_per_dim) for low, high in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, len(bounds))

# Brute-force search on the grid
grid = build_grid(bounds, N)
vals = f(grid)
idx = np.argmin(vals)
x_best = grid[idx]
y_best = vals[idx]

print("Grid search finished")
print(f"Best point: {x_best}")
print(f"Best value: {y_best}")
