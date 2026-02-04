# Run with: python run_bo.py <N>
# Import libraries
print("Starting")

import sys
import numpy as np
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
bounds = np.array([[0.0, 1.0],
                   [0.0, 1.0]])

# Define precision
N = int(sys.argv[1])  # Points per dimension
epsilon = 1 / (N - 1)
print(f"Using precision: {epsilon}")

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

# EI utilities (numpy)
sqrt2 = np.sqrt(2.0)
inv_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)

def erf_approx(x):
    """
    Approximate erf(x) using Abramowitz & Stegun 7.1.26.
    Max error ~1.5e-7.
    """
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-ax * ax)
    return sign * y

def norm_cdf(z):
    return 0.5 * (1.0 + erf_approx(z / sqrt2))

def norm_pdf(z):
    return inv_sqrt2pi * np.exp(-0.5 * z * z)

def expected_improvement(mu, sigma, y_min):
    """
    Compute Expected Improvement (EI) for a minimization objective.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    y_min = np.asarray(y_min)

    safe_sigma = np.where(sigma == 0, 1.0, sigma)
    Z = (y_min - mu) / safe_sigma  # to minimize
    ei = (y_min - mu) * norm_cdf(Z) + safe_sigma * norm_pdf(Z)  # to minimize
    return np.where(sigma == 0, 0.0, ei)

def build_grid(bounds, n_per_dim):
    axes = [np.linspace(low, high, n_per_dim) for low, high in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, len(bounds))

def bo_loop(X0, bounds, n_per_dim, gp, f, max_iters):
    # Initialize data
    X_data = X0.copy()

    for iteration in range(max_iters):
        print(f"Iteration {iteration+1}/{max_iters}")
        # Evaluate function at current data points
        y_data = f(X_data)
        print(f"Current training data: X: {X_data}, y: {y_data}")

        # Fit Gaussian Process
        gp.fit(X_data, y_data)
        print("GP fitted")

        # Grid search EI to find next point
        grid = build_grid(bounds, n_per_dim)
        mu_pred, sigma_pred = gp.predict(grid, return_std=True)
        y_min = np.min(y_data)
        ei = expected_improvement(mu_pred, sigma_pred, y_min)
        idx_best = np.argmax(ei)
        X_new = grid[idx_best]
        y_new = f(X_new)
        print(f"Evaluated new point: {X_new} -> {y_new}")

        # Update data
        X_data = np.vstack((X_data, X_new))
        y_data = np.hstack((y_data, y_new))

    return X_data, y_data

# Create initial points
X0 = np.array([[0.25, 0.25],
               [0.25, 0.75],
               [0.75, 0.25],
               [0.75, 0.75]])

# Optimization parameters
max_iters = 5

# Run optimization
res = bo_loop(X0, bounds, N, gp, f, max_iters)

Xres = res[0]
yres = res[1]

idx = np.argmin(yres)
x_best = Xres[idx]
y_best = yres[idx]

print("Optimization finished")
print(f"Best point: {x_best}")
print(f"Best value: {y_best}")
