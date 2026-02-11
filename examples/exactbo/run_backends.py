"""Minimal ExactBO backend-dispatch example."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.exactbo import run_exactbo


# 2D domain [0, 1] x [0, 1]
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

# Function to minimize
## f(x,y)= \alpha*(x^2 + y^2) - \sum_{i=1}^3 A_i \exp \left( -\frac{(x - Cx_i)^2 + (y - Cy_i)^2}{B_i} \right) + D
def objective(X):
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

def main():
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

    X0 = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ],
        dtype=float,
    )

    result = run_exactbo(
        x0=X0,
        bounds=BOUNDS,
        epsilon=0.005,
        gp=gp,
        f=objective,
        max_iters=5,
        max_partitions=20,
        backend="auto",
        verbose=True,
    )

    idx = int(np.argmin(result.y))
    print(f"backend={result.backend.selected}")
    print(f"best_x={result.X[idx]}")
    print(f"best_y={result.y[idx]}")


if __name__ == "__main__":
    main()
