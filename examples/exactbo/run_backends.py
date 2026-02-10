"""Minimal ExactBO backend-dispatch example."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.exactbo import run_exactbo


# 2D domain [0, 1] x [0, 1]
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)


def objective(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    xx = x[:, 0]
    yy = x[:, 1]
    return (xx - 0.2) ** 2 + (yy - 0.7) ** 2 - 0.2 * np.exp(-20 * ((xx - 0.8) ** 2 + (yy - 0.3) ** 2))


def main():
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

    x0 = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ],
        dtype=float,
    )

    result = run_exactbo(
        x0=x0,
        bounds=BOUNDS,
        epsilon=0.05,
        gp=gp,
        f=objective,
        max_iters=5,
        max_partitions=20,
        backend="auto",
    )

    idx = int(np.argmin(result.y))
    print(f"backend={result.backend.selected}")
    print(f"best_x={result.X[idx]}")
    print(f"best_y={result.y[idx]}")


if __name__ == "__main__":
    main()
