"""Minimal ExactBO backend-dispatch example."""

import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.exactbo import exactbo

# 2D domain [0, 1] x [0, 1]
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "minimization_2d_config.json"
LOGS_DIR = SCRIPT_DIR / "logs"


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config in {config_path} must be a JSON object.")
    return config


def save_log(log: dict | None, log_filename: str) -> Path:
    log_path = LOGS_DIR / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(log), f, indent=2)
    return log_path

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
    config = load_config()
    log_filename = config.get("log_filename")

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

    result = exactbo(
        X0=X0,
        bounds=BOUNDS,
        epsilon_X=config["epsilon_X"],
        epsilon_ei=config["epsilon_ei"],
        gp=gp,
        f=objective,
        max_iters=int(config["max_iters"]),
        max_partitions=int(config["max_partition"]),
        backend="auto",
        validation=True,
        verbose=True,
        logMask=True if log_filename else False,
    )

    idx = int(np.argmin(result.y))
    print(f"best_x={result.X[idx]}")
    print(f"best_y={result.y[idx]}")
    print(f"backend={result.backend}")
    
    if log_filename:
        saved_path = save_log(result.log, str(log_filename))
        print(f"log_saved={saved_path}")

if __name__ == "__main__":
    main()
