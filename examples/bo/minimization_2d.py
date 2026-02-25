"""Minimal BO workflow-dispatch example."""

import json
from pathlib import Path

import numpy as np

from tamubo.bo import (
    run_botorch_grid_ei,
    run_botorch_optimize_ei,
    run_sklearn_grid_ei,
)

# 2D domain [0, 1] x [0, 1]
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "minimization_2d_config.json"
LOGS_DIR = SCRIPT_DIR / "logs"
SUPPORTED_FRAMEWORKS = (
    "sklearn_grid_ei",
    "botorch_grid_ei",
    "botorch_optimize_ei",
)


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
    if hasattr(value, "__dict__"):
        return _to_serializable(value.__dict__)
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
        X = X.reshape(1, -1)

    x, y = X[:, 0], X[:, 1]

    # Parameters
    alpha = 0.1
    A = np.array([4, 3, 2])
    B = np.array([0.08, 0.05, 0.02])  # betas
    C = np.array(
        [
            [0.9, 0.3],  # centers (x1,y1)
            [0.1, 0.8],
            [0.6, 0.7],
        ]
    )
    D = 2

    # Compute function value
    val = alpha * (x**2 + y**2)
    for Ai, Bi, (xi, yi) in zip(A, B, C):
        r2 = (x - xi) ** 2 + (y - yi) ** 2
        val -= Ai * np.exp(-r2 / Bi)
    val += D
    return val


def main():
    config = load_config()
    framework = str(config["framework"])
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"Unsupported framework '{framework}'. "
            f"Choose one of: {', '.join(SUPPORTED_FRAMEWORKS)}."
        )

    log_filename = str(config.get("log_filename", ""))
    use_log = bool(log_filename)
    max_iters = int(config["max_iters"])
    validation = bool(config.get("validation", True))
    verbose = bool(config.get("verbose", True))

    X0 = np.array(
        config.get(
            "X0",
            [
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75],
            ],
        ),
        dtype=float,
    )

    if framework == "sklearn_grid_ei":
        result = run_sklearn_grid_ei(
            X0=X0,
            bounds=BOUNDS,
            f=objective,
            max_iters=max_iters,
            grid_resolution=int(config["grid_resolution"]),
            validation=validation,
            verbose=verbose,
            logMask=use_log,
        )
    elif framework == "botorch_grid_ei":
        result = run_botorch_grid_ei(
            X0=X0,
            bounds=BOUNDS,
            f=objective,
            max_iters=max_iters,
            grid_resolution=int(config["grid_resolution"]),
            validation=validation,
            verbose=verbose,
            logMask=use_log,
            device=str(config.get("device", "cuda")),
        )
    else:
        result = run_botorch_optimize_ei(
            X0=X0,
            bounds=BOUNDS,
            f=objective,
            max_iters=max_iters,
            num_restarts=int(config["num_restarts"]),
            raw_samples=int(config["raw_samples"]),
            maxiter=int(config["maxiter"]),
            validation=validation,
            verbose=verbose,
            logMask=use_log,
            device=str(config.get("device", "cuda")),
        )

    idx = int(np.argmin(result.y))
    print(f"framework={framework}")
    print(f"best_x={result.X[idx]}")
    print(f"best_y={result.y[idx]}")
    print(f"backend={result.backend}")

    if use_log:
        saved_path = save_log(result.log, log_filename)
        print(f"log_saved={saved_path}")


if __name__ == "__main__":
    main()
