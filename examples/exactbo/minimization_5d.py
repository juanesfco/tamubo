"""Minimal ExactBO example for the experiment 2 five-dimensional problem."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.exactbo import exactbo

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exactbo.experiment2.problems import (  # noqa: E402
    load_problem,
    objective_least_squares_5d,
)

CONFIG_PATH = SCRIPT_DIR / "minimization_5d_config.json"
LOGS_DIR = SCRIPT_DIR / "logs"
PROBLEM = load_problem("problem5d")


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


def main():
    config = load_config()
    log_filename = config.get("log_filename")

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.full(PROBLEM.d, 0.5), length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)

    result = exactbo(
        X0=PROBLEM.X0,
        bounds=PROBLEM.bounds,
        epsilon_X=config["epsilon_X"],
        epsilon_ei=config["epsilon_ei"],
        gp=gp,
        f=objective_least_squares_5d,
        max_iters=int(config["max_iters"]),
        max_partitions=int(config["max_partitions"]),
        backend="auto",
        validation=False,
        verbose=bool(config.get("verbose", True)),
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
