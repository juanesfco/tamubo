"""Benchmark driver for comparing ExactBO vs baseline BO workflows."""

from __future__ import annotations

import argparse
import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

# Allow running this script directly from the repository without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.bo import run_botorch_grid_ei, run_botorch_optimize_ei, run_sklearn_grid_ei
from tamubo.exactbo import exactbo

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return _to_serializable(value.__dict__)
    return value


def _save_json(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)
    return output_path


def _objective_minimization_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_eval = X.reshape(1, -1) if X.ndim == 1 else X
    x, y = X_eval[:, 0], X_eval[:, 1]

    alpha = 0.1
    A = np.array([4.0, 3.0, 2.0], dtype=float)
    B = np.array([0.08, 0.05, 0.02], dtype=float)
    C = np.array([[0.9, 0.3], [0.1, 0.8], [0.6, 0.7]], dtype=float)
    D = 2.0

    val = alpha * (x**2 + y**2)
    for Ai, Bi, (xi, yi) in zip(A, B, C):
        r2 = (x - xi) ** 2 + (y - yi) ** 2
        val -= Ai * np.exp(-r2 / Bi)
    val += D
    return val


def _build_exactbo_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _summarize_result(result: Any, wall_time_sec: float) -> dict:
    X = np.asarray(result.X, dtype=float)
    y = np.asarray(result.y, dtype=float).ravel()
    idx_best = int(np.argmin(y))
    return {
        "status": "ok",
        "wall_time_sec": float(wall_time_sec),
        "n_evals": int(X.shape[0]),
        "best_idx": idx_best,
        "best_x": X[idx_best].tolist(),
        "best_y": float(y[idx_best]),
        "backend": _to_serializable(result.backend),
    }


def _execute_workflow(
    name: str,
    runner: Callable[[], Any],
    *,
    log_each_workflow: bool,
    logs_dir: Path,
) -> tuple[dict, dict | None]:
    start = time.perf_counter()
    try:
        result = runner()
        elapsed = time.perf_counter() - start
        summary = _summarize_result(result, elapsed)
        if log_each_workflow:
            log_payload = {
                "workflow": name,
                "summary": summary,
                "X": np.asarray(result.X, dtype=float),
                "y": np.asarray(result.y, dtype=float),
                "log": result.log,
            }
            log_path = _save_json(log_payload, logs_dir / f"{name}_log.json")
            summary["log_path"] = str(log_path.resolve())
        return summary, None
    except Exception as exc:  # pragma: no cover - exercised in runtime environments
        elapsed = time.perf_counter() - start
        return (
            {
                "status": "error",
                "wall_time_sec": float(elapsed),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            {
                "workflow": name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config in {path} must be a JSON object.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ExactBO and BO baseline workflows.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    seed = int(config.get("random_seed", 0))
    _set_seed(seed)

    bounds = np.asarray(config["bounds"], dtype=float)
    X0 = np.asarray(config["X0"], dtype=float)
    max_iters = int(config["max_iters"])
    verbose = bool(config.get("verbose", False))
    log_each_workflow = bool(config.get("log_each_workflow", False))
    results_dir = (SCRIPT_DIR / config.get("results_dir", "results")).resolve()
    logs_dir = (SCRIPT_DIR / config.get("logs_dir", "logs")).resolve()
    results_filename = str(config.get("results_filename", "workflow_benchmark_results.json"))
    errors_filename = str(config.get("errors_filename", "workflow_benchmark_errors.json"))

    exactbo_cfg = deepcopy(config.get("exactbo", {}))
    sklearn_cfg = deepcopy(config.get("sklearn_grid", {}))
    botorch_grid_cfg = deepcopy(config.get("botorch_grid", {}))
    botorch_opt_cfg = deepcopy(config.get("botorch_optimize", {}))

    workflows: dict[str, Callable[[], Any]] = {
        "exactbo": lambda: exactbo(
            X0=X0.copy(),
            bounds=bounds.copy(),
            epsilon_X=exactbo_cfg["epsilon_X"],
            epsilon_ei=float(exactbo_cfg["epsilon_ei"]),
            gp=_build_exactbo_gp(),
            f=_objective_minimization_2d,
            max_iters=max_iters,
            max_partitions=int(exactbo_cfg["max_partitions"]),
            backend=str(exactbo_cfg.get("backend", "auto")),
            validation=bool(exactbo_cfg.get("validation", True)),
            verbose=verbose,
            logMask=log_each_workflow,
        ),
        "sklearn_grid_ei": lambda: run_sklearn_grid_ei(
            X0=X0.copy(),
            bounds=bounds.copy(),
            f=_objective_minimization_2d,
            max_iters=max_iters,
            grid_resolution=int(sklearn_cfg.get("grid_resolution", 50)),
            validation=bool(sklearn_cfg.get("validation", True)),
            verbose=verbose,
            logMask=log_each_workflow,
        ),
        "botorch_grid_ei": lambda: run_botorch_grid_ei(
            X0=X0.copy(),
            bounds=bounds.copy(),
            f=_objective_minimization_2d,
            max_iters=max_iters,
            grid_resolution=int(botorch_grid_cfg.get("grid_resolution", 50)),
            validation=bool(botorch_grid_cfg.get("validation", True)),
            verbose=verbose,
            logMask=log_each_workflow,
            device=str(botorch_grid_cfg.get("device", "cpu")),
        ),
        "botorch_optimize_ei": lambda: run_botorch_optimize_ei(
            X0=X0.copy(),
            bounds=bounds.copy(),
            f=_objective_minimization_2d,
            max_iters=max_iters,
            num_restarts=int(botorch_opt_cfg.get("num_restarts", 10)),
            raw_samples=int(botorch_opt_cfg.get("raw_samples", 128)),
            maxiter=int(botorch_opt_cfg.get("maxiter", 200)),
            validation=bool(botorch_opt_cfg.get("validation", True)),
            verbose=verbose,
            logMask=log_each_workflow,
            device=str(botorch_opt_cfg.get("device", "cpu")),
        ),
    }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "objective": "minimization_2d",
        "seed": seed,
        "max_iters": max_iters,
        "bounds": bounds,
        "X0": X0,
        "workflow_order": list(workflows.keys()),
        "workflow_settings": {
            "exactbo": exactbo_cfg,
            "sklearn_grid": sklearn_cfg,
            "botorch_grid": botorch_grid_cfg,
            "botorch_optimize": botorch_opt_cfg,
        },
        "workflows": {},
    }

    errors = []
    for name, runner in workflows.items():
        summary, error_entry = _execute_workflow(
            name, runner, log_each_workflow=log_each_workflow, logs_dir=logs_dir
        )
        report["workflows"][name] = summary
        if error_entry is not None:
            errors.append(error_entry)

    results_path = _save_json(report, results_dir / results_filename)

    print(f"results_saved={results_path}")
    for name in workflows:
        result_summary = report["workflows"][name]
        if result_summary["status"] == "ok":
            print(
                f"[{name}] status=ok best_y={result_summary['best_y']:.6f} "
                f"n_evals={result_summary['n_evals']} wall_time_sec={result_summary['wall_time_sec']:.3f}"
            )
        else:
            print(
                f"[{name}] status=error type={result_summary['error_type']} "
                f"msg={result_summary['error_message']}"
            )

    if errors:
        errors_payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "errors": errors,
        }
        errors_path = _save_json(errors_payload, results_dir / errors_filename)
        print(f"errors_saved={errors_path}")


if __name__ == "__main__":
    main()
