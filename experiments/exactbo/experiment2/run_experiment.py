"""Run one BO framework and write per-iteration rows to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Allow running this script directly from the repository without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

BOTORCH_IMPORT_ERROR: ImportError | None = None
try:
    from tamubo.bo import run_botorch_grid_ei, run_botorch_optimize_ei
except ImportError as exc:
    BOTORCH_IMPORT_ERROR = exc
    run_botorch_grid_ei = None
    run_botorch_optimize_ei = None

EXACTBO_IMPORT_ERROR: ModuleNotFoundError | ImportError | None = None
try:
    from tamubo.exactbo import exactbo
except (ModuleNotFoundError, ImportError) as exc:
    EXACTBO_IMPORT_ERROR = exc
    exactbo = None

from problems import list_problem_names, load_problem

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "experiment_config.json"

CSV_COLUMNS = [
    "d",
    "Method",
    "random_seed",
    "epsilonX",
    "epsilonEI",
    "i",
    "EI",
    "y",
    "y*",
    "R",
    "t (s)",
    "m (MB)",
]

FRAMEWORK_ALIASES = {
    "exactbo": "exactbo",
    "exactBO": "exactbo",
    "gridbo": "botorch_grid",
    "gridBO": "botorch_grid",
    "botorch_grid": "botorch_grid",
    "gradbo": "botorch_optimize",
    "gradBO": "botorch_optimize",
    "botorch_optimize": "botorch_optimize",
}

METHOD_LABELS = {
    "exactbo": "exactBO",
    "botorch_grid": "gridBO",
    "botorch_optimize": "gradBO",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config in {path} must be a JSON object.")
    return config


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


def _build_default_gp(d) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.full(d, 0.2), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)


def _to_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Cannot convert empty value to scalar.")
    return float(arr[0])


def _iteration_entries(log: dict[str, Any] | None) -> list[dict[str, Any]]:
    if log is None:
        return []
    indexed_entries: list[tuple[int, dict[str, Any]]] = []
    for key, value in log.items():
        if key.startswith("i") and key[1:].isdigit() and isinstance(value, dict):
            indexed_entries.append((int(key[1:]), value))
    indexed_entries.sort(key=lambda item: item[0])
    return [entry for _, entry in indexed_entries]


def _normalize_framework(raw_name: str) -> str:
    framework = FRAMEWORK_ALIASES.get(raw_name, raw_name)
    if framework not in METHOD_LABELS:
        choices = ", ".join(sorted(METHOD_LABELS.keys()))
        raise ValueError(
            f"Unknown framework '{raw_name}'. Use one of: {choices} "
            "(or aliases exactBO/gridBO/gradBO)."
        )
    return framework


def _ensure_problem_dim(bounds: np.ndarray, X0: np.ndarray, problem_dim: int) -> None:
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")
    if X0.ndim != 2:
        raise ValueError(f"X0 must have shape (N0, d), got {X0.shape}")
    if bounds.shape[0] != problem_dim:
        raise ValueError(
            f"bounds dimension ({bounds.shape[0]}) must match problem d ({problem_dim})."
        )
    if X0.shape[1] != problem_dim:
        raise ValueError(
            f"X0 dimension ({X0.shape[1]}) must match problem d ({problem_dim})."
        )


def _latin_hypercube_unit(n_points: int, dim: int) -> np.ndarray:
    """Generate Latin Hypercube points in [0, 1]^dim."""
    if n_points <= 0:
        raise ValueError(f"X0 must be > 0 when given as an integer, got {n_points}.")

    lhs = np.empty((n_points, dim), dtype=float)
    for j in range(dim):
        perm = np.random.permutation(n_points)
        lhs[:, j] = (perm + np.random.rand(n_points)) / float(n_points)
    return lhs


def _scale_unit_design_to_bounds(X_unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + np.asarray(X_unit, dtype=float) * (upper - lower)


def _resolve_initial_design(X0_raw: Any, *, default_X0: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Return initial design from config: explicit array or integer LHS size."""
    dim = int(np.asarray(bounds, dtype=float).shape[0])
    if X0_raw is None:
        return np.asarray(default_X0, dtype=float)

    if isinstance(X0_raw, (int, np.integer)) and not isinstance(X0_raw, bool):
        return _scale_unit_design_to_bounds(_latin_hypercube_unit(int(X0_raw), dim), bounds)

    X0 = np.asarray(X0_raw, dtype=float)
    if X0.ndim == 0:
        scalar = float(X0.item())
        if float(scalar).is_integer():
            return _scale_unit_design_to_bounds(_latin_hypercube_unit(int(scalar), dim), bounds)
        raise ValueError("X0 scalar must be an integer count or an array of points.")
    return X0


def _epsilon_x_to_resolution(framework: str, epsilon_X) -> int:
    if framework == "botorch_grid":
        return int(1.0 / epsilon_X + 1.0)

    return float("nan")


def _epsilon_ei_for_framework(framework: str, config: dict[str, Any]) -> float:
    if framework == "exactbo":
        exact_cfg = config.get("exactbo", {})
        return float(exact_cfg.get("epsilon_ei", float("nan")))
    return float("nan")


def _run_framework(
    framework: str,
    *,
    X0: np.ndarray,
    bounds: np.ndarray,
    objective: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    epsilon_X: float,
    normalize_to_unit_cube: bool,
    verbose: bool,
    config: dict[str, Any],
):
    exactbo_cfg = deepcopy(config.get("exactbo", {}))
    botorch_grid_cfg = deepcopy(config.get("botorch_grid", {}))
    botorch_opt_cfg = deepcopy(config.get("botorch_optimize", {}))

    if framework == "exactbo":
        if exactbo is None:
            raise ImportError(
                "Framework 'exactbo' is unavailable because tamubo.exactbo "
                "could not be imported."
            ) from EXACTBO_IMPORT_ERROR
        return exactbo(
            X0=X0.copy(),
            bounds=bounds.copy(),
            epsilon_X=epsilon_X,
            epsilon_ei=float(exactbo_cfg["epsilon_ei"]),
            gp=_build_default_gp(X0.shape[1]),
            f=objective,
            max_iters=max_iters,
            max_partitions=int(exactbo_cfg["max_partitions"]),
            backend=str(exactbo_cfg.get("backend", "auto")),
            predict_batch_size=exactbo_cfg.get("predict_batch_size"),
            bounds_batch_size=exactbo_cfg.get("bounds_batch_size"),
            max_target_boxes=exactbo_cfg.get("max_target_boxes"),
            validation=bool(exactbo_cfg.get("validation", True)),
            verbose=verbose,
            logMask=True,
            normalize_to_unit_cube=normalize_to_unit_cube,
        )

    if framework == "botorch_grid":
        if run_botorch_grid_ei is None:
            raise ImportError(
                "Framework 'botorch_grid' is unavailable because BoTorch "
                "dependencies could not be imported. Install torch, botorch, "
                "and gpytorch, or switch framework to 'exactbo'."
            ) from BOTORCH_IMPORT_ERROR
        return run_botorch_grid_ei(
            X0=X0.copy(),
            bounds=bounds.copy(),
            f=objective,
            max_iters=max_iters,
            gp_sk=_build_default_gp(X0.shape[1]),
            grid_resolution=int(_epsilon_x_to_resolution(framework, epsilon_X)),
            validation=bool(botorch_grid_cfg.get("validation", True)),
            verbose=verbose,
            logMask=True,
            device=str(botorch_grid_cfg.get("device", "cuda")),
            normalize_to_unit_cube=normalize_to_unit_cube,
        )

    if framework == "botorch_optimize":
        if run_botorch_optimize_ei is None:
            raise ImportError(
                "Framework 'botorch_optimize' is unavailable because BoTorch "
                "dependencies could not be imported. Install torch, botorch, "
                "and gpytorch, or switch framework to 'exactbo'."
            ) from BOTORCH_IMPORT_ERROR
        return run_botorch_optimize_ei(
            X0=X0.copy(),
            bounds=bounds.copy(),
            f=objective,
            max_iters=max_iters,
            gp_sk=_build_default_gp(X0.shape[1]),
            num_restarts=int(botorch_opt_cfg.get("num_restarts", 10)),
            raw_samples=int(botorch_opt_cfg.get("raw_samples", 128)),
            maxiter=int(botorch_opt_cfg.get("maxiter", 200)),
            validation=bool(botorch_opt_cfg.get("validation", True)),
            verbose=verbose,
            logMask=True,
            device=str(botorch_opt_cfg.get("device", "cuda")),
            normalize_to_unit_cube=normalize_to_unit_cube,
        )

    raise ValueError(f"Unsupported framework '{framework}'.")


def _build_rows(
    framework: str,
    *,
    problem_d: int,
    y_star: float,
    epsilon_X: float,
    log: dict[str, Any] | None,
    config: dict[str, Any],
    random_seed: int,
) -> list[dict[str, Any]]:
    entries = _iteration_entries(log)
    method = METHOD_LABELS[framework]
    epsilon_ei = _epsilon_ei_for_framework(framework, config)

    rows: list[dict[str, Any]] = []
    cum_error = 0.0
    for idx, entry in enumerate(entries, start=1):
        y_value = _to_scalar(entry["yn"])
        ei_value = _to_scalar(entry["ei_max"])
        time = _to_scalar(entry["time"])

        error = abs(y_value - y_star)
        min_error = error if idx == 1 else min(min_error, error)
        cum_error += error
        regret = cum_error - idx*min_error

        rows.append(
            {
                "d": int(problem_d),
                "Method": method,
                "random_seed": random_seed,
                "epsilonX": epsilon_X,
                "epsilonEI": epsilon_ei,
                "i": idx,
                "EI": ei_value,
                "y": y_value,
                "y*": float(y_star),
                "R": regret,
                "t (s)": time,
                "m (MB)": float("nan"),
            }
        )
    return rows


def _write_rows_to_csv(rows: list[dict[str, Any]], output_path: Path, append: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists() and output_path.stat().st_size > 0:
        mode = "a"
        write_header = False
    else:
        mode = "w"
        write_header = True

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one framework for experiment 2 and write iteration rows to CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    seed = int(config.get("random_seed", 0))
    _set_seed(seed)

    framework_raw = str(config.get("framework", "exactbo"))
    framework = _normalize_framework(framework_raw)

    problem_name = str(config.get("problem", "problem2d"))
    problem = load_problem(problem_name)

    bounds = np.asarray(config.get("bounds", problem.bounds), dtype=float)
    X0 = _resolve_initial_design(
        config.get("X0", None),
        default_X0=problem.X0,
        bounds=bounds,
    )
    _ensure_problem_dim(bounds, X0, problem.d)
    normalize_to_unit_cube = bool(config.get("normalize_to_unit_cube", True))

    max_iters = int(config.get("max_iters", 0))
    if max_iters < 0:
        raise ValueError(f"max_iters must be >= 0, got {max_iters}")
    
    if framework == "botorch_optimize":
        epsilon_X = 1e-5
    else:
        epsilon_X = float(config.get("epsilon_X", 0.01))


    verbose = bool(config.get("verbose", False))
    append_results = bool(config.get("append_results", True))
    results_dir = (SCRIPT_DIR / config.get("results_dir", "results")).resolve()
    results_filename = str(config.get("results_filename", "experiment_results.csv"))
    results_path = results_dir / results_filename

    start = time.perf_counter()
    result = _run_framework(
        framework,
        X0=X0,
        bounds=bounds,
        objective=problem.objective,
        max_iters=max_iters,
        epsilon_X=epsilon_X,
        normalize_to_unit_cube=normalize_to_unit_cube,
        verbose=verbose,
        config=config,
    )
    wall_time_sec = time.perf_counter() - start

    rows = _build_rows(
        framework,
        problem_d=problem.d,
        y_star=problem.y_star,
        epsilon_X=epsilon_X,
        log=result.log,
        config=config,
        random_seed=seed,
    )
    _write_rows_to_csv(rows, results_path, append=append_results)

    print(f"framework={framework}")
    print(f"problem={problem_name}")
    print(f"iterations={len(rows)}")
    print(f"wall_time_sec={wall_time_sec:.6f}")
    print(f"results_saved={results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        available = ", ".join(list_problem_names())
        print(f"error={type(exc).__name__}: {exc}")
        print(f"available_problems={available}")
        raise
