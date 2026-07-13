#!/usr/bin/env python3
"""Python BO orchestrator using native ExactBO partitioning as acquisition.

Python responsibilities:
- own the black-box objective function
- own/append the observed data
- fit sklearn GaussianProcessRegressor each BO iteration
- export fitted GP state to a binary file
- launch the native C++/MPI/CUDA exactbo_partitioning executable
- evaluate the returned candidate and continue
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from partitioning_workflow import (
    gp_params,
    launch_native,
    read_partition_output,
    write_partition_input,
)


def minimization_2d_objective(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x = X[:, 0]
    y = X[:, 1]
    alpha = 0.1
    amplitudes = np.array([4.0, 3.0, 2.0], dtype=np.float64)
    widths = np.array([0.08, 0.05, 0.02], dtype=np.float64)
    centers = np.array([[0.9, 0.3], [0.1, 0.8], [0.6, 0.7]], dtype=np.float64)
    value = alpha * (x * x + y * y)
    for amplitude, width, center in zip(amplitudes, widths, centers):
        dx = x - center[0]
        dy = y - center[1]
        value -= amplitude * np.exp(-(dx * dx + dy * dy) / width)
    return value + 2.0


def make_minimization_2d_data():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    X0 = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ],
        dtype=np.float64,
    )
    return minimization_2d_objective, X0, bounds


def make_gpr(dim: int):
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.full(dim, 0.2), length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)


def normalize_epsilon(epsilon, dim: int):
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=np.float64)
    eps = eps.reshape(-1)
    if eps.shape != (dim,):
        raise ValueError(f"epsilon_x must be scalar or shape ({dim},), got {eps.shape}")
    return eps


def evaluate_objective(f: Callable[[np.ndarray], np.ndarray], X, dim: int):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, dim)
    if X.ndim != 2 or X.shape[1] != dim:
        raise ValueError(f"objective input must have shape (n, {dim}), got {X.shape}")
    y = np.asarray(f(X), dtype=np.float64).reshape(-1)
    if y.shape != (X.shape[0],):
        raise ValueError(f"objective must return shape ({X.shape[0]},), got {y.shape}")
    return y


def load_callable(spec: str):
    """Load an objective from 'module:function' or '/path/file.py:function'."""
    if ":" not in spec:
        raise ValueError("objective spec must be 'module:function' or '/path/file.py:function'")
    module_spec, function_name = spec.split(":", 1)
    if module_spec.endswith(".py") or "/" in module_spec:
        module_path = Path(module_spec).resolve()
        module_name = module_path.stem
        spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"cannot import module from {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    f = getattr(module, function_name)
    if not callable(f):
        raise TypeError(f"{spec} is not callable")
    return f


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_gp_parameters(path: Path, params: dict):
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def run_native_exactbo(
    f: Callable[[np.ndarray], np.ndarray],
    X0,
    bounds,
    *,
    y0=None,
    gp: GaussianProcessRegressor | None = None,
    workdir="native/exactbo/data/exactbo_bo_workflow",
    native_build_dir="native/build",
    partitioning_exe=None,
    mpi_ranks=1,
    max_iters=3,
    max_partitions=6,
    epsilon_x=1e-3,
    epsilon_ei=1e-6,
    device_batch_rows=4096,
    verbose=False,
):
    """Run BO iterations with native ExactBO partitioning as acquisition.

    Parameters
    ----------
    f:
        Objective callable. It receives an array of shape ``(n, d)`` and returns
        a flat array of shape ``(n,)``.
    X0:
        Initial design points, shape ``(n0, d)``.
    bounds:
        Search bounds, shape ``(d, 2)``.
    y0:
        Optional already-evaluated objective values for ``X0``.
    gp:
        Optional sklearn GaussianProcessRegressor. If omitted, a default
        Constant*RBF + WhiteKernel model is created.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    native_build_dir = Path(native_build_dir)
    exe = Path(partitioning_exe) if partitioning_exe else native_build_dir / "exactbo_partitioning"

    X = np.asarray(X0, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X0 must be 2D, got shape {X.shape}")
    dim = X.shape[1]
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (dim, 2):
        raise ValueError(f"bounds must have shape ({dim}, 2), got {bounds.shape}")
    epsilon_x_array = normalize_epsilon(epsilon_x, dim)

    if y0 is None:
        y = evaluate_objective(f, X, dim)
    else:
        y = np.asarray(y0, dtype=np.float64).reshape(-1)
        if y.shape != (X.shape[0],):
            raise ValueError(f"y0 must have shape ({X.shape[0]},), got {y.shape}")

    if gp is None:
        gp = make_gpr(dim)

    history = []
    np.save(workdir / "bounds.npy", bounds)
    np.save(workdir / "epsilon_x.npy", epsilon_x_array)

    for iteration in range(int(max_iters)):
        iteration_dir = workdir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        gp.fit(X, y)
        params = gp_params(gp, X)

        np.save(iteration_dir / "X_train.npy", X)
        np.save(iteration_dir / "y_train.npy", y)
        save_gp_parameters(iteration_dir / "gpr_parameters.npz", params)

        input_path = iteration_dir / "partitioning_input.bin"
        output_path = iteration_dir / "partitioning_output.bin"
        write_partition_input(
            input_path,
            X0=X,
            bounds=bounds,
            params=params,
            epsilon_x=epsilon_x_array,
            epsilon_ei=epsilon_ei,
            max_partitions=max_partitions,
        )
        launch_native(exe, mpi_ranks, input_path, output_path, device_batch_rows=device_batch_rows, verbose=verbose)
        partition_result = read_partition_output(output_path)
        x_next = np.asarray(partition_result["best_x"], dtype=np.float64).reshape(dim)
        y_next = evaluate_objective(f, x_next, dim)[0]

        X = np.vstack((X, x_next.reshape(1, dim)))
        y = np.hstack((y, np.array([y_next], dtype=np.float64)))
        best_idx = int(np.argmin(y))

        np.save(iteration_dir / "x_next.npy", x_next)
        np.save(iteration_dir / "y_next.npy", np.array(y_next, dtype=np.float64))
        np.save(iteration_dir / "X_after.npy", X)
        np.save(iteration_dir / "y_after.npy", y)

        record = {
            "iteration": iteration,
            "x_next": x_next,
            "y_next": float(y_next),
            "best_x": X[best_idx],
            "best_y": float(y[best_idx]),
            "partitioning": partition_result,
        }
        history.append(_jsonable(record))
        (iteration_dir / "summary.json").write_text(json.dumps(_jsonable(record), indent=2) + "\n")

        print(
            f"iteration={iteration} x_next={x_next} y_next={y_next:.12g} "
            f"best_y={y[best_idx]:.12g} best_x={X[best_idx]}",
            flush=True,
        )

    best_idx = int(np.argmin(y))
    final = {
        "n_initial": int(np.asarray(X0).shape[0]),
        "max_iters": int(max_iters),
        "max_partitions": int(max_partitions),
        "epsilon_x": epsilon_x_array,
        "epsilon_ei": float(epsilon_ei),
        "device_batch_rows": int(device_batch_rows),
        "mpi_ranks": int(mpi_ranks),
        "executable": str(exe),
        "best_x": X[best_idx],
        "best_y": float(y[best_idx]),
        "history": history,
        "files": {
            "X": "X.npy",
            "y": "y.npy",
            "history": "history.json",
        },
    }
    np.save(workdir / "X.npy", X)
    np.save(workdir / "y.npy", y)
    (workdir / "history.json").write_text(json.dumps(_jsonable(history), indent=2) + "\n")
    (workdir / "manifest.json").write_text(json.dumps(_jsonable(final), indent=2) + "\n")

    return {
        "X": X,
        "y": y,
        "best_x": X[best_idx].copy(),
        "best_y": float(y[best_idx]),
        "history": history,
    }


def load_problem_from_args(args):
    if args.example == "minimization_2d":
        f, X0, bounds = make_minimization_2d_data()
    else:
        if args.objective is None:
            raise ValueError("--objective is required when --example is not used")
        if args.x0 is None or args.bounds is None:
            raise ValueError("--x0 and --bounds are required when --example is not used")
        f = load_callable(args.objective)
        X0 = np.load(args.x0)
        bounds = np.load(args.bounds)

    y0 = np.load(args.y0) if args.y0 else None
    return f, X0, bounds, y0


def main():
    parser = argparse.ArgumentParser(description="Run BO with native ExactBO partitioning as acquisition.")
    parser.add_argument("--example", choices=["minimization_2d", "none"], default="minimization_2d")
    parser.add_argument("--objective", default=None, help="Objective as module:function or /path/file.py:function.")
    parser.add_argument("--x0", default=None, help=".npy file with initial X, shape (n0, d).")
    parser.add_argument("--y0", default=None, help="Optional .npy file with initial y, shape (n0,).")
    parser.add_argument("--bounds", default=None, help=".npy file with bounds, shape (d, 2).")
    parser.add_argument("--workdir", default="native/exactbo/data/exactbo_bo_workflow")
    parser.add_argument("--native-build-dir", default="native/build")
    parser.add_argument("--partitioning-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--max-partitions", type=int, default=6)
    parser.add_argument("--epsilon-x", type=float, default=1e-3)
    parser.add_argument("--epsilon-ei", type=float, default=1e-6)
    parser.add_argument("--device-batch-rows", type=int, default=4096,
                        help="Maximum boxes in one CUDA allocation/launch batch.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    f, X0, bounds, y0 = load_problem_from_args(args)
    result = run_native_exactbo(
        f,
        X0,
        bounds,
        y0=y0,
        workdir=args.workdir,
        native_build_dir=args.native_build_dir,
        partitioning_exe=args.partitioning_exe,
        mpi_ranks=args.mpi_ranks,
        max_iters=args.max_iters,
        max_partitions=args.max_partitions,
        epsilon_x=args.epsilon_x,
        epsilon_ei=args.epsilon_ei,
        device_batch_rows=args.device_batch_rows,
        verbose=args.verbose,
    )
    print(f"final_best_x={result['best_x']}")
    print(f"final_best_y={result['best_y']}")


if __name__ == "__main__":
    main()
