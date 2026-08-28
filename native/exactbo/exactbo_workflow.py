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
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime
import importlib
import importlib.util
import json
import shlex
import struct
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

PARTITION_INPUT_MAGIC = b"TPARIN1!"
PARTITION_OUTPUT_MAGIC = b"TPAROU1!"


class TeeStream:
    """Write each message to the terminal and to the run log."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def current_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def load_problem_from_args(args):
    if args.example == "minimization_2d":
        f, X0, bounds = make_minimization_2d_data()
    elif args.example == "problem5d":
        f, X0, bounds = make_problem5d_data()
    elif args.example == "problem10d":
        f, X0, bounds = make_problem10d_data()
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


def make_problem5d_data():
    """Return the 5D nonlinear least-squares benchmark from experiment 2."""
    bounds = np.tile(np.array([[-1.0, 1.0]], dtype=np.float64), (5, 1))
    X0 = np.zeros((1, 5), dtype=np.float64)
    return problem5d_objective, X0, bounds


def problem5d_objective(X):
    """Five-dimensional nonlinear least-squares objective."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x0, x1, x2, x3, x4 = [X[:, i] for i in range(5)]

    t1 = -4.583 - 3.933 * x0 + 0.107 * x1 + 0.126 * x2 - 9.99 * x4
    t2 = 1.4185 - 0.987 * x1 - 22.95 * x3
    t3 = -0.0921 + 0.002 * x0 - 0.235 * x2 + 5.67 * x4
    t4 = 0.0084 + x1 - x3
    t5 = -0.00071 - x2 - 0.196 * x4

    t6 = -0.727 * x1 * x2 + 8.39 * x2 * x3 - 684.4 * x3 * x4 + 63.5 * x3 * x1
    t7 = 0.949 * x0 * x2 + 0.173 * x0 * x4
    t8 = -0.716 * x0 * x1 - 1.578 * x0 * x3 + 1.132 * x3 * x1
    t9 = -x0 * x4
    t10 = x0 * x3

    return (
        (t1 + t6) ** 2
        + (t2 + t7) ** 2
        + (t3 + t8) ** 2
        + (t4 + t9) ** 2
        + (t5 + t10) ** 2
    )


PROBLEM10D_A = np.array(
    [
        [-0.8123, 0.2413, -0.2964, 0.2484, 2.3081, -1.3713, 3.0762, -4.4416, -0.0310, -0.1894],
        [0.6772, 0.0609, -1.2558, -0.0741, 3.4272, 1.2321, -3.4688, -6.2450, -0.6963, -0.7903],
        [-0.3365, 0.5358, 0.8026, 1.9580, 1.4212, 1.9957, -1.1197, 4.1086, 0.5639, -0.7754],
        [-0.4282, 0.4834, -0.4652, 1.4991, -2.8462, 0.6664, -1.0966, -4.4469, -0.7988, 0.5689],
        [0.5119, 0.6563, 0.3090, 1.1751, 1.7230, 1.9228, 1.5512, -1.6881, -0.4119, -0.4169],
        [-1.0177, -0.4945, 0.6911, -2.2498, -3.0143, -2.1242, 0.8211, 0.7823, -0.5253, 0.2071],
        [-0.9385, 0.4491, -1.1320, -2.1417, -2.9591, 1.8822, 2.4259, -4.0952, 0.0617, 0.9288],
        [0.3490, 0.1322, -1.0659, -2.0633, 1.0886, 3.7698, -1.1095, 2.1446, -0.8170, -0.1350],
    ],
    dtype=np.float64,
)
PROBLEM10D_B = np.array(
    [1.7367, 6.9483, -1.1465, 2.6396, -0.5015, -0.2883, 2.1894, 1.8491],
    dtype=np.float64,
)


def make_problem10d_data():
    """Return the 10D L1-regularized least-squares benchmark from experiment 2."""
    bounds = np.tile(np.array([[-1.0, 1.0]], dtype=np.float64), (10, 1))
    X0 = np.zeros((1, 10), dtype=np.float64)
    return problem10d_objective, X0, bounds


def problem10d_objective(X):
    """Ten-dimensional L1-regularized least-squares objective."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    residual = X @ PROBLEM10D_A.T - PROBLEM10D_B
    smooth = 0.5 * np.sum(residual**2, axis=1)
    penalty = np.sum(np.abs(X), axis=1)
    return smooth + penalty


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


def make_gpr(dim: int):
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.full(dim, 0.2), length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)


def gp_params(gp, X_train):
    kernel = gp.kernel_
    return {
        "X_train": np.asarray(X_train, dtype=np.float64),
        "alpha": np.asarray(gp.alpha_, dtype=np.float64).reshape(-1),
        "L": np.asarray(gp.L_, dtype=np.float64),
        "length_scale": np.asarray(kernel.k1.k2.length_scale, dtype=np.float64).reshape(-1),
        "sigma_f_squared": float(kernel.k1.k1.constant_value),
        "sigma_n_squared": float(kernel.k2.noise_level),
        "y_train_mean": float(np.asarray(gp._y_train_mean, dtype=np.float64).reshape(-1)[0]),
        "y_train_std": float(np.asarray(gp._y_train_std, dtype=np.float64).reshape(-1)[0]),
        "y_min_scaled": float(np.min(np.asarray(gp.y_train_, dtype=np.float64))),
    }


def save_gp_parameters(path: Path, params: dict):
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def write_partition_input(path, *, X0, bounds, params, epsilon_x, epsilon_ei, max_partitions):
    X0 = np.asarray(X0, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    epsilon_x = np.asarray(epsilon_x, dtype=np.float64).reshape(-1)
    n_train, d = X0.shape
    if bounds.shape != (d, 2):
        raise ValueError("bounds must have shape (d, 2)")
    if epsilon_x.shape != (d,):
        raise ValueError("epsilon_x must have shape (d,)")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(PARTITION_INPUT_MAGIC)
        f.write(struct.pack("<QQQdddddd", n_train, d, int(max_partitions), float(epsilon_ei),
                            params["sigma_f_squared"], params["sigma_n_squared"],
                            params["y_train_mean"], params["y_train_std"], params["y_min_scaled"]))
        _write_array(f, epsilon_x)
        _write_array(f, bounds[:, 0])
        _write_array(f, bounds[:, 1])
        _write_array(f, params["X_train"])
        _write_array(f, params["alpha"])
        _write_array(f, params["L"])
        _write_array(f, params["length_scale"])


def _write_array(f, values):
    f.write(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def launch_native(
    exe,
    mpi_ranks,
    input_path,
    output_path,
    *,
    device_batch_rows=4096,
    split_batch_parents=0,
    box_storage="auto",
    host_box_limit_bytes=0,
    spill_dir=None,
    keep_spill_files=False,
    verbose=False,
):
    """Launch the native partitioner with explicit memory-policy controls."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mpi_ranks <= 1:
        cmd = [str(exe), "--input", str(input_path), "--output", str(output_path)]
    else:
        cmd = ["mpirun", "-np", str(mpi_ranks), str(exe), "--input", str(input_path), "--output", str(output_path)]
    if device_batch_rows <= 0:
        raise ValueError("device_batch_rows must be positive")
    if split_batch_parents < 0:
        raise ValueError("split_batch_parents must be nonnegative (0 means automatic)")
    if box_storage not in {"auto", "host", "file"}:
        raise ValueError("box_storage must be 'auto', 'host', or 'file'")
    if host_box_limit_bytes < 0:
        raise ValueError("host_box_limit_bytes must be nonnegative")
    cmd.extend(("--device-batch-rows", str(int(device_batch_rows))))
    cmd.extend(("--split-batch-parents", str(int(split_batch_parents))))
    cmd.extend(("--box-storage", box_storage))
    cmd.extend(("--host-box-limit-bytes", str(int(host_box_limit_bytes))))
    if spill_dir is not None:
        cmd.extend(("--spill-dir", str(spill_dir)))
    if keep_spill_files:
        cmd.append("--keep-spill-files")
    if verbose:
        cmd.append("--verbose")
    print("launch:", shlex.join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def read_partition_output(path):
    with path.open("rb") as f:
        if f.read(8) != PARTITION_OUTPUT_MAGIC:
            raise ValueError(f"{path} is not an exactbo_partitioning output file")
        d, partitions_done, n_boxes_final = struct.unpack("<QQQ", f.read(24))
        (converged,) = struct.unpack("<i", f.read(4))
        (best_ei_scaled,) = struct.unpack("<d", f.read(8))
        best_x = np.frombuffer(f.read(8 * d), dtype="<f8").copy()
    return {
        "best_x": best_x,
        "best_ei_scaled": float(best_ei_scaled),
        "partitions_done": int(partitions_done),
        "n_boxes_final": int(n_boxes_final),
        "converged": bool(converged),
    }


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


def run_native_exactbo(
    f: Callable[[np.ndarray], np.ndarray],
    X0,
    bounds,
    *,
    y0=None,
    gp: GaussianProcessRegressor | None = None,
    workdir="native/exactbo/data/exactbo_workflow",
    native_build_dir="native/exactbo/executables",
    partitioning_exe=None,
    mpi_ranks=1,
    max_iters=3,
    max_partitions=6,
    epsilon_x=1e-5,
    epsilon_ei=1e-2,
    device_batch_rows=4096,
    split_batch_parents=0,
    box_storage="auto",
    host_box_limit_bytes=0,
    spill_dir=None,
    keep_spill_files=False,
    log_file=None,
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
        launch_native(
            exe,
            mpi_ranks,
            input_path,
            output_path,
            device_batch_rows=device_batch_rows,
            split_batch_parents=split_batch_parents,
            box_storage=box_storage,
            host_box_limit_bytes=host_box_limit_bytes,
            spill_dir=spill_dir,
            keep_spill_files=keep_spill_files,
            verbose=verbose,
        )
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
        "split_batch_parents": int(split_batch_parents),
        "box_storage": box_storage,
        "host_box_limit_bytes": int(host_box_limit_bytes),
        "spill_dir": None if spill_dir is None else str(spill_dir),
        "keep_spill_files": bool(keep_spill_files),
        "log_file": None if log_file is None else str(log_file),
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


def main():
    parser = argparse.ArgumentParser(description="Run BO with native ExactBO partitioning as acquisition.")
    parser.add_argument("--example", choices=["minimization_2d", "problem5d", "problem10d", "none"], default="minimization_2d")
    parser.add_argument("--objective", default=None, help="Objective as module:function or /path/file.py:function.")
    parser.add_argument("--x0", default=None, help=".npy file with initial X, shape (n0, d).")
    parser.add_argument("--y0", default=None, help="Optional .npy file with initial y, shape (n0,).")
    parser.add_argument("--bounds", default=None, help=".npy file with bounds, shape (d, 2).")
    parser.add_argument("--workdir", default="native/exactbo/data")
    parser.add_argument("--native-build-dir", default="native/exactbo/executables")
    parser.add_argument("--partitioning-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--max-partitions", type=int, default=6)
    parser.add_argument("--epsilon-x", type=float, default=1e-5)
    parser.add_argument("--epsilon-ei", type=float, default=1e-2)
    parser.add_argument("--device-batch-rows", type=int, default=4096, help="Maximum boxes in one CUDA allocation/launch batch.")
    parser.add_argument("--split-batch-parents", type=int, default=0, help="Selected parents per CUDA split batch; 0 chooses automatically.")
    parser.add_argument("--box-storage", choices=["auto", "host", "file"], default="auto")
    parser.add_argument("--host-box-limit-bytes", type=int, default=0)
    parser.add_argument("--spill-dir", default=None, help="Shared filesystem directory for temporary box stores.")
    parser.add_argument("--keep-spill-files", action="store_true")
    parser.add_argument("--log-file", default=None, help="Optional file that receives a copy of terminal/native output.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_path = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else Path(args.workdir).expanduser().resolve() / "exactbo_run.log"
    )

    with ExitStack() as log_context:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_context.enter_context(log_path.open("w", encoding="utf-8", buffering=1))
            tee_out = TeeStream(sys.stdout, log_handle)
            tee_err = TeeStream(sys.stderr, log_handle)
            log_context.enter_context(redirect_stdout(tee_out))
            log_context.enter_context(redirect_stderr(tee_err))

        effective_arguments = vars(args).copy()
        effective_arguments["log_file"] = None if log_path is None else str(log_path)

        print("=" * 80, flush=True)
        print("ExactBO run log", flush=True)
        print(f"started_at={current_timestamp()}", flush=True)
        print(f"launch_directory={Path.cwd()}", flush=True)
        print("command=" + shlex.join([sys.executable, *sys.argv]), flush=True)
        print("arguments=" + json.dumps(effective_arguments, indent=2), flush=True)
        print("=" * 80, flush=True)

        try:
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
                split_batch_parents=args.split_batch_parents,
                box_storage=args.box_storage,
                host_box_limit_bytes=args.host_box_limit_bytes,
                spill_dir=args.spill_dir,
                keep_spill_files=args.keep_spill_files,
                log_file=log_path,
                verbose=args.verbose,
            )
            print(f"final_best_x={result['best_x']}", flush=True)
            print(f"final_best_y={result['best_y']}", flush=True)
        except BaseException:
            print("run_status=failed", flush=True)
            traceback.print_exc()
            print(f"finished_at={current_timestamp()}", flush=True)
            return 1

        print("run_status=success", flush=True)
        print(f"finished_at={current_timestamp()}", flush=True)
        if log_path is not None:
            print(f"log_file={log_path}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
