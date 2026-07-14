#!/usr/bin/env python3
"""Small file-based workflow for the native ExactBO partitioning executable."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

PARTITION_INPUT_MAGIC = b"TPARIN1!"
PARTITION_OUTPUT_MAGIC = b"TPAROU1!"


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


def make_initial_data():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    X0 = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.5, 0.5],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return X0, bounds


def make_gpr(dim):
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


def _write_array(f, values):
    f.write(np.ascontiguousarray(values, dtype=np.float64).tobytes())


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
    print("launch:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_workflow(
    *,
    workdir,
    native_build_dir,
    partitioning_exe=None,
    mpi_ranks=1,
    max_partitions=6,
    epsilon_x=1e-3,
    epsilon_ei=1e-6,
    device_batch_rows=4096,
    split_batch_parents=0,
    box_storage="auto",
    host_box_limit_bytes=0,
    spill_dir=None,
    keep_spill_files=False,
    verbose=False,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(partitioning_exe) if partitioning_exe else Path(native_build_dir) / "exactbo_partitioning"

    X0, bounds = make_initial_data()
    y0 = minimization_2d_objective(X0)
    gp = make_gpr(X0.shape[1])
    gp.fit(X0, y0)
    params = gp_params(gp, X0)
    epsilon_x_array = np.full((X0.shape[1],), float(epsilon_x), dtype=np.float64)

    np.save(workdir / "X0.npy", X0)
    np.save(workdir / "y0.npy", y0)
    np.save(workdir / "bounds.npy", bounds)
    np.savez(workdir / "gpr_parameters.npz", **params)

    input_path = workdir / "partitioning_input.bin"
    output_path = workdir / "partitioning_output.bin"
    write_partition_input(
        input_path,
        X0=X0,
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
    result = read_partition_output(output_path)
    np.save(workdir / "best_x.npy", result["best_x"])
    (workdir / "manifest.json").write_text(
        json.dumps(
            {
                "executable": str(exe),
                "mpi_ranks": int(mpi_ranks),
                "max_partitions": int(max_partitions),
                "epsilon_x": epsilon_x_array.tolist(),
                "epsilon_ei": float(epsilon_ei),
                "device_batch_rows": int(device_batch_rows),
                "split_batch_parents": int(split_batch_parents),
                "box_storage": box_storage,
                "host_box_limit_bytes": int(host_box_limit_bytes),
                "spill_dir": None if spill_dir is None else str(spill_dir),
                "keep_spill_files": bool(keep_spill_files),
                "best_ei_scaled": result["best_ei_scaled"],
                "partitions_done": result["partitions_done"],
                "n_boxes_final": result["n_boxes_final"],
                "converged": result["converged"],
                "files": {
                    "X0": "X0.npy",
                    "y0": "y0.npy",
                    "bounds": "bounds.npy",
                    "gpr_parameters": "gpr_parameters.npz",
                    "best_x": "best_x.npy",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved workflow files in {workdir}")
    print(f"best_x={result['best_x']}")
    print(f"best_ei_scaled={result['best_ei_scaled']}")
    print(f"partitions_done={result['partitions_done']} converged={result['converged']} n_boxes_final={result['n_boxes_final']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run native ExactBO partitioning on the 2D example.")
    parser.add_argument("--workdir", default="native/exactbo/data/partitioning_workflow")
    parser.add_argument("--native-build-dir", default="native/build")
    parser.add_argument("--partitioning-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--max-partitions", type=int, default=6)
    parser.add_argument("--epsilon-x", type=float, default=1e-3)
    parser.add_argument("--epsilon-ei", type=float, default=1e-6)
    parser.add_argument("--device-batch-rows", type=int, default=4096,
                        help="Maximum boxes in one CUDA allocation/launch batch.")
    parser.add_argument("--split-batch-parents", type=int, default=0,
                        help="Selected parents per CUDA split batch; 0 chooses automatically.")
    parser.add_argument("--box-storage", choices=["auto", "host", "file"], default="auto",
                        help="Where complete box populations are retained between phases.")
    parser.add_argument("--host-box-limit-bytes", type=int, default=0,
                        help="RAM budget for box stores; 0 derives a conservative limit.")
    parser.add_argument("--spill-dir", default=None,
                        help="Shared filesystem directory for temporary box stores.")
    parser.add_argument("--keep-spill-files", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_workflow(
        workdir=args.workdir,
        native_build_dir=args.native_build_dir,
        partitioning_exe=args.partitioning_exe,
        mpi_ranks=args.mpi_ranks,
        max_partitions=args.max_partitions,
        epsilon_x=args.epsilon_x,
        epsilon_ei=args.epsilon_ei,
        device_batch_rows=args.device_batch_rows,
        split_batch_parents=args.split_batch_parents,
        box_storage=args.box_storage,
        host_box_limit_bytes=args.host_box_limit_bytes,
        spill_dir=args.spill_dir,
        keep_spill_files=args.keep_spill_files,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
