#!/usr/bin/env python3
"""Small Python orchestrator for native ExactBO bound kernels.

Current workflow:
1. Receive a black-box objective function `f` and initial design `X0`.
2. Evaluate `y0 = f(X0)`.
3. Fit the same sklearn GaussianProcessRegressor style used by
   `examples/exactbo/minimization_2d.py`.
4. Save initial data, domain bounds, box bounds, and trained GP parameters.
5. Launch native CUDA/MPI kernels through files:
   - rbf_k_bounds, once per training point, to build K_lo and K_hi
   - mu_bounds
   - sigma_bounds
   - ei_bounds
6. Read the saved native outputs and end with EI_lo and EI_hi for each box.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

RBF_INPUT_MAGIC = b"TRBFKIN1"
RBF_OUTPUT_MAGIC = b"TRBFKOU1"
MU_INPUT_MAGIC = b"TMUBIN1!"
MU_OUTPUT_MAGIC = b"TMUBOU1!"
SIGMA_INPUT_MAGIC = b"TSIGIN1!"
SIGMA_OUTPUT_MAGIC = b"TSIGOU1!"
EI_INPUT_MAGIC = b"TEIBIN1!"
EI_OUTPUT_MAGIC = b"TEIBOU1!"


def minimization_2d_objective(X):
    """Same objective used in examples/exactbo/minimization_2d.py."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    x = X[:, 0]
    y = X[:, 1]

    alpha = 0.1
    amplitudes = np.array([4.0, 3.0, 2.0], dtype=np.float64)
    widths = np.array([0.08, 0.05, 0.02], dtype=np.float64)
    centers = np.array(
        [
            [0.9, 0.3],
            [0.1, 0.8],
            [0.6, 0.7],
        ],
        dtype=np.float64,
    )

    value = alpha * (x * x + y * y)
    for amplitude, width, center in zip(amplitudes, widths, centers):
        dx = x - center[0]
        dy = y - center[1]
        value -= amplitude * np.exp(-(dx * dx + dy * dy) / width)
    return value + 2.0


def make_minimization_2d_initial_data():
    domain_bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
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
    return X0, domain_bounds


def make_quadrant_boxes():
    bounds_l = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.25],
            [0.0, 0.5],
            [0.0, 0.75],
            [0.25, 0.0],
            [0.25, 0.25],
            [0.25, 0.5],
            [0.25, 0.75],
            [0.5, 0.0],
            [0.5, 0.25],
            [0.5, 0.5],
            [0.5, 0.75],
            [0.75, 0.0],
            [0.75, 0.25],
            [0.75, 0.5],
            [0.75, 0.75],
        ],
        dtype=np.float64,
    )
    bounds_u = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.5],
            [0.25, 0.75],
            [0.25, 1.0],
            [0.5, 0.25],
            [0.5, 0.5],
            [0.5, 0.75],
            [0.5, 1.0],
            [0.75, 0.25],
            [0.75, 0.5],
            [0.75, 0.75],
            [0.75, 1.0],
            [1.0, 0.25],
            [1.0, 0.5],
            [1.0, 0.75],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return bounds_l, bounds_u


def make_gpr(dim: int):
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.full(dim, 0.2), length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)


def export_training_data(workdir: Path, X0, y0, domain_bounds, bounds_l, bounds_u):
    np.save(workdir / "X0.npy", np.asarray(X0, dtype=np.float64))
    np.save(workdir / "y0.npy", np.asarray(y0, dtype=np.float64).reshape(-1))
    np.save(workdir / "domain_bounds.npy", np.asarray(domain_bounds, dtype=np.float64))
    np.save(workdir / "box_bounds_L.npy", np.asarray(bounds_l, dtype=np.float64))
    np.save(workdir / "box_bounds_U.npy", np.asarray(bounds_u, dtype=np.float64))


def export_gpr_parameters(workdir: Path, gp: GaussianProcessRegressor, X_train):
    kernel = gp.kernel_
    params = {
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

    np.savez(
        workdir / "gpr_parameters.npz",
        X_train=params["X_train"],
        alpha=params["alpha"],
        L=params["L"],
        length_scale=params["length_scale"],
        sigma_f_squared=np.array(params["sigma_f_squared"], dtype=np.float64),
        sigma_n_squared=np.array(params["sigma_n_squared"], dtype=np.float64),
        y_train_mean=np.array(params["y_train_mean"], dtype=np.float64),
        y_train_std=np.array(params["y_train_std"], dtype=np.float64),
        y_min_scaled=np.array(params["y_min_scaled"], dtype=np.float64),
    )

    (workdir / "gpr_scalars.json").write_text(
        json.dumps(
            {
                "sigma_f_squared": params["sigma_f_squared"],
                "sigma_n_squared": params["sigma_n_squared"],
                "y_train_mean": params["y_train_mean"],
                "y_train_std": params["y_train_std"],
                "y_min_scaled": params["y_min_scaled"],
            },
            indent=2,
        )
        + "\n"
    )
    return params


def _write_array(f, values):
    f.write(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def write_rbf_input(path: Path, bounds_l, bounds_u, xi, sigma_f_squared: float, length_scale):
    bounds_l = np.asarray(bounds_l, dtype=np.float64)
    bounds_u = np.asarray(bounds_u, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    length_scale = np.asarray(length_scale, dtype=np.float64).reshape(-1)
    n, d = bounds_l.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(RBF_INPUT_MAGIC)
        f.write(struct.pack("<QQd", n, d, float(sigma_f_squared)))
        _write_array(f, bounds_l)
        _write_array(f, bounds_u)
        _write_array(f, xi)
        _write_array(f, length_scale)


def read_rbf_output(path: Path):
    with path.open("rb") as f:
        if f.read(8) != RBF_OUTPUT_MAGIC:
            raise ValueError(f"{path} is not an rbf_k_bounds output file")
        n, _d = struct.unpack("<QQ", f.read(16))
        lo = np.frombuffer(f.read(8 * n), dtype="<f8").copy()
        hi = np.frombuffer(f.read(8 * n), dtype="<f8").copy()
    return lo, hi


def write_mu_input(path: Path, K_lo, K_hi, params, *, scaled_output=True):
    n, N = K_lo.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MU_INPUT_MAGIC)
        f.write(struct.pack("<QQddQ", n, N, params["y_train_mean"], params["y_train_std"], int(scaled_output)))
        _write_array(f, params["alpha"])
        _write_array(f, K_lo)
        _write_array(f, K_hi)


def write_sigma_input(path: Path, K_lo, K_hi, params, *, scaled_output=True):
    n, N = K_lo.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(SIGMA_INPUT_MAGIC)
        f.write(struct.pack("<QQddQ", n, N, params["sigma_f_squared"], params["y_train_std"], int(scaled_output)))
        _write_array(f, params["L"])
        _write_array(f, K_lo)
        _write_array(f, K_hi)


def write_ei_input(path: Path, mu_lo, mu_hi, sig_lo, sig_hi, y_min, *, pad=1e-12):
    n = np.asarray(mu_lo).shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(EI_INPUT_MAGIC)
        f.write(struct.pack("<Qdd", n, float(y_min), float(pad)))
        _write_array(f, mu_lo)
        _write_array(f, mu_hi)
        _write_array(f, sig_lo)
        _write_array(f, sig_hi)


def read_pair_output(path: Path, expected_magic: bytes):
    with path.open("rb") as f:
        if f.read(8) != expected_magic:
            raise ValueError(f"unexpected magic in {path}")
        (n,) = struct.unpack("<Q", f.read(8))
        lo = np.frombuffer(f.read(8 * n), dtype="<f8").copy()
        hi = np.frombuffer(f.read(8 * n), dtype="<f8").copy()
    return lo, hi


def launch_native(exe: Path, mpi_ranks: int, input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mpi_ranks <= 1:
        cmd = [str(exe), "--input", str(input_path), "--output", str(output_path)]
    else:
        cmd = ["mpirun", "-np", str(mpi_ranks), str(exe), "--input", str(input_path), "--output", str(output_path)]
    print("launch:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def resolve_executables(native_build_dir, rbf_exe=None, mu_exe=None, sigma_exe=None, ei_exe=None):
    build_dir = Path(native_build_dir)
    return {
        "rbf": Path(rbf_exe) if rbf_exe else build_dir / "exactbo_rbf_k_bounds",
        "mu": Path(mu_exe) if mu_exe else build_dir / "exactbo_mu_bounds",
        "sigma": Path(sigma_exe) if sigma_exe else build_dir / "exactbo_sigma_bounds",
        "ei": Path(ei_exe) if ei_exe else build_dir / "exactbo_ei_bounds",
    }


def run_workflow(
    f: Callable[[np.ndarray], np.ndarray],
    X0,
    domain_bounds,
    box_bounds_l,
    box_bounds_u,
    *,
    workdir="native/data/exactbo/rbf_k_bounds_workflow",
    native_build_dir="native/build",
    rbf_exe=None,
    mu_exe=None,
    sigma_exe=None,
    ei_exe=None,
    mpi_ranks=1,
):
    """Run Python-orchestrated native ExactBO bound kernels through EI bounds."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = resolve_executables(native_build_dir, rbf_exe, mu_exe, sigma_exe, ei_exe)

    X0 = np.asarray(X0, dtype=np.float64)
    domain_bounds = np.asarray(domain_bounds, dtype=np.float64)
    box_bounds_l = np.asarray(box_bounds_l, dtype=np.float64)
    box_bounds_u = np.asarray(box_bounds_u, dtype=np.float64)

    y0 = np.asarray(f(X0), dtype=np.float64).reshape(-1)
    gp = make_gpr(X0.shape[1])
    gp.fit(X0, y0)

    export_training_data(workdir, X0, y0, domain_bounds, box_bounds_l, box_bounds_u)
    params = export_gpr_parameters(workdir, gp, X0)

    n_boxes = box_bounds_l.shape[0]
    n_train = X0.shape[0]
    K_lo = np.empty((n_boxes, n_train), dtype=np.float64)
    K_hi = np.empty_like(K_lo)

    rbf_input_dir = workdir / "rbf_inputs"
    rbf_output_dir = workdir / "rbf_outputs"
    for i, xi in enumerate(X0):
        input_path = rbf_input_dir / f"train_{i}.bin"
        output_path = rbf_output_dir / f"train_{i}.bin"
        write_rbf_input(input_path, box_bounds_l, box_bounds_u, xi, params["sigma_f_squared"], params["length_scale"])
        launch_native(exe["rbf"], mpi_ranks, input_path, output_path)
        K_lo[:, i], K_hi[:, i] = read_rbf_output(output_path)

    np.save(workdir / "K_lo.npy", K_lo)
    np.save(workdir / "K_hi.npy", K_hi)

    mu_input = workdir / "mu_bounds_input.bin"
    mu_output = workdir / "mu_bounds_output.bin"
    write_mu_input(mu_input, K_lo, K_hi, params, scaled_output=True)
    launch_native(exe["mu"], mpi_ranks, mu_input, mu_output)
    mu_lo, mu_hi = read_pair_output(mu_output, MU_OUTPUT_MAGIC)
    np.save(workdir / "mu_lo.npy", mu_lo)
    np.save(workdir / "mu_hi.npy", mu_hi)

    sigma_input = workdir / "sigma_bounds_input.bin"
    sigma_output = workdir / "sigma_bounds_output.bin"
    write_sigma_input(sigma_input, K_lo, K_hi, params, scaled_output=True)
    launch_native(exe["sigma"], mpi_ranks, sigma_input, sigma_output)
    sig_lo, sig_hi = read_pair_output(sigma_output, SIGMA_OUTPUT_MAGIC)
    np.save(workdir / "sig_lo.npy", sig_lo)
    np.save(workdir / "sig_hi.npy", sig_hi)

    ei_input = workdir / "ei_bounds_input.bin"
    ei_output = workdir / "ei_bounds_output.bin"
    write_ei_input(ei_input, mu_lo, mu_hi, sig_lo, sig_hi, params["y_min_scaled"])
    launch_native(exe["ei"], mpi_ranks, ei_input, ei_output)
    ei_lo, ei_hi = read_pair_output(ei_output, EI_OUTPUT_MAGIC)
    np.save(workdir / "ei_lo.npy", ei_lo)
    np.save(workdir / "ei_hi.npy", ei_hi)

    manifest = {
        "workdir": str(workdir),
        "executables": {name: str(path) for name, path in exe.items()},
        "mpi_ranks": int(mpi_ranks),
        "n_initial": int(n_train),
        "n_boxes": int(n_boxes),
        "dim": int(X0.shape[1]),
        "files": {
            "X0": "X0.npy",
            "y0": "y0.npy",
            "domain_bounds": "domain_bounds.npy",
            "box_bounds_L": "box_bounds_L.npy",
            "box_bounds_U": "box_bounds_U.npy",
            "gpr_parameters": "gpr_parameters.npz",
            "K_lo": "K_lo.npy",
            "K_hi": "K_hi.npy",
            "mu_lo": "mu_lo.npy",
            "mu_hi": "mu_hi.npy",
            "sig_lo": "sig_lo.npy",
            "sig_hi": "sig_hi.npy",
            "ei_lo": "ei_lo.npy",
            "ei_hi": "ei_hi.npy",
        },
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"saved workflow files in {workdir}")
    print(f"K_lo shape={K_lo.shape}")
    print(f"K_hi shape={K_hi.shape}")
    print(f"mu_lo shape={mu_lo.shape}")
    print(f"sig_lo shape={sig_lo.shape}")
    print(f"ei_lo shape={ei_lo.shape}")
    print(f"ei_hi shape={ei_hi.shape}")
    return ei_lo, ei_hi


def run_minimization_2d_example(args):
    X0, domain_bounds = make_minimization_2d_initial_data()
    bounds_l, bounds_u = make_quadrant_boxes()
    return run_workflow(
        minimization_2d_objective,
        X0,
        domain_bounds,
        bounds_l,
        bounds_u,
        workdir=args.workdir,
        native_build_dir=args.native_build_dir,
        rbf_exe=args.rbf_exe,
        mu_exe=args.mu_exe,
        sigma_exe=args.sigma_exe,
        ei_exe=args.ei_exe,
        mpi_ranks=args.mpi_ranks,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Python-orchestrated native ExactBO bounds through EI.")
    parser.add_argument("--example", choices=["minimization_2d"], default="minimization_2d")
    parser.add_argument("--workdir", default="native/data/exactbo/bounds_workflow")
    parser.add_argument("--native-build-dir", default="native/build")
    parser.add_argument("--rbf-exe", default=None)
    parser.add_argument("--mu-exe", default=None)
    parser.add_argument("--sigma-exe", default=None)
    parser.add_argument("--ei-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    args = parser.parse_args()

    if args.example == "minimization_2d":
        run_minimization_2d_example(args)


if __name__ == "__main__":
    main()
