"""Multi-GPU scaling of tamubo.exactbo's CuPy backend on one node.

Runs the same ExactBO search with n_gpus = 1, 2, 4, ... and reports the
partitioning time per iteration, the speedup over 1 GPU, and whether each run
selects the same points as the 1-GPU run. Objective: Levy on [-10, 10]^d.

Usage (on a GPU node; see bench_multigpu.sbatch):
    python bench_multigpu.py --dims 2 5 10 --preset moderate --gpus 1 2 4 8 --iters 3 --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cupy as cp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tamubo.exactbo import exactbo

PRESETS = {
    # Tolerances comparable to experiment 2's published runs: seconds per iteration.
    "moderate": dict(n0=16, eps_X=0.1, eps_ei=0.1, max_partitions=40, batch=1e7, max_target=5e4),
    # Large search (eps_X=1e-5, up to 1e7 target boxes): minutes per iteration on one GPU.
    "stress": dict(n0=32, eps_X=1e-5, eps_ei=0.1, max_partitions=100, batch=2e7, max_target=1e7),
}


def levy(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    w = 1.0 + (X - 1.0) / 4.0
    wi, wd = w[:, :-1], w[:, -1]
    return (
        np.sin(np.pi * w[:, 0]) ** 2
        + np.sum((wi - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * wi + 1.0) ** 2), axis=1)
        + (wd - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * wd) ** 2)
    )


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    strata = (np.arange(n)[:, None] + rng.random((n, d))) / n
    return np.take_along_axis(strata, rng.random((n, d)).argsort(axis=0), axis=0)


def make_gp(d: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.full(d, 0.2), (1e-2, 1e2))
        + WhiteKernel(1e-3, (1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)


def _memory_pools():
    pools = []
    for device in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device):
            pools.append(cp.get_default_memory_pool())
    return pools


def run(d: int, preset: dict, n_gpus: int, iters: int, box_sampling: str = "lhs"):
    bounds = np.tile([-10.0, 10.0], (d, 1))
    X0 = -10.0 + 20.0 * latin_hypercube(preset["n0"], d, np.random.default_rng(0))
    pools = _memory_pools()
    for pool in pools:
        pool.free_all_blocks()
    start = time.perf_counter()
    result = exactbo(
        X0=X0, bounds=bounds, epsilon_X=preset["eps_X"], epsilon_ei=preset["eps_ei"],
        gp=make_gp(d), f=levy, max_iters=iters, max_partitions=preset["max_partitions"],
        backend="cupy", n_gpus=n_gpus, box_sampling=box_sampling,
        predict_batch_size=int(preset["batch"]), bounds_batch_size=int(preset["batch"]),
        max_target_boxes=int(preset["max_target"]),
        validation=False, logMask=True, normalize_to_unit_cube=True,
    )
    wall = time.perf_counter() - start
    part = [float(result.log[f"i{i}"]["time"]) for i in range(iters)]
    # The pools keep freed blocks, so their size after the run tracks its peak usage.
    peak_gb = max(pool.total_bytes() for pool in pools) / 1e9
    return result.X, part, wall, peak_gb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="moderate")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU counts to compare (default: 1, 2, 4, ... up to all visible).")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--box-sampling", choices=["lhs", "center"], default="lhs",
                        help="EI sampling inside each box: 2**d LHS points, or the center only.")
    parser.add_argument("--batch", type=float, default=None,
                        help="Override the preset's predict/bounds batch sizes (per GPU).")
    parser.add_argument("--out", type=Path, required=True, help="CSV file to append results to.")
    args = parser.parse_args()

    visible = cp.cuda.runtime.getDeviceCount()
    gpus = args.gpus or [g for g in (1, 2, 4, 8, 16) if g <= visible]
    if gpus[0] != 1:
        gpus = [1] + gpus  # the 1-GPU run is the reference for speedup and points
    preset = dict(PRESETS[args.preset])
    if args.batch is not None:
        preset["batch"] = args.batch
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"{visible} x {name} visible; preset={args.preset} {preset}; "
          f"box_sampling={args.box_sampling}; gpus={gpus}", flush=True)

    # Warm up every GPU (CUDA context, kernel loading) so the 1-GPU run is not penalized.
    run(2, dict(PRESETS["moderate"], max_partitions=3), max(gpus), 1)

    new_file = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["d", "preset", "box_sampling", "batch", "n_gpus", "iteration", "partition_s",
                             "speedup_vs_1gpu", "same_points_as_1gpu", "peak_gpu_pool_GB", "gpu"])
        for d in args.dims:
            ref_X = ref_part = None
            for g in gpus:
                X, part, wall, peak_gb = run(d, preset, g, args.iters, args.box_sampling)
                if ref_X is None:
                    ref_X, ref_part = X, part
                same = X.shape == ref_X.shape and np.allclose(X, ref_X, rtol=1e-6, atol=1e-8)
                speedup = [r / p for r, p in zip(ref_part, part)]
                print(f"RESULT d={d:2d} {args.preset} {args.box_sampling} gpus={g}: wall={wall:8.2f}s "
                      f"partition_s/iter={np.round(part, 3).tolist()} "
                      f"speedup={np.round(speedup, 2).tolist()} same_points={same} "
                      f"peak_pool={peak_gb:.1f}GB", flush=True)
                for i, (p, s) in enumerate(zip(part, speedup)):
                    writer.writerow([d, args.preset, args.box_sampling, f"{preset['batch']:.0e}", g, i,
                                     f"{p:.4f}", f"{s:.3f}", same, f"{peak_gb:.2f}", name])
                fh.flush()


if __name__ == "__main__":
    main()
