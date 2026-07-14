#!/usr/bin/env python3
"""Small file-based workflow for the native DIRECT-style split_boxes kernel."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

SPLIT_INPUT_MAGIC = b"TSPLIN1!"
SPLIT_OUTPUT_MAGIC = b"TSPLOU1!"


def make_example_boxes():
    bounds_l = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    bounds_u = np.array(
        [
            [0.5, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    active_mask = np.array([True, False, True, True], dtype=np.bool_)
    domain_width = np.array([1.0, 1.0], dtype=np.float64)
    return bounds_l, bounds_u, active_mask, domain_width


def _write_array(f, values):
    f.write(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def write_split_input(path: Path, bounds_l, bounds_u, active_mask, domain_width, *, keep_inactive=True):
    bounds_l = np.asarray(bounds_l, dtype=np.float64)
    bounds_u = np.asarray(bounds_u, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=np.uint8).reshape(-1)
    domain_width = np.asarray(domain_width, dtype=np.float64).reshape(-1)

    n, d = bounds_l.shape
    if bounds_u.shape != (n, d):
        raise ValueError("bounds_u must have the same shape as bounds_l")
    if active_mask.shape != (n,):
        raise ValueError("active_mask must have shape (n,)")
    if domain_width.shape != (d,):
        raise ValueError("domain_width must have shape (d,)")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(SPLIT_INPUT_MAGIC)
        f.write(struct.pack("<QQQ", n, d, int(keep_inactive)))
        _write_array(f, domain_width)
        _write_array(f, bounds_l)
        _write_array(f, bounds_u)
        f.write(np.ascontiguousarray(active_mask, dtype=np.uint8).tobytes())


def read_split_output(path: Path):
    with path.open("rb") as f:
        if f.read(8) != SPLIT_OUTPUT_MAGIC:
            raise ValueError(f"{path} is not a split_boxes output file")
        n_out, d, n_active, stride = struct.unpack("<QQQQ", f.read(32))
        count = n_out * d
        bounds_l = np.frombuffer(f.read(8 * count), dtype="<f8").copy().reshape(n_out, d)
        bounds_u = np.frombuffer(f.read(8 * count), dtype="<f8").copy().reshape(n_out, d)
    return bounds_l, bounds_u, {"n_out": n_out, "d": d, "n_active": n_active, "stride": stride}


def launch_native(
    exe: Path,
    mpi_ranks: int,
    input_path: Path,
    output_path: Path,
    *,
    split_batch_parents: int = 0,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if int(split_batch_parents) < 0:
        raise ValueError("split_batch_parents must be nonnegative (0 selects automatic sizing)")
    if mpi_ranks <= 1:
        cmd = [str(exe), "--input", str(input_path), "--output", str(output_path)]
    else:
        cmd = ["mpirun", "-np", str(mpi_ranks), str(exe), "--input", str(input_path), "--output", str(output_path)]
    cmd.extend(("--split-batch-parents", str(int(split_batch_parents))))
    print("launch:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_example(
    *,
    workdir,
    native_build_dir,
    split_exe=None,
    mpi_ranks=1,
    keep_inactive=True,
    split_batch_parents=0,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(split_exe) if split_exe else Path(native_build_dir) / "exactbo_split_boxes"

    bounds_l, bounds_u, active_mask, domain_width = make_example_boxes()
    np.save(workdir / "input_bounds_L.npy", bounds_l)
    np.save(workdir / "input_bounds_U.npy", bounds_u)
    np.save(workdir / "active_mask.npy", active_mask)
    np.save(workdir / "domain_width.npy", domain_width)

    input_path = workdir / "split_boxes_input.bin"
    output_path = workdir / "split_boxes_output.bin"
    write_split_input(input_path, bounds_l, bounds_u, active_mask, domain_width, keep_inactive=keep_inactive)
    launch_native(
        exe,
        mpi_ranks,
        input_path,
        output_path,
        split_batch_parents=split_batch_parents,
    )

    out_l, out_u, meta = read_split_output(output_path)
    np.save(workdir / "split_bounds_L.npy", out_l)
    np.save(workdir / "split_bounds_U.npy", out_u)
    (workdir / "manifest.json").write_text(
        json.dumps(
            {
                "executable": str(exe),
                "mpi_ranks": int(mpi_ranks),
                "keep_inactive": bool(keep_inactive),
                "split_batch_parents": int(split_batch_parents),
                **{k: int(v) for k, v in meta.items()},
                "files": {
                    "input_bounds_L": "input_bounds_L.npy",
                    "input_bounds_U": "input_bounds_U.npy",
                    "active_mask": "active_mask.npy",
                    "domain_width": "domain_width.npy",
                    "split_bounds_L": "split_bounds_L.npy",
                    "split_bounds_U": "split_bounds_U.npy",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved workflow files in {workdir}")
    print(f"split_bounds_L shape={out_l.shape}")
    print(f"split_bounds_U shape={out_u.shape}")
    print(f"metadata={meta}")
    return out_l, out_u, meta


def main():
    parser = argparse.ArgumentParser(description="Run a small native split_boxes workflow.")
    parser.add_argument("--workdir", default="native/exactbo/data/split_boxes_workflow")
    parser.add_argument("--native-build-dir", default="native/build")
    parser.add_argument("--split-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument(
        "--split-batch-parents",
        type=int,
        default=0,
        help="Maximum selected parents per CUDA split batch; 0 chooses automatically.",
    )
    parser.add_argument("--drop-inactive", action="store_true")
    args = parser.parse_args()

    run_example(
        workdir=args.workdir,
        native_build_dir=args.native_build_dir,
        split_exe=args.split_exe,
        mpi_ranks=args.mpi_ranks,
        keep_inactive=not args.drop_inactive,
        split_batch_parents=args.split_batch_parents,
    )


if __name__ == "__main__":
    main()
