#!/usr/bin/env python3
"""Repeated all-active 2D DIRECT splitting with bounded host memory.

By default this workflow keeps one rolling native input and output. It copies
the planar lower/upper arrays between those binary files in fixed-size chunks,
so a large generation is never materialized as a NumPy array. Use
``--debug-save-numpy`` only for small runs that need per-step inspection files.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from split_boxes_workflow import (
    SPLIT_INPUT_MAGIC,
    SPLIT_OUTPUT_MAGIC,
    launch_native,
    read_split_output,
    write_split_input,
)

_SPLIT_OUTPUT_HEADER_BYTES = 8 + 4 * 8
_COPY_CHUNK_BYTES = 64 * 1024 * 1024


def initial_2d_box(domain_l, domain_u):
    domain_l = np.asarray(domain_l, dtype=np.float64).reshape(2)
    domain_u = np.asarray(domain_u, dtype=np.float64).reshape(2)
    if np.any(domain_u <= domain_l):
        raise ValueError("domain upper bounds must be larger than lower bounds")
    return domain_l.reshape(1, 2), domain_u.reshape(1, 2), domain_u - domain_l


def read_split_output_metadata(path: Path):
    """Read and validate a native split output without loading its arrays."""
    path = Path(path)
    with path.open("rb") as f:
        if f.read(8) != SPLIT_OUTPUT_MAGIC:
            raise ValueError(f"{path} is not a split_boxes output file")
        header = f.read(32)
        if len(header) != 32:
            raise ValueError(f"{path} has a truncated split_boxes header")
        n_out, d, n_active, stride = struct.unpack("<QQQQ", header)

    expected_bytes = _SPLIT_OUTPUT_HEADER_BYTES + 2 * n_out * d * 8
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes} "
            f"for {n_out} boxes in {d} dimensions"
        )
    return {"n_out": n_out, "d": d, "n_active": n_active, "stride": stride}


def _copy_exact(source, destination, n_bytes: int, *, chunk_bytes: int = _COPY_CHUNK_BYTES):
    remaining = int(n_bytes)
    while remaining:
        block = source.read(min(remaining, int(chunk_bytes)))
        if not block:
            raise ValueError("split output ended while copying its bound arrays")
        destination.write(block)
        remaining -= len(block)


def write_all_active_input_from_output(
    output_path: Path,
    input_path: Path,
    domain_width,
    *,
    chunk_bytes: int = _COPY_CHUNK_BYTES,
):
    """Convert TSPLOU1 planar bounds to TSPLIN1 using bounded buffers."""
    output_path = Path(output_path)
    input_path = Path(input_path)
    meta = read_split_output_metadata(output_path)
    n = int(meta["n_out"])
    d = int(meta["d"])
    domain_width = np.asarray(domain_width, dtype=np.float64).reshape(-1)
    if domain_width.shape != (d,):
        raise ValueError(f"domain_width must have shape ({d},)")
    if int(meta["stride"]) != 2 * d + 1:
        raise ValueError(f"unexpected split stride {meta['stride']} for d={d}")

    input_path.parent.mkdir(parents=True, exist_ok=True)
    bound_bytes = 2 * n * d * 8
    with output_path.open("rb") as source, input_path.open("wb") as destination:
        source.seek(_SPLIT_OUTPUT_HEADER_BYTES)
        destination.write(SPLIT_INPUT_MAGIC)
        destination.write(struct.pack("<QQQ", n, d, 0))
        destination.write(np.ascontiguousarray(domain_width, dtype=np.float64).tobytes())
        _copy_exact(source, destination, bound_bytes, chunk_bytes=chunk_bytes)

        mask_chunk = b"\x01" * min(max(1, int(chunk_bytes)), max(1, n))
        remaining = n
        while remaining:
            count = min(remaining, len(mask_chunk))
            destination.write(mask_chunk[:count])
            remaining -= count
    return meta


def mmap_split_output(path: Path):
    """Return read-only views of the final planar output without copying it."""
    path = Path(path)
    meta = read_split_output_metadata(path)
    n = int(meta["n_out"])
    d = int(meta["d"])
    if n == 0:
        empty = np.empty((0, d), dtype=np.float64)
        return empty, empty.copy(), meta
    plane_bytes = n * d * 8
    bounds_l = np.memmap(
        path,
        mode="r",
        dtype="<f8",
        offset=_SPLIT_OUTPUT_HEADER_BYTES,
        shape=(n, d),
    )
    bounds_u = np.memmap(
        path,
        mode="r",
        dtype="<f8",
        offset=_SPLIT_OUTPUT_HEADER_BYTES + plane_bytes,
        shape=(n, d),
    )
    return bounds_l, bounds_u, meta


def run_partition_all_boxes(
    *,
    workdir,
    native_build_dir,
    split_exe=None,
    mpi_ranks=1,
    split_batch_parents=0,
    steps=3,
    max_boxes=10000000000,
    domain_l=(0.0, 0.0),
    domain_u=(1.0, 1.0),
    debug_save_numpy=False,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(split_exe) if split_exe else Path(native_build_dir) / "exactbo_split_boxes"

    if int(steps) < 0:
        raise ValueError("steps must be >= 0")
    if int(max_boxes) <= 0:
        raise ValueError("max_boxes must be > 0")
    if int(split_batch_parents) < 0:
        raise ValueError("split_batch_parents must be nonnegative (0 selects automatic sizing)")

    initial_l, initial_u, domain_width = initial_2d_box(domain_l, domain_u)
    np.save(workdir / "domain_L.npy", np.asarray(domain_l, dtype=np.float64))
    np.save(workdir / "domain_U.npy", np.asarray(domain_u, dtype=np.float64))
    np.save(workdir / "domain_width.npy", domain_width)

    rolling_input = workdir / "split_boxes_input.bin"
    rolling_output = workdir / "split_boxes_output.bin"
    next_input = workdir / "split_boxes_input.next.bin"
    final_output = workdir / "final_split_boxes_output.bin"
    if not debug_save_numpy:
        for stale in (rolling_input, rolling_output, next_input, final_output):
            stale.unlink(missing_ok=True)
        write_split_input(
            rolling_input,
            initial_l,
            initial_u,
            np.ones(1, dtype=np.uint8),
            domain_width,
            keep_inactive=False,
        )

    bounds_l = initial_l
    bounds_u = initial_u
    n_boxes = 1
    history = []
    final_meta = None

    for step in range(int(steps)):
        d = 2
        expected_next = n_boxes * (2 * d + 1)
        if expected_next > int(max_boxes):
            raise RuntimeError(
                f"step {step} would create {expected_next} boxes, above max_boxes={max_boxes}"
            )

        if debug_save_numpy:
            step_dir = workdir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            input_path = step_dir / "split_boxes_input.bin"
            output_path = step_dir / "split_boxes_output.bin"
            active_mask = np.ones(n_boxes, dtype=np.uint8)
            np.save(step_dir / "input_bounds_L.npy", bounds_l)
            np.save(step_dir / "input_bounds_U.npy", bounds_u)
            np.save(step_dir / "active_mask.npy", active_mask.astype(bool))
            write_split_input(
                input_path,
                bounds_l,
                bounds_u,
                active_mask,
                domain_width,
                keep_inactive=False,
            )
        else:
            input_path = rolling_input
            output_path = rolling_output

        print(
            f"step {step}: input_boxes={n_boxes} active_boxes={n_boxes} "
            f"expected_output_boxes={expected_next}",
            flush=True,
        )
        launch_native(
            exe,
            int(mpi_ranks),
            input_path,
            output_path,
            split_batch_parents=split_batch_parents,
        )

        meta = read_split_output_metadata(output_path)
        if int(meta["n_out"]) != expected_next or int(meta["n_active"]) != n_boxes:
            raise RuntimeError(
                "native split count mismatch: "
                f"expected active={n_boxes}, output={expected_next}; got {meta}"
            )
        final_meta = meta
        record = {
            "step": step,
            "input_boxes": int(n_boxes),
            "output_boxes": int(meta["n_out"]),
            "d": int(meta["d"]),
            "stride": int(meta["stride"]),
            "n_active_reported": int(meta["n_active"]),
            "retained_step_files": bool(debug_save_numpy),
        }

        if debug_save_numpy:
            next_l, next_u, _ = read_split_output(output_path)
            np.save(step_dir / "output_bounds_L.npy", next_l)
            np.save(step_dir / "output_bounds_U.npy", next_u)
            record.update(
                {
                    "input_file": str(input_path.relative_to(workdir)),
                    "output_file": str(output_path.relative_to(workdir)),
                }
            )
            bounds_l, bounds_u = next_l, next_u
        elif step + 1 < int(steps):
            write_all_active_input_from_output(output_path, next_input, domain_width)
            next_input.replace(rolling_input)
            rolling_output.unlink()

        history.append(record)
        n_boxes = int(meta["n_out"])

    files = {
        "domain_L": "domain_L.npy",
        "domain_U": "domain_U.npy",
        "domain_width": "domain_width.npy",
    }
    if debug_save_numpy:
        np.save(workdir / "final_bounds_L.npy", bounds_l)
        np.save(workdir / "final_bounds_U.npy", bounds_u)
        files.update(
            {
                "final_bounds_L": "final_bounds_L.npy",
                "final_bounds_U": "final_bounds_U.npy",
            }
        )
    elif int(steps) > 0:
        rolling_output.replace(final_output)
        rolling_input.unlink(missing_ok=True)
        bounds_l, bounds_u, final_meta = mmap_split_output(final_output)
        history[-1]["output_file"] = final_output.name
        history[-1]["retained_step_files"] = True
        files["final_split_output"] = final_output.name
    else:
        rolling_input.unlink(missing_ok=True)

    manifest = {
        "purpose": "Repeated all-active 2D split_boxes memory test",
        "executable": str(exe),
        "mpi_ranks": int(mpi_ranks),
        "split_batch_parents": int(split_batch_parents),
        "steps": int(steps),
        "max_boxes": int(max_boxes),
        "debug_save_numpy": bool(debug_save_numpy),
        "stream_copy_chunk_bytes": int(_COPY_CHUNK_BYTES),
        "final_boxes": int(n_boxes),
        "files": files,
        "history": history,
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"saved workflow files in {workdir}")
    print(f"final_bounds_L shape={bounds_l.shape}")
    print(f"final_bounds_U shape={bounds_u.shape}")
    if final_meta is not None:
        print(f"final native metadata={final_meta}")
    return bounds_l, bounds_u, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Repeatedly split all current 2D boxes with exactbo_split_boxes."
    )
    parser.add_argument("--workdir", default="native/exactbo/data/partition_all_boxes_workflow")
    parser.add_argument("--native-build-dir", default="native/build")
    parser.add_argument("--split-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument(
        "--split-batch-parents",
        type=int,
        default=0,
        help="Maximum selected parents per CUDA split batch; 0 chooses automatically.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-boxes", type=int, default=10000000000)
    parser.add_argument("--domain-l", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--domain-u", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument(
        "--debug-save-numpy",
        action="store_true",
        help="Materialize and retain per-step NumPy arrays and native files (small runs only).",
    )
    args = parser.parse_args()

    run_partition_all_boxes(
        workdir=args.workdir,
        native_build_dir=args.native_build_dir,
        split_exe=args.split_exe,
        mpi_ranks=args.mpi_ranks,
        split_batch_parents=args.split_batch_parents,
        steps=args.steps,
        max_boxes=args.max_boxes,
        domain_l=args.domain_l,
        domain_u=args.domain_u,
        debug_save_numpy=args.debug_save_numpy,
    )


if __name__ == "__main__":
    main()
