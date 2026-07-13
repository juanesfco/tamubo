#!/usr/bin/env python3
"""Repeated all-active 2D DIRECT splitting with one native launch per step.

This is a memory-behavior smoke test: every step writes the current boxes to a
file, launches exactbo_split_boxes, reads the next boxes, and repeats. Since the
native executable exits after each step, CUDA allocations and the CUDA context
from that process are released between steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from split_boxes_workflow import launch_native, read_split_output, write_split_input


def initial_2d_box(domain_l, domain_u):
    domain_l = np.asarray(domain_l, dtype=np.float64).reshape(2)
    domain_u = np.asarray(domain_u, dtype=np.float64).reshape(2)
    if np.any(domain_u <= domain_l):
        raise ValueError("domain upper bounds must be larger than lower bounds")
    return domain_l.reshape(1, 2), domain_u.reshape(1, 2), domain_u - domain_l


def run_partition_all_boxes(
    *,
    workdir,
    native_build_dir,
    split_exe=None,
    mpi_ranks=1,
    steps=3,
    max_boxes=10000000000,
    domain_l=(0.0, 0.0),
    domain_u=(1.0, 1.0),
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(split_exe) if split_exe else Path(native_build_dir) / "exactbo_split_boxes"

    if int(steps) < 0:
        raise ValueError("steps must be >= 0")
    if int(max_boxes) <= 0:
        raise ValueError("max_boxes must be > 0")

    bounds_l, bounds_u, domain_width = initial_2d_box(domain_l, domain_u)
    np.save(workdir / "domain_L.npy", np.asarray(domain_l, dtype=np.float64))
    np.save(workdir / "domain_U.npy", np.asarray(domain_u, dtype=np.float64))
    np.save(workdir / "domain_width.npy", domain_width)

    history = []
    for step in range(int(steps)):
        n_boxes, d = bounds_l.shape
        active_mask = np.ones(n_boxes, dtype=np.uint8)
        expected_next = n_boxes * (2 * d + 1)
        if expected_next > int(max_boxes):
            raise RuntimeError(
                f"step {step} would create {expected_next} boxes, above max_boxes={max_boxes}"
            )

        step_dir = workdir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        input_path = step_dir / "split_boxes_input.bin"
        output_path = step_dir / "split_boxes_output.bin"

        np.save(step_dir / "input_bounds_L.npy", bounds_l)
        np.save(step_dir / "input_bounds_U.npy", bounds_u)
        np.save(step_dir / "active_mask.npy", active_mask.astype(bool))
        write_split_input(input_path, bounds_l, bounds_u, active_mask, domain_width, keep_inactive=False)

        print(
            f"step {step}: input_boxes={n_boxes} active_boxes={n_boxes} expected_output_boxes={expected_next}",
            flush=True,
        )
        launch_native(exe, int(mpi_ranks), input_path, output_path)

        next_l, next_u, meta = read_split_output(output_path)
        np.save(step_dir / "output_bounds_L.npy", next_l)
        np.save(step_dir / "output_bounds_U.npy", next_u)

        history.append(
            {
                "step": step,
                "input_boxes": int(n_boxes),
                "output_boxes": int(next_l.shape[0]),
                "d": int(d),
                "stride": int(meta["stride"]),
                "n_active_reported": int(meta["n_active"]),
                "input_file": str(input_path.relative_to(workdir)),
                "output_file": str(output_path.relative_to(workdir)),
            }
        )
        bounds_l, bounds_u = next_l, next_u

    np.save(workdir / "final_bounds_L.npy", bounds_l)
    np.save(workdir / "final_bounds_U.npy", bounds_u)
    manifest = {
        "purpose": "Repeated all-active 2D split_boxes memory test",
        "executable": str(exe),
        "mpi_ranks": int(mpi_ranks),
        "steps": int(steps),
        "max_boxes": int(max_boxes),
        "final_boxes": int(bounds_l.shape[0]),
        "files": {
            "domain_L": "domain_L.npy",
            "domain_U": "domain_U.npy",
            "domain_width": "domain_width.npy",
            "final_bounds_L": "final_bounds_L.npy",
            "final_bounds_U": "final_bounds_U.npy",
        },
        "history": history,
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"saved workflow files in {workdir}")
    print(f"final_bounds_L shape={bounds_l.shape}")
    print(f"final_bounds_U shape={bounds_u.shape}")
    return bounds_l, bounds_u, manifest


def main():
    parser = argparse.ArgumentParser(description="Repeatedly split all current 2D boxes with exactbo_split_boxes.")
    parser.add_argument("--workdir", default="native/exactbo/data/partition_all_boxes_workflow")
    parser.add_argument("--native-build-dir", default="native/build")
    parser.add_argument("--split-exe", default=None)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-boxes", type=int, default=10000000000)
    parser.add_argument("--domain-l", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--domain-u", type=float, nargs=2, default=(1.0, 1.0))
    args = parser.parse_args()

    run_partition_all_boxes(
        workdir=args.workdir,
        native_build_dir=args.native_build_dir,
        split_exe=args.split_exe,
        mpi_ranks=args.mpi_ranks,
        steps=args.steps,
        max_boxes=args.max_boxes,
        domain_l=args.domain_l,
        domain_u=args.domain_u,
    )


if __name__ == "__main__":
    main()
