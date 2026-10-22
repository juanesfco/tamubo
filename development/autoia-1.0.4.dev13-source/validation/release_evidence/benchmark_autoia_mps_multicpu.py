"""
benchmark_autoia_mps_multicpu.py

AutoIA 0.24.0.dev14 hardware benchmark.

This benchmark deliberately separates:

A. PROPOSAL THROUGHPUT (same numerical work)
   - Torch CPU float32
   - Apple MPS float32

B. RIGOROUS PROOF THROUGHPUT (same certified work)
   - AutoIA serial CPU binary64
   - AutoIA multicore/process CPU binary64

The GPU/MPS calculations are NOT certificates.  They are proposal-only.
All rigorous enclosures come from AutoIA's trusted CPU binary64 backend.

Run from Terminal, for example:

    /opt/anaconda3/envs/autoia/bin/python benchmark_autoia_mps_multicpu.py

The script writes:
    autoia_mps_multicpu_benchmark.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

from autoia import Interval, available_accelerators, prepare, proposal_map


# ---------------------------------------------------------------------------
# User-adjustable benchmark settings
# ---------------------------------------------------------------------------

SIZES = [100, 1000, 10000]

# Proposal timings are inexpensive, so repeat them and report the median.
PROPOSAL_REPEATS = 5

# Rigorous proof runs are much more expensive.  One run is enough for the
# initial hardware characterization.
PROOF_REPEATS = 1

# Box half-widths.  These are large enough that a float32 center proposal
# should comfortably lie inside the rigorous box certificate.
HALF_WIDTH_X = 0.01
HALF_WIDTH_Y = 0.01

CSV_PATH = Path("autoia_mps_multicpu_benchmark.csv")


# ---------------------------------------------------------------------------
# Same mathematical model in NumPy and Torch
# ---------------------------------------------------------------------------

def engineering_model(p):
    """Ordinary NumPy model used for rigorous AutoIA certification."""
    x, y = p
    return (
        np.exp(-x * x)
        + 0.25 * np.sin(3.0 * y)
        + x * y
        + 0.10 * x * x * y * y
    )


def torch_model(z):
    """Torch version used only for CPU/MPS proposal throughput."""
    x = z[:, 0]
    y = z[:, 1]
    return (
        torch.exp(-x * x)
        + 0.25 * torch.sin(3.0 * y)
        + x * y
        + 0.10 * x * x * y * y
    )


# torch is imported at module scope so spawned AutoIA worker processes can
# import this file cleanly on macOS.
import torch


# ---------------------------------------------------------------------------
# Deterministic boxes
# ---------------------------------------------------------------------------

def make_centers(n):
    """
    Deterministic quasi-grid-like centers in a 2D design region.

    x in approximately [-0.98, 0.98]
    y in approximately [-0.48, 0.48]
    """
    i = np.arange(n, dtype=np.float64)

    # Two irrational rotations give a deterministic, well-spread sequence.
    u = np.mod((i + 0.5) * 0.6180339887498949, 1.0)
    v = np.mod((i + 0.5) * 0.4142135623730950, 1.0)

    x = -0.98 + 1.96 * u
    y = -0.48 + 0.96 * v
    return np.column_stack([x, y])


def make_boxes(centers):
    boxes = []
    for x, y in centers:
        boxes.append([
            Interval(float(x - HALF_WIDTH_X), float(x + HALF_WIDTH_X)),
            Interval(float(y - HALF_WIDTH_Y), float(y + HALF_WIDTH_Y)),
        ])
    return boxes


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def synchronize_if_needed(device):
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_proposal(centers, device, repeats=PROPOSAL_REPEATS):
    """
    Time AutoIA proposal_map on a Torch-compatible point-valued function.

    This is NOT a proof computation.
    """
    times = []
    last = None

    for _ in range(repeats):
        synchronize_if_needed(device)
        t0 = time.perf_counter()
        last = proposal_map(
            torch_model,
            centers,
            device=device,
            dtype="float32",
        )
        synchronize_if_needed(device)
        times.append(time.perf_counter() - t0)

    return float(np.median(times)), last


def time_proof(session, boxes, workers, parallel):
    """
    Time rigorous AutoIA binary64 many-box certification.
    """
    t0 = time.perf_counter()
    lower, upper, info = session.enclose_many_bounds(
        boxes,
        workers=workers,
        parallel=parallel,
        return_info=True,
        use_cache=False,
    )
    wall = time.perf_counter() - t0
    return wall, lower, upper, info


def throughput(n, seconds):
    return float(n / seconds) if seconds > 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("AutoIA 0.24.0.dev14: Apple MPS + multicore CPU benchmark")
    print("=" * 78)

    accel = available_accelerators()
    print("\nAccelerator report:")
    for name, item in accel.items():
        print(name, item)

    has_mps = bool(accel.get("mps") and accel["mps"].available)

    if not has_mps:
        print("\nWARNING: MPS is not available. MPS timing will be skipped.")

    # Prepare the rigorous model once.
    session = prepare(engineering_model)

    # Warm up the trusted serial proof path so first-use preparation is not
    # charged to the measured batches.
    warm_centers = make_centers(4)
    warm_boxes = make_boxes(warm_centers)
    _ = session.enclose_many_bounds(
        warm_boxes,
        workers=1,
        parallel="serial",
        return_info=False,
        use_cache=False,
    )

    # Warm up proposal paths.  MPS in particular can have substantial first-call
    # initialization overhead that should not dominate steady-state throughput.
    proposal_warm = make_centers(2048)
    _ = proposal_map(torch_model, proposal_warm, device="cpu", dtype="float32")
    if has_mps:
        _ = proposal_map(torch_model, proposal_warm, device="mps", dtype="float32")
        synchronize_if_needed("mps")

    rows = []

    for n in SIZES:
        print("\n" + "-" * 78)
        print(f"N = {n:,} boxes / centers")
        print("-" * 78)

        centers = make_centers(n)
        boxes = make_boxes(centers)

        # ---------------------------------------------------------------
        # A. Same proposal calculation: Torch CPU vs Apple MPS
        # ---------------------------------------------------------------
        cpu_prop_time, cpu_prop = time_proposal(
            centers, device="cpu", repeats=PROPOSAL_REPEATS
        )

        print(
            f"Proposal CPU float32: {cpu_prop_time:.6f} s, "
            f"{throughput(n, cpu_prop_time):,.1f} points/s"
        )

        rows.append({
            "n": n,
            "category": "proposal",
            "backend": "torch-cpu-float32",
            "workers": "",
            "parallel": "",
            "seconds": cpu_prop_time,
            "throughput_per_second": throughput(n, cpu_prop_time),
            "machine_rigorous": False,
            "resolved_workers": "",
            "resolved_parallel": "",
            "notes": "median proposal_map timing",
        })

        mps_prop = None
        if has_mps:
            mps_prop_time, mps_prop = time_proposal(
                centers, device="mps", repeats=PROPOSAL_REPEATS
            )

            speedup = cpu_prop_time / mps_prop_time
            cpu_vals = np.asarray(cpu_prop.values, dtype=np.float64)
            mps_vals = np.asarray(mps_prop.values, dtype=np.float64)
            max_abs_diff = float(np.max(np.abs(cpu_vals - mps_vals)))

            print(
                f"Proposal MPS float32: {mps_prop_time:.6f} s, "
                f"{throughput(n, mps_prop_time):,.1f} points/s"
            )
            print(f"MPS proposal speedup vs Torch CPU: {speedup:.3f}x")
            print(
                "Max |Torch CPU proposal - MPS proposal|: "
                f"{max_abs_diff:.3e}"
            )
            print(
                "MPS proposal machine_rigorous:",
                mps_prop.machine_rigorous,
            )

            rows.append({
                "n": n,
                "category": "proposal",
                "backend": "apple-mps-float32",
                "workers": "",
                "parallel": "",
                "seconds": mps_prop_time,
                "throughput_per_second": throughput(n, mps_prop_time),
                "machine_rigorous": bool(mps_prop.machine_rigorous),
                "resolved_workers": "",
                "resolved_parallel": "",
                "notes": (
                    f"median proposal_map timing; speedup_vs_cpu={speedup:.6g}; "
                    f"max_abs_diff={max_abs_diff:.6g}"
                ),
            })

        # ---------------------------------------------------------------
        # B. Same rigorous work: serial CPU vs multicore CPU
        # ---------------------------------------------------------------
        serial_wall, lo_serial, hi_serial, info_serial = time_proof(
            session,
            boxes,
            workers=1,
            parallel="serial",
        )

        print(
            f"Rigorous serial CPU: {serial_wall:.6f} s, "
            f"{throughput(n, serial_wall):,.1f} boxes/s"
        )
        print("  ", info_serial)

        rows.append({
            "n": n,
            "category": "proof",
            "backend": "autoia-cpu-binary64",
            "workers": 1,
            "parallel": "serial",
            "seconds": serial_wall,
            "throughput_per_second": throughput(n, serial_wall),
            "machine_rigorous": bool(info_serial.machine_rigorous),
            "resolved_workers": info_serial.resolved_workers,
            "resolved_parallel": info_serial.resolved_parallel,
            "notes": "trusted rigorous proof",
        })

        multi_wall, lo_multi, hi_multi, info_multi = time_proof(
            session,
            boxes,
            workers="auto",
            parallel="process",
        )

        cpu_speedup = serial_wall / multi_wall

        print(
            f"Rigorous multicore CPU: {multi_wall:.6f} s, "
            f"{throughput(n, multi_wall):,.1f} boxes/s"
        )
        print("  ", info_multi)
        print(f"Multicore proof speedup vs serial: {cpu_speedup:.3f}x")

        rows.append({
            "n": n,
            "category": "proof",
            "backend": "autoia-cpu-binary64",
            "workers": "auto",
            "parallel": "process",
            "seconds": multi_wall,
            "throughput_per_second": throughput(n, multi_wall),
            "machine_rigorous": bool(info_multi.machine_rigorous),
            "resolved_workers": info_multi.resolved_workers,
            "resolved_parallel": info_multi.resolved_parallel,
            "notes": f"trusted rigorous proof; speedup_vs_serial={cpu_speedup:.6g}",
        })

        # Scheduling is not allowed to alter rigor.  Bitwise equality is a
        # useful diagnostic but is not required by the mathematical contract.
        exact_same = (
            np.array_equal(lo_serial, lo_multi)
            and np.array_equal(hi_serial, hi_multi)
        )

        max_endpoint_diff = max(
            float(np.max(np.abs(lo_serial - lo_multi))),
            float(np.max(np.abs(hi_serial - hi_multi))),
        )

        print("Serial/process endpoints exactly identical:", exact_same)
        print(
            "Maximum serial/process endpoint difference:",
            f"{max_endpoint_diff:.3e}",
        )
        print(
            "Both proof paths machine rigorous:",
            bool(info_serial.machine_rigorous and info_multi.machine_rigorous),
        )

        # ---------------------------------------------------------------
        # C. End-to-end architectural check:
        #    Is the MPS center proposal contained in the CPU box proof?
        # ---------------------------------------------------------------
        if mps_prop is not None:
            mps_vals = np.asarray(mps_prop.values, dtype=np.float64).reshape(-1)
            lo = np.asarray(lo_multi, dtype=np.float64).reshape(n, -1)[:, 0]
            hi = np.asarray(hi_multi, dtype=np.float64).reshape(n, -1)[:, 0]

            contained = (lo <= mps_vals) & (mps_vals <= hi)
            n_contained = int(np.sum(contained))

            print(
                f"MPS center proposals contained in rigorous CPU box bounds: "
                f"{n_contained}/{n}"
            )

            if n_contained != n:
                bad = np.flatnonzero(~contained)[:10]
                print("First containment misses:", bad.tolist())

    # -------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------
    fieldnames = [
        "n",
        "category",
        "backend",
        "workers",
        "parallel",
        "seconds",
        "throughput_per_second",
        "machine_rigorous",
        "resolved_workers",
        "resolved_parallel",
        "notes",
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("Benchmark complete.")
    print(f"CSV written to: {CSV_PATH.resolve()}")
    print("=" * 78)
    print(
        "\nInterpretation rule:\n"
        "  - Torch CPU vs MPS compares the SAME point-valued proposal calculation.\n"
        "  - Serial vs multicore CPU compares the SAME rigorous AutoIA proof work.\n"
        "  - Do NOT compare MPS proposal time directly with AutoIA proof time as if\n"
        "    they were equivalent computations."
    )


if __name__ == "__main__":
    main()
