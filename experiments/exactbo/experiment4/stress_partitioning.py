"""
ExactBO partitioning stress runs on TAMU Vision (experiment-3 GP, no max_target_boxes cap).

Like experiment3/run_partitioning.py, plus the knobs the stress test varies
(GPUs, box sampling, AutoBound degree/batch) and peak GPU memory. Appends one
row per partition to --out and one JSON line per run to --summary (target
counts, chosen point, incumbent, memory), so runs can be compared exactly.

    python stress_partitioning.py --acquisition ei --max-partitions 9 --n-gpus 8 --label ei/autobound/8gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiment3"))

from common import make_gp  # noqa: E402

from tamubo.exactbo import exactbo_partitioning  # noqa: E402


def peak_memory_gb() -> dict:
    """Largest CuPy pool size and JAX peak allocation over the visible GPUs (GB)."""
    import cupy as cp

    pools = []
    for device in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device):
            pools.append(cp.get_default_memory_pool().total_bytes())
    out = {"cupy_pool_gb": max(pools) / 1e9}
    if "jax" in sys.modules:
        jax = sys.modules["jax"]
        peaks = [(dev.memory_stats() or {}).get("peak_bytes_in_use", 0) for dev in jax.devices("gpu")]
        out["jax_peak_gb"] = max(peaks) / 1e9
    return out


def _report_splits(t0: float) -> None:
    """Wrap exactbo's split_boxes to report each split; a run that runs out of memory usually dies there."""
    import tamubo.exactbo.run as run_mod

    split = run_mod.split_boxes
    count = [0]

    def split_with_progress(bounds_L, bounds_U, mask, *args, **kwargs):
        count[0] += 1
        n_split = int(mask.sum())
        print(f"PROGRESS split {count[0]}: {n_split} targets -> {n_split * (2 * bounds_L.shape[1] + 1)} boxes, "
              f"t={time.perf_counter() - t0:.1f}s, before {peak_memory_gb()}", flush=True)
        out = split(bounds_L, bounds_U, mask, *args, **kwargs)
        print(f"PROGRESS split {count[0]} done, after {peak_memory_gb()}", flush=True)
        return out

    run_mod.split_boxes = split_with_progress


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="problem10d")
    ap.add_argument("--n0", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon-ei", type=float, default=0.1)
    ap.add_argument("--epsilon-x", type=float, default=1e-5)
    ap.add_argument("--max-partitions", type=int, default=9)
    ap.add_argument("--acquisition", default="logei", choices=["logei", "ei"])
    ap.add_argument("--bound-method", default="autobound", choices=["autobound", "interval"])
    ap.add_argument("--box-sampling", default="lhs", choices=["lhs", "center"])
    ap.add_argument("--autobound-degree", type=int, default=2)
    ap.add_argument("--autobound-batch-size", type=int, default=8192)
    ap.add_argument("--n-gpus", type=int, default=None, help="default: all visible GPUs")
    ap.add_argument("--predict-batch-size", type=float, default=None)
    ap.add_argument("--bounds-batch-size", type=float, default=None)
    ap.add_argument("--verbose", action="store_true", help="exactbo's own (very chatty) progress output")
    ap.add_argument("--progress", action="store_true",
                    help="one line before/after each box split with target count and memory")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(HERE / "results" / "stress_partition_counts.csv"))
    ap.add_argument("--summary", default=str(HERE / "results" / "stress_runs.jsonl"))
    args = ap.parse_args()

    gp, X, bounds = make_gp(args.problem, args.n0, args.seed)
    d = X.shape[1]
    t = time.perf_counter()
    if args.progress:
        _report_splits(t)
    result = exactbo_partitioning(
        X,
        bounds,
        np.full(d, args.epsilon_x),
        args.epsilon_ei,
        gp,
        args.max_partitions,
        backend="cupy",
        n_gpus=args.n_gpus,
        box_sampling=args.box_sampling,
        acquisition=args.acquisition,
        bound_method=args.bound_method,
        autobound_degree=args.autobound_degree,
        autobound_batch_size=args.autobound_batch_size,
        predict_batch_size=None if args.predict_batch_size is None else int(args.predict_batch_size),
        bounds_batch_size=None if args.bounds_batch_size is None else int(args.bounds_batch_size),
        validation=False,
        verbose=args.verbose,
        logMask=True,
    )
    elapsed = time.perf_counter() - t
    memory = peak_memory_gb()
    label = args.label or f"{args.acquisition}/{args.bound_method}"
    parts = result.log["partitions"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "problem", "n0", "seed", "epsilon_ei", "n_gpus", "box_sampling", "autobound_degree",
              "partition", "n_boxes", "n_active", "n_target", "max_acq_hi", "incumbent", "time_s"]
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for p, row in enumerate(parts):
            writer.writerow(dict(
                label=label, problem=args.problem, n0=args.n0, seed=args.seed, epsilon_ei=args.epsilon_ei,
                n_gpus=args.n_gpus or "all", box_sampling=args.box_sampling,
                autobound_degree=args.autobound_degree, partition=p, n_boxes=row["n_boxes"],
                n_active=row["n_active"], n_target=row["n_target"], max_acq_hi=row["max_ei_hi"],
                incumbent=row["best"], time_s=round(row["time"], 2),
            ))
            print(f"partition {p}: boxes {row['n_boxes']}, active {row['n_active']}, target {row['n_target']}, "
                  f"incumbent {row['best']:.6f}, t={row['time']:.1f}s", flush=True)

    summary = dict(
        label=label, args=vars(args), elapsed_s=round(elapsed, 2), partitions=len(parts),
        n_target=[int(r["n_target"]) for r in parts], incumbent=float(parts[-1]["best"]) if parts else None,
        ei_max_scaled=float(result.log["ei_max_scaled"]), x=np.asarray(result.X, dtype=float).ravel().tolist(),
        **{k: round(v, 2) for k, v in memory.items()},
    )
    with Path(args.summary).open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"RUN {label}: {elapsed:.1f}s, partitions {len(parts)}, incumbent {summary['incumbent']:.6f}, "
          f"memory {memory}", flush=True)


if __name__ == "__main__":
    main()
