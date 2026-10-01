"""
ExactBO partitioning on the experiment-2 GP: target boxes, incumbent and time per partition.

Runs one `exactbo_partitioning` call (the first BO iteration's search, no
max_target_boxes cap) and appends one row per partition to
results/partition_counts.csv, labelled with the configuration.

    python run_partitioning.py --acquisition ei --bound-method interval --max-partitions 6
    python run_partitioning.py --max-partitions 9         # logEI + AutoBound (defaults)

Memory: the box arrays and their split grow with the number of targets. On a
121 GB DGX Spark, the uncapped 10-d run with acquisition=ei fills memory while
splitting into partition 9 (~151M boxes); stay at --max-partitions <= 9 there.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from common import make_gp

from tamubo.exactbo import exactbo_partitioning

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="problem10d")
    ap.add_argument("--n0", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon-ei", type=float, default=0.1)
    ap.add_argument("--epsilon-x", type=float, default=1e-5)
    ap.add_argument("--max-partitions", type=int, default=8)
    ap.add_argument("--acquisition", default="logei", choices=["logei", "ei"])
    ap.add_argument("--bound-method", default="autobound", choices=["autobound", "interval"])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--label", default=None, help="row label (default: acquisition/bound-method)")
    ap.add_argument("--out", default=str(RESULTS / "partition_counts.csv"))
    args = ap.parse_args()

    gp, X, bounds = make_gp(args.problem, args.n0, args.seed)
    d = X.shape[1]
    t = time.perf_counter()
    result = exactbo_partitioning(
        X,
        bounds,
        np.full(d, args.epsilon_x),
        args.epsilon_ei,
        gp,
        args.max_partitions,
        backend=args.backend,
        box_sampling="lhs",  # the results in README.md use 1024 LHS points per box
        acquisition=args.acquisition,
        bound_method=args.bound_method,
        validation=False,
        logMask=True,
    )
    elapsed = time.perf_counter() - t
    label = args.label or f"{args.acquisition}/{args.bound_method}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    fields = ["label", "problem", "n0", "seed", "epsilon_ei", "partition", "n_boxes", "n_active", "n_target",
              "max_acq_hi", "incumbent", "time_s"]
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for p, row in enumerate(result.log["partitions"]):
            writer.writerow(dict(
                label=label, problem=args.problem, n0=args.n0, seed=args.seed, epsilon_ei=args.epsilon_ei,
                partition=p, n_boxes=row["n_boxes"], n_active=row["n_active"], n_target=row["n_target"],
                max_acq_hi=row["max_ei_hi"], incumbent=row["best"], time_s=round(row["time"], 2),
            ))
            print(f"partition {p}: boxes {row['n_boxes']}, active {row['n_active']}, target {row['n_target']}, "
                  f"incumbent {row['best']:.4f}, t={row['time']:.1f}s")
    print(f"{label}: {elapsed:.1f}s, EI max (scaled) {result.log['ei_max_scaled']:.4f}. Appended to {out}")


if __name__ == "__main__":
    main()
