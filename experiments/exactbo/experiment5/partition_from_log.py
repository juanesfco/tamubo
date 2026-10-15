"""
Run ExactBO's uncapped partition search on the GP of one logged BO iteration.

Rebuilds the GP from an experiment-2 log (as max_ei_probe.py) and runs
``exactbo_partitioning`` with the experiment-2 settings, so a single hard
iteration (e.g. EI iteration 20 or logEI iteration 14, which ran out of
memory) can be rerun with new bounds without replaying the BO loop. Appends a
JSON summary to --summary.

    python partition_from_log.py ../experiment2/logs/exp2-677272.out 20 --acquisition ei
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from max_ei_probe import HERE, _build_default_gp, load_problem, parse_iteration

from tamubo.exactbo import exactbo_partitioning  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("iteration", type=int)
    ap.add_argument("--problem", default="problem10d")
    ap.add_argument("--acquisition", default="logei", choices=["logei", "ei"])
    ap.add_argument("--epsilon-ei", type=float, default=0.1)
    ap.add_argument("--epsilon-x", type=float, default=1e-5)
    ap.add_argument("--max-partitions", type=int, default=1000)
    ap.add_argument("--n-gpus", type=int, default=None)
    ap.add_argument("--batch-size", type=float, default=2e7, help="predict_batch_size and bounds_batch_size")
    ap.add_argument("--label", default=None)
    ap.add_argument("--summary", default=str(HERE / "results" / "partition_from_log.jsonl"))
    args = ap.parse_args()

    X_phys, y, _, logged_incumbents = parse_iteration(args.log, args.iteration)
    bounds = np.asarray(load_problem(args.problem).bounds, dtype=float)
    X = (X_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    d = X.shape[1]
    gp = _build_default_gp(d)
    gp.fit(X, y)

    t = time.perf_counter()
    result = exactbo_partitioning(
        X, np.array([[0.0, 1.0]] * d), np.full(d, args.epsilon_x), args.epsilon_ei, gp, args.max_partitions,
        backend="cupy", n_gpus=args.n_gpus, acquisition=args.acquisition, validation=False, logMask=True,
        predict_batch_size=int(args.batch_size), bounds_batch_size=int(args.batch_size),
    )
    elapsed = time.perf_counter() - t
    parts = result.log["partitions"]
    for p, row in enumerate(parts):
        print(f"partition {p}: boxes {row['n_boxes']}, active {row['n_active']}, target {row['n_target']}, "
              f"max bound {row['max_ei_hi']:.6f}, incumbent {row['best']:.6f}, t={row['time']:.1f}s", flush=True)
    summary = dict(
        label=args.label or f"{args.log.name}/iteration {args.iteration}/{args.acquisition}", args=vars(args) | {
            "log": str(args.log)}, elapsed_s=round(elapsed, 2), partitions=len(parts),
        converged=bool(result.log.get("converged")), n_target=[int(r["n_target"]) for r in parts],
        n_boxes=[int(r["n_boxes"]) for r in parts], incumbent=float(parts[-1]["best"]),
        x=np.asarray(result.X, dtype=float).ravel().tolist(),
    )
    with Path(args.summary).open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"RUN {summary['label']}: {elapsed:.1f}s, partitions {len(parts)}, converged {summary['converged']}, "
          f"peak targets {max(summary['n_target'])}, incumbent {summary['incumbent']:.6f}", flush=True)


if __name__ == "__main__":
    main()
