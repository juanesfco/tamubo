"""
Where multi-GPU partitioning time goes: per sharded function (EI bounds, box
sampling), wall time, GPUs used per call, and each GPU's busy time; plus box
splitting and the remainder, which runs on the main GPU only.

    python profile_sharding.py --n-gpus 2 --acquisition ei --max-partitions 9
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiment3"))

from common import make_gp  # noqa: E402

import tamubo.exactbo.run as run_mod  # noqa: E402
from tamubo.exactbo import exactbo_partitioning  # noqa: E402

stats = defaultdict(lambda: {"calls": 0, "wall": 0.0, "boxes": 0, "gpus": Counter(), "busy": defaultdict(float)})


def _timed_run_sharded(original):
    def run_sharded(xp, fn, states, boxes_L, boxes_U, work_per_box):
        s = stats[fn.__name__]
        used = set()

        def timed_fn(state, L, U):
            t = time.perf_counter()
            out = fn(state, L, U)
            xp.cuda.Device(state.device).synchronize()
            s["busy"][state.device] += time.perf_counter() - t
            used.add(state.device)
            return out

        timed_fn.__name__ = fn.__name__
        xp.cuda.Device(states[0].device).synchronize()
        t = time.perf_counter()
        out = original(xp, timed_fn, states, boxes_L, boxes_U, work_per_box)
        xp.cuda.Device(states[0].device).synchronize()
        s["wall"] += time.perf_counter() - t
        s["calls"] += 1
        s["boxes"] += int(boxes_L.shape[0])
        s["gpus"][len(used)] += 1
        return out

    return run_sharded


def _timed_split(original):
    def split_boxes(*args, **kwargs):
        import cupy as cp

        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        out = original(*args, **kwargs)
        cp.cuda.Device().synchronize()
        s = stats["split_boxes (main GPU)"]
        s["wall"] += time.perf_counter() - t
        s["calls"] += 1
        return out

    return split_boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-gpus", type=int, default=None)
    ap.add_argument("--acquisition", default="ei", choices=["logei", "ei"])
    ap.add_argument("--max-partitions", type=int, default=9)
    args = ap.parse_args()

    run_mod._run_sharded = _timed_run_sharded(run_mod._run_sharded)
    run_mod.split_boxes = _timed_split(run_mod.split_boxes)

    gp, X, bounds = make_gp("problem10d", 32, 0)
    t = time.perf_counter()
    result = exactbo_partitioning(
        X, bounds, np.full(X.shape[1], 1e-5), 0.1, gp, args.max_partitions,
        backend="cupy", n_gpus=args.n_gpus, box_sampling="lhs", acquisition=args.acquisition,
        validation=False, logMask=True,
    )
    total = time.perf_counter() - t

    print(f"PROFILE n_gpus={args.n_gpus} {args.acquisition} partitions={len(result.log['partitions'])} "
          f"total={total:.1f}s incumbent={result.log['partitions'][-1]['best']:.6f}")
    accounted = 0.0
    for name, s in stats.items():
        accounted += s["wall"]
        busy = ", ".join(f"GPU{d}: {b:.1f}s" for d, b in sorted(s["busy"].items()))
        gpus = ", ".join(f"{k} GPU(s) x{v}" for k, v in sorted(s["gpus"].items()))
        print(f"  {name}: calls {s['calls']}, wall {s['wall']:.1f}s, boxes {s['boxes']:,}"
              + (f", calls by GPUs used [{gpus}], busy [{busy}]" if s["busy"] else ""))
    print(f"  everything else (main GPU, host): {total - accounted:.1f}s")


if __name__ == "__main__":
    main()
