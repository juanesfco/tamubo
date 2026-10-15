"""
Subsystem 7: the partition loop that ties subsystems 1-6 together (sampling, incumbent,
pruning, termination). A simplified version of tamubo.exactbo.exactbo_partitioning.

Each partition:
  1. bound:   upper bound of log EI over every frontier box (subsystems 2-6);
  2. sample:  log EI at every box center; the best sampled value so far is the
              incumbent (an attained value, so it only grows);
  3. prune:   a box whose upper bound is <= incumbent + eps_ei cannot beat the incumbent
              by more than eps_ei, now or later, and is dropped for good;
  4. stop:    when no box is left to refine and the box of the best sample is narrower
              than eps_X in every dimension: then max log EI <= incumbent + eps_ei over
              the whole domain (certified up to eps_X / eps_ei);
  5. split:   the remaining boxes (plus the best box) become the parents of the next
              frontier (subsystem 1, widths in length scales).

Simplifications vs tamubo: every frontier box is sampled (tamubo samples only those
near the top bounds, then the active ones), children are stored rather than rebuilt from
their parents in chunks, and everything runs on one device.

The check compares the incumbent with log EI on a dense grid (d <= 3) or on many random
points: none may exceed incumbent + eps_ei.

Run: python development/exactbo_subsystems/s7_exact_search.py [--d 2] [--no-autobound]
"""

import argparse
import time

import numpy as np

from common import add_common_args, get_xp, make_gp, sync
from s1_partition import trisect
from s4_mu_bounds import mu_bounds_autobound
from s5_sigma_bounds import qs_bounds_autobound
from s6_logei_bounds import box_logei_bounds, log_ei

BOUNDS_CHUNK = 8192  # boxes per bound call (caps memory of the (n, N, d) temporaries)


def exact_search(gp, eps_X, eps_ei, autobound=True, max_partitions=200, verbose=True):
    xp, d = gp.xp, gp.X.shape[1]
    ab = (mu_bounds_autobound(gp), qs_bounds_autobound(gp)) if autobound else (None, None)
    L, U = xp.zeros((1, d)), xp.ones((1, d))
    best_x, incumbent = None, -np.inf
    times = {"bound": 0.0, "sample": 0.0, "split": 0.0}

    def timed(key, fn):
        sync(xp)
        start = time.perf_counter()
        out = fn()
        sync(xp)
        times[key] += time.perf_counter() - start
        return out

    for p in range(max_partitions):
        ub = timed("bound", lambda: xp.concatenate(
            [box_logei_bounds(gp, L[i:i + BOUNDS_CHUNK], U[i:i + BOUNDS_CHUNK], *ab)[1]
             for i in range(0, L.shape[0], BOUNDS_CHUNK)]))

        centers = 0.5 * (L + U)
        sampled = timed("sample", lambda: log_ei(*gp.posterior(centers), gp.y_min))
        b = int(xp.argmax(sampled))
        if float(sampled[b]) > incumbent:
            best_x, incumbent = centers[b], float(sampled[b])

        target = ub > incumbent + eps_ei
        n_target = int(target.sum())
        if verbose:
            print(f"  partition {p:3d}: {L.shape[0]:8d} boxes, max bound {float(ub.max()):9.4f}, "
                  f"incumbent {incumbent:9.4f}, to refine {n_target}")
        if n_target == 0 and bool(xp.all(U[b] - L[b] < eps_X)):
            return best_x, incumbent, True, times
        target[b] = True  # always refine the box of the best sample

        L, U, _ = timed("split", lambda: trisect(L[target], U[target], gp.length_scale, 1 / 3, eps_X))
    return best_x, incumbent, False, times


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser(description=__doc__), N=16, d=2)
    parser.add_argument("--eps-x", type=float, default=1e-3)
    parser.add_argument("--eps-ei", type=float, default=0.1, help="tolerance on log EI (0.1 ~ 10.5%% of EI)")
    parser.add_argument("--no-autobound", action="store_true")
    args = parser.parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} N={args.N} d={args.d} eps_X={args.eps_x} eps_ei={args.eps_ei} "
          f"autobound={not args.no_autobound}")
    gp = make_gp(xp, args.d, args.N, args.seed)

    start = time.perf_counter()
    x, inc, converged, times = exact_search(gp, xp.full(args.d, args.eps_x), args.eps_ei, not args.no_autobound)
    total = time.perf_counter() - start
    print(f"converged={converged} x={np.round(np.asarray(x.tolist()), 4)} log EI={inc:.6f} in {total:.2f} s "
          f"(bound {times['bound']:.2f} s, sample {times['sample']:.2f} s, split {times['split']:.2f} s)")

    # Certificate check: no point of the domain may beat the incumbent by more than eps_ei.
    if args.d <= 3:
        m = {1: 200001, 2: 1001, 3: 101}[args.d]
        g = np.linspace(0, 1, m)
        pts = np.stack(np.meshgrid(*[g] * args.d, indexing="ij"), axis=-1).reshape(-1, args.d)
    else:
        pts = np.random.default_rng(args.seed).random((10**6, args.d))
    pts = xp.asarray(pts)
    vals = xp.concatenate([log_ei(*gp.posterior(pts[i:i + 65536]), gp.y_min) for i in range(0, len(pts), 65536)])
    best = float(vals.max())
    print(f"check: max log EI over {len(pts)} {'grid' if args.d <= 3 else 'random'} points = {best:.6f}, "
          f"incumbent + eps_ei = {inc + args.eps_ei:.6f} -> {'OK' if best <= inc + args.eps_ei else 'VIOLATED'}")
