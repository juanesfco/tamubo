"""
Subsystem 2: distance bounds between a box and the training points.

For a box [l, u] and a training point x_i, in length-scale units
(a_j = (l_j - x_ij) / ls_j, b_j = (u_j - x_ij) / ls_j):

    nearest point:  clip(x_i, l, u)        ->  dmin_j = max(a_j, -b_j, 0)
    farthest point: the farther corner     ->  dmax_j = max(|a_j|, |b_j|)
    D^2 in [sum dmin_j^2, sum dmax_j^2]

Both ends are attained (at the clipped point and at the farthest corner), so the
bounds are exact. The RBF kernel only needs squared distances, so those are returned.

Run: python development/exactbo_subsystems/s2_distance_bounds.py [--backend numpy]
"""

import argparse

from common import add_common_args, best_time, get_module, get_xp, make_gp, random_boxes, sample_in_boxes


def sq_distance_bounds(L, U, X, length_scale):
    """
    Bounds on ||(x - X_i) / ls||^2 over x in each box. L, U: (n, d), X: (N, d).
    Returns (d2_lo, d2_hi), each (n, N). Memory: a few (n, N, d) temporaries.
    """
    xp = get_module(L)
    a = (L[:, None, :] - X[None, :, :]) / length_scale  # (n, N, d)
    b = (U[:, None, :] - X[None, :, :]) / length_scale
    d_min = xp.maximum(xp.maximum(a, -b), 0.0)  # 0 when x_ij is inside [l_j, u_j]
    d_max = xp.maximum(xp.abs(a), xp.abs(b))
    return xp.sum(d_min * d_min, axis=2), xp.sum(d_max * d_max, axis=2)


def check(xp, gp, d, seed):
    print("== check ==")
    L, U = random_boxes(xp, 200, d, seed, depth=(0, 3))
    d2_lo, d2_hi = sq_distance_bounds(L, U, gp.X, gp.length_scale)

    # 1. Every sampled point of a box lies within the bounds.
    pts = sample_in_boxes(xp, L, U, 256, seed)  # (n, m, d)
    diff = (pts[:, :, None, :] - gp.X[None, None]) / gp.length_scale
    d2 = xp.sum(diff * diff, axis=3)  # (n, m, N)
    ok = bool(xp.all(d2 >= d2_lo[:, None, :] - 1e-12) and xp.all(d2 <= d2_hi[:, None, :] + 1e-12))
    print(f"  sampled distances inside the bounds: {ok}")
    slack_lo = float(xp.median(d2.min(axis=1) - d2_lo))
    slack_hi = float(xp.median(d2_hi - d2.max(axis=1)))
    print(f"  median gap to sampled min / max (sampling, not the bounds, is loose): {slack_lo:.3g} / {slack_hi:.3g}")

    # 2. Both ends are attained: the nearest point is clip(X_i, l, u), the farthest the far corner.
    nearest = xp.clip(gp.X[None], L[:, None], U[:, None])  # (n, N, d)
    far = xp.where(xp.abs(L[:, None] - gp.X[None]) > xp.abs(U[:, None] - gp.X[None]), L[:, None], U[:, None])
    sq = lambda p: xp.sum(((p - gp.X[None]) / gp.length_scale) ** 2, axis=2)
    print(f"  lower bound attained (max error {float(xp.max(xp.abs(sq(nearest) - d2_lo))):.1e}), "
          f"upper bound attained (max error {float(xp.max(xp.abs(sq(far) - d2_hi))):.1e})")


def bench(xp, gp, n, d, seed):
    print("== benchmark ==")
    L, U = random_boxes(xp, n, d, seed)
    t = best_time(lambda: sq_distance_bounds(L, U, gp.X, gp.length_scale), xp)
    N = gp.X.shape[0]
    print(f"  {n} boxes x {N} points: {t * 1e3:.2f} ms ({n * N / t / 1e9:.2f} G box-point pairs/s)")


if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} n={args.n} N={args.N} d={args.d}")
    gp = make_gp(xp, args.d, args.N, args.seed)
    check(xp, gp, args.d, args.seed)
    bench(xp, gp, args.n, args.d, args.seed)
