"""
Subsystem 1: DIRECT partitioning with widths measured in length scales.

DIRECT trisects a box along its widest dimensions: the two outer thirds become new
boxes and the middle third is split along the next dimension; the final middle box
comes last. A box split along m dimensions has 2m + 1 children.

ExactBO measures widths in units of the GP length scales (width / length_scale) and
splits only the dimensions within `split_ratio` of the widest (default 1/3), skipping
dimensions already narrower than epsilon_X. The bounds depend on width / length_scale,
so splitting a direction where the GP is nearly flat multiplies boxes and tightens
nothing.

Run: python development/exactbo_subsystems/s1_partition.py [--backend numpy]
"""

import argparse

import numpy as np

from common import add_common_args, best_time, get_module, get_xp, random_boxes, to_numpy


def trisect(L, U, scale, split_ratio=1 / 3, min_width=None):
    """
    Children of every box (L, U of shape (n, d)).

    Returns (child_L, child_U, counts): children of box 0 first (outer thirds of its
    widest split dimension, then the next widest, ..., center box last), then box 1, ...
    counts[i] = 2 m_i + 1 children of box i.
    """
    xp = get_module(L)
    n, d = L.shape
    width = U - L
    scaled = width / scale

    # Dimensions to split: within split_ratio of the widest candidate, where candidates
    # are at least min_width wide (all dimensions if none is).
    candidate = xp.ones((n, d), dtype=bool) if min_width is None else width >= min_width
    candidate |= ~candidate.any(axis=1, keepdims=True)
    widest = xp.where(candidate, scaled, -xp.inf).max(axis=1, keepdims=True)
    split = candidate & (scaled >= split_ratio * widest * (1 - 1e-9))  # rtol: ties after many thirds

    # Widest first; ties go to the higher index (stable argsort, reversed), as in tamubo.
    order = xp.argsort(scaled, axis=1, kind="stable")[:, ::-1]

    # Slot 2j, 2j + 1: outer thirds along the j-th widest dimension; slot 2d: center.
    slots_L = xp.repeat(L[:, None, :], 2 * d + 1, axis=1)
    slots_U = xp.repeat(U[:, None, :], 2 * d + 1, axis=1)
    kept = xp.zeros((n, 2 * d + 1), dtype=bool)
    kept[:, 2 * d] = True
    rows = xp.arange(n)
    center_L, center_U = L.copy(), U.copy()
    for j in range(d):
        dim = order[:, j]
        s = split[rows, dim]                      # is box i split along this dimension?
        lo = L[rows, dim] + width[rows, dim] / 3  # inner cut points
        hi = U[rows, dim] - width[rows, dim] / 3
        # Both outer thirds have the current center box's extent in every other dimension.
        slots_L[:, 2 * j], slots_U[:, 2 * j] = center_L, center_U
        slots_L[:, 2 * j + 1], slots_U[:, 2 * j + 1] = center_L, center_U
        slots_U[rows, 2 * j, dim] = lo            # lower third
        slots_L[rows, 2 * j + 1, dim] = hi        # upper third
        kept[:, 2 * j] = kept[:, 2 * j + 1] = s
        # The center box shrinks to the middle third along this dimension.
        center_L[rows, dim] = xp.where(s, lo, center_L[rows, dim])
        center_U[rows, dim] = xp.where(s, hi, center_U[rows, dim])
    slots_L[:, 2 * d], slots_U[:, 2 * d] = center_L, center_U
    return slots_L[kept], slots_U[kept], kept.sum(axis=1)


def check(xp, d, seed):
    print("== check ==")
    # Anisotropic GP: length scales 0.1 ... 2, so with split_ratio = 1/3 only the
    # first dimensions of the unit box are split.
    length_scale = xp.asarray(np.geomspace(0.1, 2.0, d))
    L0, U0 = xp.zeros((1, d)), xp.ones((1, d))
    for ratio, scale, name in [(0.0, xp.ones(d), "every dimension (2d+1 children)"),
                               (1.0, xp.ones(d), "DIRECT longest side, domain units"),
                               (1 / 3, length_scale, "ExactBO: within 1/3 of widest, length-scale units")]:
        cL, cU, counts = trisect(L0, U0, scale, ratio)
        print(f"  unit box, {name}: {int(counts[0])} children")

    # Three generations of splits from the unit box, then check the partition properties.
    L, U = L0, U0
    eps_X = xp.full(d, 1e-2)
    for _ in range(3):
        L, U, counts = trisect(L, U, length_scale, 1 / 3, eps_X)
    vol = float(xp.prod(U - L, axis=1).sum())
    inside = bool(xp.all(L >= 0) and xp.all(U <= 1))
    # No overlap: random points fall in exactly one child (boundaries have measure 0).
    pts = xp.asarray(np.random.default_rng(seed).random((2000, d)))
    hits = ((pts[:, None, :] >= L) & (pts[:, None, :] < U)).all(axis=2).sum(axis=1)
    print(f"  3 generations: {L.shape[0]} boxes, total volume {vol:.12f} (1), inside domain: {inside}, "
          f"points in exactly one box: {bool(xp.all(hits == 1))}")

    try:
        from tamubo.exactbo.partition import split_boxes
    except ImportError:
        print("  tamubo not importable, skipping comparison")
        return
    Lr, Ur = random_boxes(xp, 500, d, seed)
    for ratio in (0.0, 1.0, 1 / 3):
        mine = trisect(Lr, Ur, length_scale, ratio, eps_X if ratio else None)
        ref = split_boxes(Lr, Ur, xp.ones(500, dtype=bool), length_scale, 500, d, keep_inactive=False,
                          split_ratio=ratio, min_width=eps_X if ratio else None, return_counts=True,
                          backend="numpy" if xp is np else "cupy")
        same = all(np.array_equal(to_numpy(a), to_numpy(b)) for a, b in zip(mine, ref))
        print(f"  split_ratio={ratio:.3f}: identical to tamubo split_boxes: {same}")


def bench(xp, n, d, seed):
    print("== benchmark ==")
    length_scale = xp.asarray(np.geomspace(0.1, 2.0, d))
    L, U = random_boxes(xp, n, d, seed)
    for ratio in (0.0, 1 / 3):
        n_children = int(trisect(L, U, length_scale, ratio)[2].sum())
        t = best_time(lambda: trisect(L, U, length_scale, ratio), xp)
        print(f"  split_ratio={ratio:.3f}: {n} parents -> {n_children} children in {t * 1e3:.2f} ms "
              f"({n_children / t / 1e6:.1f} M children/s)")


if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser(description=__doc__), n=2**16).parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} n={args.n} d={args.d}")
    check(xp, args.d, args.seed)
    bench(xp, args.n, args.d, args.seed)
