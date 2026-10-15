"""
How tight are ExactBO's EI bounds on the boxes that made the BO loop run out of memory?

Rebuilds one BO iteration's GP from an experiment-2 log (as max_ei_probe.py),
runs ExactBO on it up to partition P on one GPU, and takes a random sample of
that partition's frontier: the children of random targets of partition P - 1.
For every sampled box it compares upper bounds on (log) EI over the box:

  rect deg 2   the current bound: interval and degree-2 AutoBound enclosures of
               mu and of Q = k^T (K + s_n^2 I)^-1 k, then EI at (mu_lo, sigma_hi)
  rect deg 3   the same with degree-3 AutoBound enclosures
  rect + CS    new: Q's lower bound also intersected with the Cauchy-Schwarz bound
               Q(x) >= (k(x)^T a)^2 / Q(x0), a = (K + s_n^2 I)^-1 k(x0), x0 the box
               center: exact at x0, and k(x)^T a is bounded like the mean
  joint deg 2  new: one degree-2 Taylor model of (mu(x), sigma(x)) in the same x.
               Over the box, (mu, sigma) lies in a 2-d zonotope (the linear terms)
               plus a box (the second-order remainder). EI = sigma h((y_min -
               mu) / sigma) is jointly convex in (mu, sigma) (the perspective of
               the convex h), so its max over the zonotope is at one of its
               2(d + 2) vertices. Intersected with rect deg 2 (min of the two).
  rect ideal   EI at the exact (min mu, max sigma) over the box, found by
               optimization: the best any bound on mu and sigma separately can do
  true max     the max of EI over the box, found by optimization

The optimization (projected Adam, several starts per box) gives values at or
below the true extremes, so "rect ideal" and "true max" are slightly
optimistic. Reports the gap of each bound to the true max (in log EI, nats),
the fraction of boxes each would prune at ExactBO's incumbent and at the true
max EI, and the time per box.

    python bound_tightness.py ../experiment2/logs/exp2-677272.out 20 --acquisition ei --partition 9 \\
        --true-max 0.042606
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np

from max_ei_probe import HERE, _build_default_gp, load_problem, parse_iteration

import cupy as cp  # noqa: E402

import tamubo.exactbo.run as run  # noqa: E402
from tamubo.exactbo.autobound_bounds import _ab, _ea, jax, jnp, taylor_mu_q_bounds  # noqa: E402
from tamubo.exactbo.bounds import (  # noqa: E402
    _log_ei_corner,
    ei_bounds,
    mu_bounds,
    rbf_k_bounds,
    sigma_bound_factors,
    sigma_bounds,
)
from tamubo.exactbo.partition import split_boxes  # noqa: E402
from tamubo.utils import to_numpy  # noqa: E402


class _Captured(Exception):
    pass


def capture_frontier(X, bounds, gp, args):
    """Run ExactBO up to partition `args.partition`; return a sample of its frontier boxes and the incumbent."""
    count_children = run.split_child_counts
    calls = [0]
    captured = {}

    def hook(bounds_L, bounds_U, *a, **kw):
        calls[0] += 1  # one GPU: call k splits the targets of partition k - 1
        if calls[0] == args.partition:
            frame = inspect.currentframe()
            while frame.f_code.co_name != "exactbo_partitioning":
                frame = frame.f_back
            captured["incumbent"] = float(frame.f_locals["best_score_incumbent"])
            captured["n_targets"] = int(bounds_L.shape[0])
            rng = np.random.default_rng(args.seed)
            pick = cp.asarray(np.sort(rng.choice(int(bounds_L.shape[0]), size=min(args.parents, int(bounds_L.shape[0])),
                                                 replace=False)))
            captured["parents"] = (bounds_L[pick].copy(), bounds_U[pick].copy())
            captured["split"] = (a[0], kw.get("split_ratio"), kw.get("min_width"))
            raise _Captured
        return count_children(bounds_L, bounds_U, *a, **kw)

    run.split_child_counts = hook
    d = X.shape[1]
    try:
        run.exactbo_partitioning(X, bounds, np.full(d, args.epsilon_x), args.epsilon_ei, gp, args.partition + 1,
                                 backend="cupy", n_gpus=1, acquisition=args.acquisition, validation=False,
                                 predict_batch_size=int(2e7), bounds_batch_size=int(2e7))
        raise RuntimeError("ExactBO converged before the requested partition.")
    except _Captured:
        pass
    finally:
        run.split_child_counts = count_children
    (pL, pU), (scale, ratio, min_width) = captured["parents"], captured["split"]
    m = int(pL.shape[0])
    L, U = split_boxes(pL, pU, cp.ones((m,), dtype=bool), scale, m, d, keep_inactive=False, split_ratio=ratio,
                       min_width=min_width, backend="cupy", validation=False)
    return L, U, captured


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("iteration", type=int)
    ap.add_argument("--problem", default="problem10d")
    ap.add_argument("--acquisition", default="logei", choices=["logei", "ei"])
    ap.add_argument("--partition", type=int, default=9, help="sample the frontier of this partition")
    ap.add_argument("--parents", type=int, default=1024, help="random targets of the previous partition to split")
    ap.add_argument("--epsilon-ei", type=float, default=0.1)
    ap.add_argument("--epsilon-x", type=float, default=1e-5)
    ap.add_argument("--true-max", type=float, default=None, help="global max of the acquisition (max_ei_probe.py)")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", default=str(HERE / "results" / "bound_tightness.jsonl"))
    args = ap.parse_args()
    use_log = args.acquisition == "logei"

    X_phys, y, _, _ = parse_iteration(args.log, args.iteration)
    bounds = np.asarray(load_problem(args.problem).bounds, dtype=float)
    X = (X_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    N, d = X.shape
    gp = _build_default_gp(d)
    gp.fit(X, y)
    params = gp.kernel_.get_params()
    sf2 = float(params["k1__k1__constant_value"])
    ls = np.asarray(params["k1__k2__length_scale"], dtype=float)
    y_mean, y_std = float(np.ravel(gp._y_train_mean)[0]), float(np.ravel(gp._y_train_std)[0])
    y_min = float(np.min(gp.y_train_))
    L_inv, lam = sigma_bound_factors(gp.L_)
    K_inv = L_inv.T @ L_inv
    alpha = np.asarray(gp.alpha_, dtype=float).ravel()

    t = time.perf_counter()
    box_L, box_U, captured = capture_frontier(X, np.array([[0.0, 1.0]] * d), gp, args)
    n = int(box_L.shape[0])
    cp.get_default_memory_pool().free_all_blocks()  # ExactBO's buffers; JAX allocates outside the pool
    print(f"partition {args.partition}: {captured['n_targets']} targets in partition {args.partition - 1}, "
          f"incumbent {captured['incumbent']:.6f}; {n} sampled frontier boxes ({time.perf_counter() - t:.0f} s)",
          flush=True)

    Xc, alpha_c, Lc, L_inv_c, ls_c = (cp.asarray(a) for a in (X, alpha, gp.L_, L_inv, ls))

    parts = {}  # degree -> (mu_lo, sig_hi) of the rect bound

    def rect(degree, cs=False):
        """
        The current bound (run.py's _ei_hi_bounds_chunked) with the given AutoBound
        degree; with ``cs``, Q's lower bound is also intersected with the
        Cauchy-Schwarz bound (``cs_q_lo``).
        """
        K_lo, K_hi = cp.empty((n, N)), cp.empty((n, N))
        for i in range(N):
            K_lo[:, i], K_hi[:, i] = rbf_k_bounds(box_L, box_U, Xc[i], n, d, sf2, ls_c, backend="cupy",
                                                  validation=False)
        mu_lo, mu_hi = mu_bounds(alpha_c, K_lo, K_hi, n, N, y_train_mean=y_mean, y_train_std=y_std,
                                 scaled_output=True, backend="cupy", validation=False)
        # Degree 3 needs ~4.6 MB per box at N ~ 50, d = 10.
        ab = taylor_mu_q_bounds(box_L, box_U, Xc, alpha_c, cp.asarray(K_inv), ls_c, sf2, xp=cp, degree=degree,
                                batch_size=8192 if degree <= 2 else 512)
        cp.maximum(mu_lo, ab[0], out=mu_lo)
        cp.minimum(mu_hi, ab[1], out=mu_hi)
        q_lo = cp.maximum(ab[2], cs_q_lo(K_lo, K_hi)) if cs else ab[2]
        sig_lo, sig_hi = sigma_bounds(K_lo, K_hi, Lc, n, N, sf2, y_train_std=y_std, scaled_output=True,
                                      backend="cupy", validation=False, L_inv=L_inv_c, lambda_max=lam,
                                      q_bounds=(q_lo, ab[3]))
        parts[(degree, cs)] = (mu_lo, sig_hi)
        return ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min, backend="cupy", validation=False, log=True)[1]

    gp_j = tuple(jnp.asarray(a) for a in (X, alpha, K_inv, ls))

    @jax.jit
    def linear_lo(bL, bU, w, Xt, l):
        """Degree-2 AutoBound lower bound of k(x)^T w over each box (one weight vector per box)."""
        def per_box(lo, hi, wb):
            def f(x):
                diff = (x - Xt) / l
                return (sf2 * jnp.exp(-0.5 * jnp.sum(diff * diff, axis=1))) @ wb

            x0 = 0.5 * (lo + hi)
            enc = _ab.taylor_bounds(f, 2)(x0, (lo, hi)).coefficients
            return _ea.enclose_enclosure(enc, (lo - x0, hi - x0), 0, jnp)[0][0]

        return jax.vmap(per_box)(bL, bU, w)

    K_inv_c = cp.asarray(K_inv)

    def cs_q_lo(K_lo, K_hi):
        """
        Q(x) = k^T K^-1 k >= (k(x)^T a)^2 / (a^T K a) for any a (Cauchy-Schwarz in the
        K^-1 inner product). With a = K^-1 k(x0), x0 the box center, it is exact at x0
        and a^T K a = Q(x0); k(x)^T a is linear in k, bounded like the mean (interval
        sign split and a degree-2 Taylor model, intersected).
        """
        x0 = 0.5 * (box_L + box_U)
        diff = (x0[:, None, :] - Xc[None, :, :]) / ls_c
        k0 = sf2 * cp.exp(-0.5 * cp.sum(diff * diff, axis=2))  # (n, N)
        a = k0 @ K_inv_c  # K^-1 symmetric
        q0 = cp.sum(k0 * a, axis=1)
        s_lo = cp.sum(K_lo * cp.maximum(a, 0.0) + K_hi * cp.minimum(a, 0.0), axis=1)
        s_ab = cp.from_dlpack(linear_lo(*(jnp.asarray(to_numpy(v)) for v in (box_L, box_U, a)), gp_j[0], gp_j[3]))
        s_lo = cp.maximum(s_lo, s_ab)
        return cp.where(q0 > 0, cp.maximum(s_lo, 0.0) ** 2 / cp.where(q0 > 0, q0, 1.0), 0.0)

    def mu_sigma(x, Xt, al, Ki, l):
        diff = (x - Xt) / l
        k = sf2 * jnp.exp(-0.5 * jnp.sum(diff * diff, axis=1))
        return k @ al, jnp.sqrt(jnp.maximum(sf2 - k @ (Ki @ k), 0.0))

    @jax.jit
    def joint_vertices(bL, bU, Xt, al, Ki, l):
        """(n, 2(d + 2), 2) vertices of the zonotope enclosing (mu, sigma) over each box."""
        def per_box(lo, hi):
            def f(x):
                diff = (x - Xt) / l
                k = sf2 * jnp.exp(-0.5 * jnp.sum(diff * diff, axis=1))
                return jnp.array([1.0, 0.0]) * (k @ al) + jnp.array([0.0, 1.0]) * jnp.sqrt(sf2 - k @ (Ki @ k))

            x0 = 0.5 * (lo + hi)
            c0, c1, c2 = _ab.taylor_bounds(f, 2)(x0, (lo, hi)).coefficients
            zero = (jnp.zeros_like(c0), jnp.zeros_like(c1), c2)
            (r_lo, r_hi), = _ea.enclose_enclosure(zero, (lo - x0, hi - x0), 0, jnp)
            center = c0 + 0.5 * (r_lo + r_hi)
            gens = jnp.concatenate([(c1 * (0.5 * (hi - lo))).T, jnp.diag(0.5 * (r_hi - r_lo))])  # (d + 2, 2)
            # Orient every generator into the upper half-plane and sort by angle: the
            # vertices are the partial sums walking around the zonotope.
            flip = (gens[:, 1] < 0) | ((gens[:, 1] == 0) & (gens[:, 0] < 0))
            gens = jnp.where(flip[:, None], -gens, gens)
            gens = gens[jnp.argsort(jnp.arctan2(gens[:, 1], gens[:, 0]))]
            start = center - jnp.sum(gens, axis=0)
            walk = start + 2.0 * jnp.cumsum(gens, axis=0)
            return jnp.concatenate([walk, 2.0 * center - walk])

        return jax.vmap(per_box)(bL, bU)

    def joint():
        verts = cp.from_dlpack(joint_vertices(jnp.asarray(to_numpy(box_L)), jnp.asarray(to_numpy(box_U)), *gp_j))
        mu, sig = verts[..., 0].ravel(), cp.maximum(verts[..., 1].ravel(), 0.0)  # EI(mu, max(sigma, 0)) is convex too
        val = _log_ei_corner(mu, sig, y_min, 1e-12, cp, "cupy").reshape(verts.shape[:2])
        val = cp.max(val, axis=1)
        return cp.where(cp.isnan(val), cp.inf, val)

    def log_h_jax(z):
        # log(phi(z) + z Phi(z)); accurate enough to locate maximizers (the values
        # are re-evaluated with ExactBO's own log EI below).
        log_2pi_half = 0.5 * jnp.log(2.0 * jnp.pi)
        zc = jnp.minimum(z, -1.0)
        log_phi_c = -0.5 * zc * zc - log_2pi_half
        mills = jnp.exp(jax.scipy.special.log_ndtr(zc) - log_phi_c)  # Phi / phi
        tail = log_phi_c + jnp.log(jnp.maximum(1.0 + zc * mills, 1e-300))
        far = log_phi_c - 2.0 * jnp.log(-zc)
        zm = jnp.maximum(z, -1.0)
        mid = jnp.log(jnp.exp(-0.5 * zm * zm - log_2pi_half) + zm * jax.scipy.special.ndtr(zm))
        return jnp.where(z > -1.0, mid, jnp.where(z < -1e3, far, tail))

    def maximize(objective):
        """Projected Adam in box coordinates t in [0, 1]^d from `starts` starts per box; best x per box."""
        rng = np.random.default_rng(args.seed + 1)
        t0 = np.concatenate([np.full((n, 1, d), 0.5), rng.random((n, args.starts - 1, d))], axis=1)
        bL, bU = jnp.asarray(to_numpy(box_L))[:, None, :], jnp.asarray(to_numpy(box_U))[:, None, :]

        def value(tt):
            return objective(bL + tt * (bU - bL))

        grad = jax.jit(jax.grad(lambda tt: jnp.sum(value(tt))))

        @jax.jit
        def steps(tt):
            def body(i, state):
                tt, m, v = state
                g = grad(tt)
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                step = 0.02 * (m / (1 - 0.9 ** (i + 1))) / (jnp.sqrt(v / (1 - 0.999 ** (i + 1))) + 1e-12)
                return jnp.clip(tt + step, 0.0, 1.0), m, v
            return jax.lax.fori_loop(0, args.steps, body, (tt, jnp.zeros_like(tt), jnp.zeros_like(tt)))[0]

        tt = steps(jnp.asarray(t0))
        vals = value(tt)
        best = jnp.argmax(vals, axis=1)
        return (bL[:, 0] + tt[jnp.arange(n), best] * (bU[:, 0] - bL[:, 0]))  # (n, d)

    def field(x, which):
        return jax.vmap(jax.vmap(lambda p: mu_sigma(p, *gp_j)[which]))(x)

    def log_ei_jax(x):
        mu, sig = field(x, 0), field(x, 1)
        sig = jnp.maximum(sig, 1e-12)
        return jnp.log(sig) + log_h_jax((y_min - mu) / sig)

    def exact_log_ei(x):
        mu, sig = jax.vmap(lambda p: mu_sigma(p, *gp_j))(x)
        return _log_ei_corner(cp.from_dlpack(mu), cp.from_dlpack(sig), y_min, 1e-12, cp, "cupy")

    results, timings = {}, {}
    for name, fn in [("rect deg 2", lambda: rect(2)), ("rect deg 3", lambda: rect(3)), ("joint deg 2", joint),
                     ("rect deg 2 + CS", lambda: rect(2, cs=True)), ("rect deg 3 + CS", lambda: rect(3, cs=True))]:
        fn()  # compile
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        results[name] = fn()
        cp.cuda.Device().synchronize()
        timings[name] = (time.perf_counter() - t) / n
        print(f"{name}: {timings[name] * 1e6:.1f} us/box", flush=True)
    joint_alone = results.pop("joint deg 2")
    print(f"joint deg 2 alone: {float(cp.mean(cp.isinf(joint_alone))):.3f} of boxes without a finite enclosure "
          f"(sigma's enclosure reaches 0)", flush=True)
    results["joint deg 2 alone"] = joint_alone
    results["joint deg 2"] = cp.minimum(joint_alone, results["rect deg 2"])

    t = time.perf_counter()
    x_mu = maximize(lambda x: -field(x, 0))
    x_sig = maximize(lambda x: field(x, 1))
    mu_min = jax.vmap(lambda p: mu_sigma(p, *gp_j)[0])(x_mu)
    sig_max = jax.vmap(lambda p: mu_sigma(p, *gp_j)[1])(x_sig)
    mu_min, sig_max = cp.from_dlpack(mu_min), cp.from_dlpack(sig_max)
    results["rect ideal"] = _log_ei_corner(mu_min, sig_max, y_min, 1e-12, cp, "cupy")
    # Which enclosure loses more: the exact range of one with the degree-2 bound of the other.
    results["mu exact, sigma deg 2"] = _log_ei_corner(mu_min, parts[(2, False)][1], y_min, 1e-12, cp, "cupy")
    results["mu deg 2, sigma exact"] = _log_ei_corner(parts[(2, False)][0], sig_max, y_min, 1e-12, cp, "cupy")
    x_ei = maximize(log_ei_jax)
    true_max = exact_log_ei(x_ei)
    # Box centers too (what ExactBO samples), in case the optimizer stalls.
    true_max = cp.maximum(true_max, exact_log_ei(jnp.asarray(to_numpy(0.5 * (box_L + box_U)))))
    results["true max"] = true_max
    print(f"optimization: {time.perf_counter() - t:.0f} s", flush=True)

    def prunes(log_bound, incumbent):
        bound = log_bound if use_log else cp.exp(log_bound)
        return float(cp.mean(bound <= incumbent + args.epsilon_ei))

    # Validity: every bound must be >= the values attained in the box (gap min >= 0),
    # and every sigma bound >= the largest sigma found.
    for key, (_, sig_hi_k) in parts.items():
        bad = int(cp.sum(sig_hi_k < sig_max * (1 - 1e-12)))
        print(f"sigma bound {key}: {bad} boxes below the attained max sigma; "
              f"min sig_hi / sig_max - 1 = {float(cp.min(sig_hi_k / sig_max - 1)):.3e}", flush=True)
    rows = []
    print(f"\n{'bound':22s} {'gap min':>9s} {'gap median':>11s} {'gap p90':>8s} {'pruned @incumbent':>18s} {'pruned @true max':>17s}")
    for name, log_bound in results.items():
        gap = to_numpy(log_bound - true_max)
        row = dict(bound=name, gap_min=float(np.min(gap)), gap_median=float(np.median(gap)),
                   gap_p90=float(np.quantile(gap, 0.9)),
                   pruned_at_incumbent=prunes(log_bound, captured["incumbent"]),
                   pruned_at_true_max=None if args.true_max is None else prunes(log_bound, args.true_max),
                   us_per_box=timings.get(name, None) and timings[name] * 1e6)
        rows.append(row)
        tm = "" if row["pruned_at_true_max"] is None else f"{row['pruned_at_true_max']:.3f}"
        print(f"{name:22s} {row['gap_min']:9.2e} {row['gap_median']:11.3f} {row['gap_p90']:8.3f} {row['pruned_at_incumbent']:18.3f} {tm:>17s}")

    summary = dict(log=str(args.log), iteration=args.iteration, acquisition=args.acquisition, partition=args.partition,
                   n_targets_prev=captured["n_targets"], incumbent=captured["incumbent"], true_max=args.true_max,
                   n_boxes=n, rows=rows)
    with Path(args.summary).open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
