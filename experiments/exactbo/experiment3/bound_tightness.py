"""
Bound tightness vs box depth on the experiment-2 GP.

For random DIRECT-like boxes at each depth, compares the bounds against the GP
posterior sampled densely inside each box (a lower estimate of the true range):
the gap is the overestimate, and every bound must contain every sample.

Methods:
  interval   per-entry kernel intervals -> mu (alpha sign split), sigma (L^{-1}
             sign split + lambda bound), EI (exact over the mu/sigma rectangle)
  autobound  interval intersected with AutoBound degree-2 Taylor enclosures of
             mu(x) and Q(x)

Writes results/bound_tightness.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm

from common import gp_arrays, make_gp, random_boxes, sampled_posterior

from tamubo.exactbo.bounds import ei_bounds, mu_bounds, rbf_k_bounds, sigma_bound_factors, sigma_bounds
from tamubo.utils import get_array_module, to_numpy

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="problem10d")
    ap.add_argument("--n0", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boxes", type=int, default=256, help="boxes per depth")
    ap.add_argument("--samples", type=int, default=2000, help="posterior samples per box")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 5, 10, 15, 20, 30, 50])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default=str(RESULTS / "bound_tightness.csv"))
    args = ap.parse_args()

    from tamubo.exactbo.autobound_bounds import taylor_mu_q_bounds

    xp = get_array_module(args.backend)
    gp, X, _ = make_gp(args.problem, args.n0, args.seed)
    P = gp_arrays(gp, X)
    N, d = P["X"].shape
    L_inv, lambda_max = sigma_bound_factors(P["L"])
    K_inv = L_inv.T @ L_inv
    rng = np.random.default_rng(args.seed)
    rows = []

    for depth in args.depths:
        n = args.boxes
        lo, hi = random_boxes(n, d, depth, rng)
        mu_s, sd_s = sampled_posterior(P, lo, hi, args.samples, rng)
        z = (P["ymin"] - mu_s) / sd_s
        ei_s = ((P["ymin"] - mu_s) * norm.cdf(z) + sd_s * norm.pdf(z)).max(1)

        bL, bU = xp.asarray(lo), xp.asarray(hi)
        K_lo = xp.empty((n, N))
        K_hi = xp.empty((n, N))
        for i in range(N):
            K_lo[:, i], K_hi[:, i] = rbf_k_bounds(bL, bU, xp.asarray(P["X"][i]), n, d, P["sf2"], xp.asarray(P["ls"]), backend=args.backend)
        mu_lo, mu_hi = mu_bounds(xp.asarray(P["alpha"]), K_lo, K_hi, n, N, scaled_output=True, backend=args.backend)
        ab_mu_lo, ab_mu_hi, ab_q_lo, ab_q_hi = taylor_mu_q_bounds(bL, bU, P["X"], P["alpha"], K_inv, P["ls"], P["sf2"], xp=xp)

        methods = {
            "interval": (mu_lo, mu_hi, None),
            "autobound": (xp.maximum(mu_lo, ab_mu_lo), xp.minimum(mu_hi, ab_mu_hi), (ab_q_lo, ab_q_hi)),
        }
        for method, (m_lo, m_hi, q_bounds) in methods.items():
            s_lo, s_hi = sigma_bounds(K_lo, K_hi, xp.asarray(P["L"]), n, N, P["sf2"], scaled_output=True,
                                      backend=args.backend, L_inv=L_inv, lambda_max=lambda_max, q_bounds=q_bounds)
            _, e_hi = ei_bounds(m_lo, m_hi, s_lo, s_hi, n, P["ymin"], backend=args.backend)
            m_lo, s_hi, e_hi = to_numpy(m_lo), to_numpy(s_hi), to_numpy(e_hi)
            sound = bool(
                np.all(m_lo <= mu_s.min(1) + 1e-9)
                and np.all(s_hi >= sd_s.max(1) - 1e-9)
                and np.all(e_hi >= ei_s - 1e-12)
            )
            row = dict(
                depth=depth,
                method=method,
                sound=sound,
                mu_range_sampled=float(np.median(mu_s.max(1) - mu_s.min(1))),
                mu_lo_gap=float(np.median(mu_s.min(1) - m_lo)),
                sigma_hi_gap=float(np.median(s_hi - sd_s.max(1))),
                ei_hi_median=float(np.median(e_hi)),
                ei_sampled_median=float(np.median(ei_s)),
                # Share of boxes whose bound would prune them against the best sampled EI (EI mode, eps 0.1).
                prunable_eps0p1=float(np.mean(e_hi <= ei_s.max() + 0.1)),
            )
            rows.append(row)
            print(f"depth {depth:3d} {method:9s} sound={sound}  mu_lo gap {row['mu_lo_gap']:.2e}  "
                  f"sigma_hi gap {row['sigma_hi_gap']:.2e}  EI_hi median {row['ei_hi_median']:.2e}  "
                  f"prunable {row['prunable_eps0p1']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
