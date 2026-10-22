#!/usr/bin/env python3
"""Benchmark AutoIA's cheap split score against the reference lookahead oracle.

This is a controller-quality audit, not a soundness proof.  Both controllers
certify every retained child independently; the benchmark measures whether the
cheap score chooses similar directions at much lower decision cost.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from autoia import Interval, RangePolicy, SubdivisionPolicy, autoia_adaptive_subdivide
from autoia.controller.execution import autoia_single
from autoia.controller.subdivision import (
    _AdaptiveBranch,
    _branch_controller_metrics,
    _direction_ambiguity_scores,
    _evaluate_subdivision_candidate,
)
from autoia.controller.split_scoring import (
    score_split_direction,
    cheap_score_key,
    cheap_score_accept,
)


def make_random_cubic_model(terms):
    def model(p, terms=terms):
        z = 0.0
        for c, alpha in terms:
            t = c
            for i, power in enumerate(alpha):
                if power:
                    t = t * (p[i] ** power)
            z = z + t
        return z
    return model


def random_terms(rng, m):
    terms = []
    for i in range(m):
        for degree, scale in ((1, 1.0), (2, 1.2), (3, 0.7)):
            alpha = [0] * m
            alpha[i] = degree
            terms.append((rng.uniform(-scale, scale), tuple(alpha)))
    for _ in range(2 * m):
        i, j = rng.sample(range(m), 2)
        alpha = [0] * m
        alpha[i] = 1
        alpha[j] = 1
        terms.append((rng.uniform(-0.8, 0.8), tuple(alpha)))
    return terms


def oracle_key(p):
    return (
        p["unresolved_gain"],
        p["structural_gain"],
        p["relative_width_gain"],
        p["ambiguity_score"],
    )


def oracle_accept(p, policy):
    return bool(
        p["unresolved_gain"] > 0
        or p["structural_gain"] >= policy.min_structural_gain
        or p["relative_width_gain"] >= policy.min_relative_width_gain
    )


def random_parent_audit(target_cases=40, seed=1234):
    rng = random.Random(seed)
    policy = SubdivisionPolicy(max_probe_directions=2, split_method="cheap")
    rp = RangePolicy(bernstein_max_coefficients=8)

    accepted_cases = 0
    exact_match = 0
    top2_match = 0
    accept_match = 0
    rank_sum = 0.0
    cheap_score_time = 0.0
    lookahead_probe_time = 0.0
    attempts = 0

    while accepted_cases < target_cases and attempts < 20 * target_cases:
        attempts += 1
        m = rng.choice((3, 4))
        model = make_random_cubic_model(random_terms(rng, m))
        box = [Interval(-0.7, 0.7) for _ in range(m)]
        try:
            root = autoia_single(model, box, order=3, range_policy=rp)
        except Exception:
            continue
        metrics = _branch_controller_metrics(root)
        if not metrics["needs_subdivision"] or len(metrics["unresolved_variables"]) < 2:
            continue

        ambiguity = _direction_ambiguity_scores(root, box)
        candidates = list(metrics["unresolved_variables"])
        candidates.sort(key=lambda i: ambiguity[i], reverse=True)
        candidates = candidates[:policy.max_probe_directions]
        if len(candidates) < 2:
            continue

        t0 = time.perf_counter()
        scores = [
            score_split_direction(root, box, d, float(ambiguity[d]))
            for d in candidates
        ]
        cheap_score_time += time.perf_counter() - t0
        cheap = max(scores, key=cheap_score_key)

        branch = _AdaptiveBranch(box, root, 0, 0)
        probes = []
        t0 = time.perf_counter()
        for direction in candidates:
            try:
                probe = _evaluate_subdivision_candidate(
                    model, branch, direction, order=3,
                    backend=root.objects[0].rep.backend,
                    representation="polynomial", range_policy=rp,
                )
                probe["ambiguity_score"] = float(ambiguity[direction])
                probes.append(probe)
            except Exception:
                pass
        lookahead_probe_time += time.perf_counter() - t0
        if len(probes) != len(candidates):
            continue

        oracle = max(probes, key=oracle_key)
        ordered = sorted(probes, key=oracle_key, reverse=True)
        oracle_dirs = [p["direction"] for p in ordered]
        rank = oracle_dirs.index(cheap["direction"]) + 1

        accepted_cases += 1
        exact_match += int(cheap["direction"] == oracle["direction"])
        top2_match += int(rank <= 2)
        rank_sum += rank
        accept_match += int(
            cheap_score_accept(cheap, policy) == oracle_accept(oracle, policy)
        )

    if accepted_cases == 0:
        raise RuntimeError("No eligible random controller cases were generated.")

    return {
        "cases": accepted_cases,
        "exact_direction_agreement": exact_match / accepted_cases,
        "top2_direction_agreement": top2_match / accepted_cases,
        "accept_stop_agreement": accept_match / accepted_cases,
        "mean_oracle_rank": rank_sum / accepted_cases,
        "cheap_score_time_s": cheap_score_time,
        "lookahead_probe_time_s": lookahead_probe_time,
        "decision_cost_speedup": lookahead_probe_time / max(cheap_score_time, 1e-15),
    }


def hidden_monotone_model(p):
    z = 0.0
    for x in p:
        z = z + x + x*x + (x**3)/3.0
    return z


def end_to_end_hidden(m):
    box = [Interval(-0.6, 0.6) for _ in range(m)]
    rp = RangePolicy(bernstein_max_coefficients=256)
    kwargs = dict(
        max_branches=max(1, 2 ** (m - 3)),
        max_depth=6,
        max_probe_directions=2,
        min_structural_gain=0.01,
        min_relative_width_gain=1.0e-4,
    )
    results = {}
    for method in ("cheap", "lookahead"):
        t0 = time.perf_counter()
        r = autoia_adaptive_subdivide(
            hidden_monotone_model, box, order=3, range_policy=rp,
            subdivision_policy=SubdivisionPolicy(**kwargs, split_method=method),
        )
        dt = time.perf_counter() - t0
        results[method] = {
            "time_s": dt,
            "width": float(r.width[0]),
            "branches": len(r.branch_results),
        }
    results["speedup"] = (
        results["lookahead"]["time_s"] / max(results["cheap"]["time_s"], 1e-15)
    )
    return results


def stationary_model(p):
    x = p[0]
    return x*x + 0.5*(x**3)


def stationary_check():
    box = [Interval(-1.0, 1.0)]
    rp = RangePolicy(bernstein_max_coefficients=2)
    kwargs = dict(
        max_branches=8, max_depth=8, max_probe_directions=1,
        min_structural_gain=0.05, min_relative_width_gain=1.0e-6,
    )
    out = {}
    for method in ("cheap", "lookahead"):
        r = autoia_adaptive_subdivide(
            stationary_model, box, order=3, range_policy=rp,
            subdivision_policy=SubdivisionPolicy(**kwargs, split_method=method),
        )
        out[method] = {
            "lower": float(r.lower[0]),
            "upper": float(r.upper[0]),
            "width": float(r.width[0]),
            "branches": len(r.branch_results),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random_stats = random_parent_audit(args.cases, args.seed)
    h5 = end_to_end_hidden(5)
    h6 = end_to_end_hidden(6)
    stat = stationary_check()

    print(f"Random eligible parent cases: {random_stats['cases']}")
    print(f"Exact reference-direction agreement: {100*random_stats['exact_direction_agreement']:.1f}%")
    print(f"Reference top-2 agreement: {100*random_stats['top2_direction_agreement']:.1f}%")
    print(f"Accept/stop agreement: {100*random_stats['accept_stop_agreement']:.1f}%")
    print(f"Mean reference rank of cheap choice: {random_stats['mean_oracle_rank']:.3f}")
    print(f"Cheap score time: {random_stats['cheap_score_time_s']:.6f} s")
    print(f"Reference probe time: {random_stats['lookahead_probe_time_s']:.6f} s")
    print(f"Direction-decision cost speedup: {random_stats['decision_cost_speedup']:.1f}x")
    for m, result in ((5, h5), (6, h6)):
        print(
            f"Hidden-monotone m={m}: cheap={result['cheap']['time_s']:.4f}s, "
            f"lookahead={result['lookahead']['time_s']:.4f}s, "
            f"speedup={result['speedup']:.2f}x, "
            f"widths=({result['cheap']['width']:.12g}, {result['lookahead']['width']:.12g})"
        )
    print(
        "Stationary 8-branch widths: "
        f"cheap={stat['cheap']['width']:.12g}, "
        f"lookahead={stat['lookahead']['width']:.12g}"
    )


if __name__ == "__main__":
    main()
