#!/usr/bin/env python3
"""Independent sampled audit for the Step-7b generator-aware linear solve.

The first half constructs synthetic shared-generator matrix/vector families
A(eps), b(eps) directly and samples eps itself.  This audits the declared
auxiliary-generator domain independently of the public engineering executor.

The second half exercises provenance through the public executor and compares a
canonical solve-through cancellation with the dependency-blind polynomial path.
Sampling is diagnostic evidence only; rigor comes from the residual/contraction
certificate.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoia import autoia, Interval
from autoia.enclosures import SharedGeneratorPolynomialEnclosure
from autoia.semantic.linear_solve import generator_semantic_solve
from autoia.trusted import Interval as TrustedInterval, RigorousBinary64Backend


def direct_generator_family_audit(families: int, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    backend = RigorousBinary64Backend()
    h = np.array([0.0])
    proto = SharedGeneratorPolynomialEnclosure.constant(
        0.0, h, 0, backend, "portfolio"
    )
    checks = 0
    max_d = 0.0
    preserved = 0

    for fam in range(families):
        A0 = np.array([
            [2.2 + 0.5 * rng.random(), rng.uniform(-0.15, 0.15)],
            [rng.uniform(-0.12, 0.12), 1.8 + 0.5 * rng.random()],
        ])
        Ag1 = rng.uniform(-0.06, 0.06, size=(2, 2))
        Ag2 = rng.uniform(-0.06, 0.06, size=(2, 2))
        b0 = rng.uniform(-0.7, 0.7, size=2)
        bg1 = rng.uniform(-0.15, 0.15, size=2)
        bg2 = rng.uniform(-0.15, 0.15, size=2)
        gids = (10000 + 2 * fam, 10001 + 2 * fam)

        def rep(c, g1=0.0, g2=0.0):
            gens = {}
            if g1 != 0.0:
                gens[gids[0]] = float(g1)
            if g2 != 0.0:
                gens[gids[1]] = float(g2)
            return SharedGeneratorPolynomialEnclosure(
                {(0,): float(c)}, TrustedInterval(0.0, 0.0), h, 0,
                backend, "portfolio", generators=gens, lift_remainder=False,
            )

        A = np.empty((2, 2), dtype=object)
        b = np.empty(2, dtype=object)
        for i in range(2):
            b[i] = rep(b0[i], bg1[i], bg2[i])
            for j in range(2):
                A[i, j] = rep(A0[i, j], Ag1[i, j], Ag2[i, j])

        out, cert = generator_semantic_solve(A, b, proto, np.linalg.solve)
        ranges = [v.range()[0] for v in out]
        if set(cert["preserved_input_generator_ids"]) == set(gids):
            preserved += 1
        max_d = max(max_d, float(np.max(cert["d_componentwise"])))

        for _ in range(samples):
            e1, e2 = rng.uniform(-1.0, 1.0, size=2)
            Atrue = A0 + Ag1 * e1 + Ag2 * e2
            btrue = b0 + bg1 * e1 + bg2 * e2
            truth = np.linalg.solve(Atrue, btrue)
            for j in range(2):
                if not (ranges[j].lo <= truth[j] <= ranges[j].hi):
                    raise AssertionError(
                        f"containment failure family={fam}, eps=({e1},{e2}), "
                        f"component={j}, truth={truth[j]}, range={ranges[j]}"
                    )
                checks += 1

    return checks, preserved, max_d


def public_provenance_check():
    def model(p):
        x = p[0]
        y = x * x
        A = np.zeros((1, 1))
        b = np.zeros(1)
        A[0, 0] = 1.0
        b[0] = y
        u = np.linalg.solve(A, b)
        return u[0] - y

    gen = autoia(model, [Interval(-1.0, 1.0)], order=1,
                 representation="generator")
    poly = autoia(model, [Interval(-1.0, 1.0)], order=1,
                  representation="polynomial")
    return float(gen.width[0]), float(poly.width[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    checks, preserved, max_d = direct_generator_family_audit(
        args.families, args.samples, args.seed
    )
    gen_width, poly_width = public_provenance_check()

    print(f"Generator-aware solve auxiliary-domain checks passed: {checks}")
    print(f"Random generator matrix families: {args.families}")
    print(f"Samples per family: {args.samples}")
    print(f"Families preserving all incoming generator ids: {preserved}/{args.families}")
    print(f"Maximum componentwise residual correction observed: {max_d:.6e}")
    print(f"Canonical solve-through cancellation width (generator): {gen_width:.6e}")
    print(f"Canonical solve-through cancellation width (polynomial): {poly_width:.6e}")


if __name__ == "__main__":
    main()
