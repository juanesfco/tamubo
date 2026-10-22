#!/usr/bin/env python3
"""Independent boundary/predicate audit for Step 10a domain certification."""

from __future__ import annotations

import sys
from pathlib import Path
from fractions import Fraction
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia.controller.execution import rigorous_center_and_halfwidth
from autoia.enclosures import PolynomialEnclosure, ValidatedValue
from autoia.enclosures.domain import DomainRequirement, prove_domain
from autoia.trusted import Interval, RigorousBinary64Backend


def exact_ge_zero(I: Interval) -> bool:
    return Fraction.from_float(float(I.lo)) >= 0


def exact_gt_zero(I: Interval) -> bool:
    return Fraction.from_float(float(I.lo)) > 0


def make_parameter(I: Interval, backend):
    c, h = rigorous_center_and_halfwidth(I, backend)
    rep = PolynomialEnclosure.parameter(c, 0, [h], 4, backend, "termwise")
    return ValidatedValue(rep, natural_interval=I)


def main():
    rng = np.random.default_rng(2026083019)
    backend = RigorousBinary64Backend()
    universal_checks = 0
    boundary_checks = 0
    strict_checks = 0

    # Random direct physical parameters.  Whenever the prover says the
    # universal predicate is true, compare against the exact dyadic endpoint
    # order represented by the supplied binary64 box.
    for _ in range(1000):
        a, b = sorted(rng.uniform(-10.0, 10.0, size=2))
        I = Interval(float(a), float(b))
        v = make_parameter(I, backend)
        for relation, exact_truth in (("ge", exact_ge_zero(I)), ("gt", exact_gt_zero(I))):
            req = DomainRequirement("audit", relation, 0.0)
            proof = prove_domain(v, req)
            if proof.proved and not exact_truth:
                raise AssertionError(("false-positive", relation, I, proof))
            universal_checks += 1

    # The exact boundary that motivated Step 10a: the centered structural range
    # is allowed to have a tiny negative predecessor, but the independent
    # physical-box enclosure proves sqrt's nonnegative domain exactly.
    for hi in (1.0, 2.0, 4.0, 10.0, 1.0e6):
        I = Interval(0.0, hi)
        v = make_parameter(I, backend)
        structural, _ = v.rep.range("termwise")
        proof = prove_domain(v, DomainRequirement("sqrt", "ge", 0.0))
        if structural.lo >= 0.0:
            raise AssertionError(("boundary-case-did-not-exercise-structural-overreach", I, structural))
        if not proof.proved or proof.interval.lo != 0.0:
            raise AssertionError(("boundary-not-proved", I, structural, proof))
        if proof.methods_tried != ("natural-physical-box",):
            raise AssertionError(("unexpected-boundary-proof-path", proof.methods_tried))
        boundary_checks += 1

    # Strict positivity must never be inferred merely from nonnegativity at the
    # physical boundary.
    for hi in (1.0, 4.0, 100.0):
        I = Interval(0.0, hi)
        v = make_parameter(I, backend)
        proof = prove_domain(v, DomainRequirement("log", "gt", 0.0))
        if proof.proved:
            raise AssertionError(("strict-boundary-false-positive", I, proof))
        strict_checks += 1

    total = universal_checks + boundary_checks + strict_checks
    print(f"direct_affine_universal_predicates_checked={universal_checks}")
    print(f"exact_nonnegative_boundaries_checked={boundary_checks}")
    print(f"strict_zero_boundaries_rejected={strict_checks}")
    print(f"total_step10a_domain_checks={total}")
    print("status=PASS")


if __name__ == "__main__":
    main()
