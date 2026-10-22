"""Independent Step-20a Phase-2 audit for automatic configuration logic.

This is intentionally *not* a trusted-kernel arithmetic audit.  It checks the
set-theoretic controller contract and the two generic escalation mechanisms:

* the returned numeric certificate is exactly the componentwise intersection
  of all successful independently rigorous attempts;
* it is never wider than the baseline attempt;
* deterministic dense point samples lie inside the returned enclosure (sanity
  check only, never proof);
* higher-order and shared-generator retries are triggered by the documented
  structural mechanisms on isolation cases.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from autoia import autoia, Interval, ConfigurationPolicy, __version__


def models():
    return [
        ("exp_affine", lambda p: np.exp(0.4*p[0]) + 0.2*p[0], lambda x: np.exp(0.4*x)+0.2*x),
        ("smooth_poly", lambda p: p[0]**4 - 0.7*p[0]**2 + 0.1*p[0], lambda x: x**4-0.7*x**2+0.1*x),
        ("positive_recip", lambda p: 1.0/(2.0 + 0.3*p[0]), lambda x: 1.0/(2.0+0.3*x)),
        ("sqrt_shift", lambda p: np.sqrt(2.0 + 0.2*p[0]), lambda x: np.sqrt(2.0+0.2*x)),
    ]


def main():
    rng = np.random.default_rng(20260830)
    checks = 0
    records = []
    policy = ConfigurationPolicy(mode="auto")

    for name, model, numeric in models():
        for _ in range(8):
            c = float(rng.uniform(-0.7, 0.7))
            h = float(rng.uniform(0.05, 0.25))
            lo, hi = c-h, c+h
            baseline = autoia(model, [Interval(lo, hi)], order=4)
            result = autoia(model, [Interval(lo, hi)], configuration_policy=policy)
            attempts = result.attempt_results

            # Exact set-theoretic intersection of successful attempts.
            expected_lo = max(float(a.lower[0]) for a in attempts)
            expected_hi = min(float(a.upper[0]) for a in attempts)
            assert float(result.lower[0]) == expected_lo
            assert float(result.upper[0]) == expected_hi
            checks += 2

            # The intersection cannot be wider than any contributing attempt.
            for attempt in attempts:
                assert float(result.lower[0]) >= float(attempt.lower[0])
                assert float(result.upper[0]) <= float(attempt.upper[0])
                checks += 2

            # In particular it cannot be wider than the fixed baseline.
            assert float(result.lower[0]) >= float(baseline.lower[0])
            assert float(result.upper[0]) <= float(baseline.upper[0])
            checks += 2

            assert bool(result.machine_rigorous)
            assert all(bool(a.machine_rigorous) for a in attempts)
            checks += 2

            # Point sampling is a compatibility sanity check, not the proof.
            xs = np.linspace(lo, hi, 101)
            vals = np.asarray(numeric(xs), dtype=float)
            assert float(result.lower[0]) <= float(vals.min())
            assert float(result.upper[0]) >= float(vals.max())
            checks += 2

            records.append({
                "model": name,
                "box": [lo, hi],
                "attempts": len(attempts),
                "lower": float(result.lower[0]),
                "upper": float(result.upper[0]),
            })

    # Mechanism isolation 1: remainder-dominated nonlinear truncation.
    def order_case(p):
        return np.exp(3.0*p[0])

    order_result = autoia(
        order_case, [Interval(-1.0, 1.0)],
        configuration_policy=ConfigurationPolicy(mode="auto"),
    )
    order_attempts = order_result.configuration_diagnostics["attempts"]
    assert any(a.get("reason") == "remainder-dominates-retained-variation" for a in order_attempts)
    assert any(a.get("order") == 5 for a in order_attempts)
    assert order_result.width[0] <= order_result.attempt_results[0].width[0]
    checks += 3

    # Mechanism isolation 2: one approximation is reused and recombines.
    def generator_case(p):
        y = np.exp(p[0])
        return y-y

    generator_result = autoia(
        generator_case, [Interval(-1.0, 1.0)],
        configuration_policy=ConfigurationPolicy(mode="auto"),
    )
    gen_attempts = generator_result.configuration_diagnostics["attempts"]
    assert any(a.get("reason") == "shared-error-recombination" for a in gen_attempts)
    assert any(a.get("representation") == "generator" for a in gen_attempts)
    assert float(generator_result.lower[0]) == 0.0
    assert float(generator_result.upper[0]) == 0.0
    checks += 4

    # Phase 3 public-API semantics: zero-knob is automatic; explicit order is fixed.
    novice = autoia(order_case, [Interval(-0.5, 0.5)])
    fixed = autoia(order_case, [Interval(-0.5, 0.5)], order=4)
    assert getattr(novice, "automatic_configuration", False) is True
    assert not hasattr(fixed, "automatic_configuration")
    assert novice.configuration_policy.mode == "auto"
    checks += 3

    out = {
        "autoia_version": __version__,
        "seed": 20260830,
        "smooth_cases": len(records),
        "mechanism_cases": 2,
        "checks": checks,
        "status": "PASS",
        "note": "Dense samples are sanity checks only; every accepted interval comes from independently rigorous AutoIA attempts.",
    }
    Path("STEP20A_CONFIGURATION_AUDIT.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
