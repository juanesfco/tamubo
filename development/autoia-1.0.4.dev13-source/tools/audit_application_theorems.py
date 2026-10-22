#!/usr/bin/env python3
"""Independent audit of the Step-20c public application-theorem boundary."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]

from autoia.application import ApplicationBox, CertificationContext
from validation.downstream_gp_application.paper_states import load_state, representative_box, point_ei
from validation.downstream_gp_application.public_application_bridge import rigorous_downstream_gp_application_public_upper_one_box
from validation.downstream_gp_application.rigorous_bridge import rigorous_downstream_gp_application_ei_bounds_one_box


def audit_import_boundary():
    path = ROOT / "validation" / "downstream_gp_application" / "public_application_bridge.py"
    tree = ast.parse(path.read_text())
    forbidden = (
        "autoia.trusted", "autoia.enclosures", "autoia.polynomial",
        "autoia.semantic", "autoia.controller",
    )
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(forbidden):
                found.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(forbidden):
                    found.append(alias.name)
    if found:
        raise AssertionError(f"forbidden downstream GP application public-bridge imports: {found}")
    return {"forbidden_internal_imports": found, "pass": True}


def audit_scalar_arithmetic(cases=300, seed=20260830):
    mp.mp.dps = 100
    rng = np.random.default_rng(seed)
    checks = 0
    for _ in range(cases):
        a, b = sorted(rng.uniform(-2.0, 2.0, 2))
        c, d = sorted(rng.uniform(0.2, 2.0, 2))
        cert = CertificationContext()
        A = cert.interval(a, b)
        C = cert.interval(c, d)
        # H(x,y)=exp((x*y+1)/(y+2)); denominator is positive on this audit domain.
        H = cert.exp((A * C + 1.0) / (C + 2.0))
        for xv in (a, b, 0.5 * (a + b)):
            for yv in (c, d, 0.5 * (c + d)):
                exact = mp.e ** ((mp.mpf(float(xv)) * mp.mpf(float(yv)) + 1) /
                                 (mp.mpf(float(yv)) + 2))
                if not (mp.mpf(H.lower) <= exact <= mp.mpf(H.upper)):
                    raise AssertionError((a, b, c, d, H, exact))
                checks += 1
    return {"cases": cases, "high_precision_point_checks": checks, "pass": True}


def audit_downstream_gp_application():
    levels = {2: (1, 2, 4), 5: (2, 3, 5), 10: (3, 6, 9)}
    endpoint_checks = 0
    sample_checks = 0
    rows = []
    for d, levs in levels.items():
        state = load_state(d)
        for level in levs:
            lo, hi = representative_box(state, level)
            public = rigorous_downstream_gp_application_public_upper_one_box(lo, hi, state)
            reference = rigorous_downstream_gp_application_ei_bounds_one_box(lo, hi, state)
            if public.value != reference.ei.hi:
                raise AssertionError((d, level, public.value, reference.ei.hi))
            endpoint_checks += 1
            rng = np.random.default_rng(70000 + 100 * d + level)
            P = lo + (hi - lo) * rng.random((64, d))
            vals = point_ei(P, state.X_train, state.sigma_f2, state.length_scale,
                            state.L, state.alpha, state.y_best_scaled)
            if float(np.max(vals)) > public.value:
                raise AssertionError((d, level, float(np.max(vals)), public.value))
            sample_checks += len(vals)
            rows.append({"dimension": d, "level": level, "upper": public.value})
    return {
        "frozen_endpoint_matches": endpoint_checks,
        "sample_containment_checks": sample_checks,
        "rows": rows,
        "pass": True,
    }


def main():
    result = {
        "import_boundary": audit_import_boundary(),
        "scalar_arithmetic": audit_scalar_arithmetic(),
        "downstream_gp_application": audit_downstream_gp_application(),
    }
    out_json = ROOT / "STEP20C_APPLICATION_THEOREM_AUDIT.json"
    out_txt = ROOT / "STEP20C_APPLICATION_THEOREM_AUDIT.txt"
    out_json.write_text(json.dumps(result, indent=2))
    lines = [
        "AutoIA Step 20c application-theorem audit -- PASS",
        f"Forbidden downstream GP application public-bridge internal imports: {len(result['import_boundary']['forbidden_internal_imports'])}",
        f"High-precision certified-arithmetic checks: {result['scalar_arithmetic']['high_precision_point_checks']}",
        f"Frozen downstream GP application reference endpoint matches: {result['downstream_gp_application']['frozen_endpoint_matches']}",
        f"Sampled downstream GP application containment checks: {result['downstream_gp_application']['sample_containment_checks']}",
    ]
    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
