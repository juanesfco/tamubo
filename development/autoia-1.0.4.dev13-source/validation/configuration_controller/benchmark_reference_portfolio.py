"""Step 20a Phase-1 benchmark for the automatic configuration reference portfolio.

The benchmark is intentionally small enough to run routinely.  It combines two
synthetic mechanism-isolation cases with representative internal/external
engineering cases that previously motivated manual order or representation
choices.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from autoia import autoia, Interval, ConfigurationPolicy, __version__
from validation.engineering_corpus.cases import CASE_BY_NAME as INTERNAL
from validation.external_engineering.cases import CASE_BY_NAME as EXTERNAL


def exp_case(p):
    return np.exp(p[0])


def cancellation_case(p):
    x = p[0]
    y = x*x
    return np.array([y, y-y], dtype=object)


CASES = [
    ("synthetic", "exp_order", exp_case, ((-1.0, 1.0),), 4),
    ("synthetic", "shared_error_cancellation", cancellation_case, ((-1.0, 1.0),), 1),
    ("internal", "pressure_vessel_generator", INTERNAL["pressure_vessel_generator"].model,
     INTERNAL["pressure_vessel_generator"].bounds, 4),
    ("internal", "heat_exchanger_lmtd", INTERNAL["heat_exchanger_lmtd"].model,
     INTERNAL["heat_exchanger_lmtd"].bounds, 4),
    ("external", "fluids_churchill_friction", EXTERNAL["fluids_churchill_friction"].model,
     EXTERNAL["fluids_churchill_friction"].bounds, 4),
    ("external", "grheat_plane_instantaneous", EXTERNAL["grheat_plane_instantaneous"].model,
     EXTERNAL["grheat_plane_instantaneous"].bounds, 4),
]


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _run_case(source, name, model, bounds, baseline_order):
    box = [Interval(lo, hi) for lo, hi in bounds]
    t0 = time.perf_counter()
    baseline = autoia(model, box, order=baseline_order, representation="polynomial")
    baseline_time = time.perf_counter() - t0

    policy = ConfigurationPolicy(
        mode="reference", extra_orders=(5,),
        extra_representations=("generator",), max_attempts=3,
    )
    t0 = time.perf_counter()
    configured = autoia(
        model, box, order=baseline_order, representation="polynomial",
        configuration_policy=policy,
    )
    configured_time = time.perf_counter() - t0

    bw = np.asarray(baseline.width, dtype=float)
    cw = np.asarray(configured.width, dtype=float)
    ratios = np.divide(cw, bw, out=np.ones_like(cw), where=bw > 0.0)
    return {
        "source": source,
        "name": name,
        "baseline_order": baseline_order,
        "baseline_runtime_seconds": baseline_time,
        "configured_runtime_seconds": configured_time,
        "runtime_ratio": configured_time/max(baseline_time, np.finfo(float).tiny),
        "baseline_lower": baseline.lower,
        "baseline_upper": baseline.upper,
        "baseline_width": bw,
        "configured_lower": configured.lower,
        "configured_upper": configured.upper,
        "configured_width": cw,
        "width_ratio": ratios,
        "attempts": configured.configuration_diagnostics["attempts"],
        "selected_representation": configured.configuration_diagnostics["selected_representation"],
        "selected_order": configured.configuration_diagnostics["selected_order"],
        "nondominated_attempt_indices": configured.configuration_diagnostics["nondominated_attempt_indices"],
    }


def main():
    rows=[]
    for args in CASES:
        print(args[0], args[1], flush=True)
        rows.append(_run_case(*args))
    all_ratios=np.concatenate([np.asarray(r["width_ratio"]).reshape(-1) for r in rows])
    result={
        "autoia_version": __version__,
        "platform": {"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"numpy":np.__version__},
        "policy": {
            "candidates": "baseline polynomial/order-k; polynomial/order-5; generator/order-k, duplicate-free",
            "final_numeric_certificate": "componentwise intersection of all successful rigorous runs",
            "structural_result": "one whole result selected by enclosure dominance, baseline on incomparability",
            "purpose": "reference oracle for Step 20a; not yet the cheap release policy",
        },
        "summary": {
            "cases": len(rows),
            "outputs": int(all_ratios.size),
            "strictly_tighter_outputs": int(np.sum(all_ratios < 1.0)),
            "unchanged_outputs": int(np.sum(all_ratios == 1.0)),
            "wider_outputs": int(np.sum(all_ratios > 1.0)),
            "median_width_ratio": float(np.median(all_ratios)),
            "min_width_ratio": float(np.min(all_ratios)),
            "median_runtime_ratio": float(np.median([r["runtime_ratio"] for r in rows])),
        },
        "cases": rows,
    }
    Path("STEP20A_REFERENCE_PORTFOLIO_RESULTS.json").write_text(json.dumps(_jsonable(result),indent=2)+"\n")
    lines=[
        f"AutoIA {__version__} Step 20a Phase-1 reference portfolio",
        f"cases={result['summary']['cases']} outputs={result['summary']['outputs']}",
        f"tighter={result['summary']['strictly_tighter_outputs']} unchanged={result['summary']['unchanged_outputs']} wider={result['summary']['wider_outputs']}",
        f"median_width_ratio={result['summary']['median_width_ratio']:.6g}",
        f"min_width_ratio={result['summary']['min_width_ratio']:.6g}",
        f"median_runtime_ratio={result['summary']['median_runtime_ratio']:.6g}",
        "",
    ]
    for r in rows:
        lines.append(
            f"{r['source']:9s} {r['name']:32s} "
            f"width_ratio={np.array2string(np.asarray(r['width_ratio']), precision=5)} "
            f"runtime_ratio={r['runtime_ratio']:.3g} "
            f"structural={r['selected_representation']}/order-{r['selected_order']}"
        )
    Path("STEP20A_REFERENCE_PORTFOLIO_RESULTS.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
