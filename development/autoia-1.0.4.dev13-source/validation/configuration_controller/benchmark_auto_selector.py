"""Step 20a Phase-2 benchmark: cheap selector versus baseline/reference oracle."""
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


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _run_case(source, name, model, bounds, baseline_order):
    box = [Interval(lo, hi) for lo, hi in bounds]
    baseline, tb = _timed(lambda: autoia(
        model, box, order=baseline_order, representation="polynomial"
    ))
    automatic, ta = _timed(lambda: autoia(
        model, box, order=baseline_order, representation="polynomial",
        configuration_policy=ConfigurationPolicy(mode="auto"),
    ))
    reference, tr = _timed(lambda: autoia(
        model, box, order=baseline_order, representation="polynomial",
        configuration_policy=ConfigurationPolicy(mode="reference"),
    ))

    bw = np.asarray(baseline.width, dtype=float)
    aw = np.asarray(automatic.width, dtype=float)
    rw = np.asarray(reference.width, dtype=float)
    auto_ratio = np.divide(aw, bw, out=np.ones_like(aw), where=bw > 0.0)
    ref_ratio = np.divide(rw, bw, out=np.ones_like(rw), where=bw > 0.0)
    reference_gain = np.maximum(0.0, bw-rw)
    automatic_gain = np.maximum(0.0, bw-aw)
    gain_capture = np.divide(
        automatic_gain, reference_gain,
        out=np.ones_like(automatic_gain), where=reference_gain > 0.0,
    )
    return {
        "source": source, "name": name, "baseline_order": baseline_order,
        "baseline_runtime_seconds": tb,
        "automatic_runtime_seconds": ta,
        "reference_runtime_seconds": tr,
        "automatic_runtime_ratio": ta/max(tb, np.finfo(float).tiny),
        "reference_runtime_ratio": tr/max(tb, np.finfo(float).tiny),
        "baseline_width": bw, "automatic_width": aw, "reference_width": rw,
        "automatic_width_ratio": auto_ratio,
        "reference_width_ratio": ref_ratio,
        "reference_gain": reference_gain,
        "automatic_gain": automatic_gain,
        "gain_capture": gain_capture,
        "automatic_attempts": automatic.configuration_diagnostics["attempts"],
        "reference_attempts": reference.configuration_diagnostics["attempts"],
        "automatic_selector_signals": automatic.configuration_diagnostics["selector_signals"],
    }


def main():
    rows=[]
    for args in CASES:
        print(args[1], flush=True)
        rows.append(_run_case(*args))
    auto_ratios=np.concatenate([np.asarray(r["automatic_width_ratio"]).reshape(-1) for r in rows])
    ref_ratios=np.concatenate([np.asarray(r["reference_width_ratio"]).reshape(-1) for r in rows])
    captures=np.concatenate([np.asarray(r["gain_capture"]).reshape(-1) for r in rows])
    ref_gains=np.concatenate([np.asarray(r["reference_gain"]).reshape(-1) for r in rows])
    auto_gains=np.concatenate([np.asarray(r["automatic_gain"]).reshape(-1) for r in rows])
    result={
        "autoia_version": __version__,
        "platform": {"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"numpy":np.__version__},
        "selector": {
            "higher_order_trigger": "nonzero remainder width >= retained polynomial termwise width, except exact-zero shared-error recombination output",
            "generator_trigger": "shared approximation ancestry recombines in an output whose retained physical-polynomial variation is exactly zero and remainder is nonzero",
            "fitted_weights": False,
            "proof_role": "controller only; every attempted configuration is independently rigorous",
        },
        "summary": {
            "cases": len(rows),
            "outputs": int(auto_ratios.size),
            "automatic_tighter_outputs": int(np.sum(auto_ratios < 1.0)),
            "automatic_unchanged_outputs": int(np.sum(auto_ratios == 1.0)),
            "automatic_wider_outputs": int(np.sum(auto_ratios > 1.0)),
            "reference_tighter_outputs": int(np.sum(ref_ratios < 1.0)),
            "exact_reference_width_matches": int(np.sum(aw_eq := np.isclose(auto_ratios, ref_ratios, rtol=0.0, atol=0.0))),
            "median_automatic_runtime_ratio": float(np.median([r["automatic_runtime_ratio"] for r in rows])),
            "median_reference_runtime_ratio": float(np.median([r["reference_runtime_ratio"] for r in rows])),
            "median_automatic_width_ratio": float(np.median(auto_ratios)),
            "median_reference_width_ratio": float(np.median(ref_ratios)),
            "total_reference_gain_numeric": float(np.sum(ref_gains)),
            "total_automatic_gain_numeric": float(np.sum(auto_gains)),
            "gain_capture_per_output": captures,
        },
        "cases": rows,
    }
    Path("STEP20A_AUTO_SELECTOR_RESULTS.json").write_text(json.dumps(_jsonable(result),indent=2)+"\n")
    lines=[
        f"AutoIA {__version__} Step 20a Phase-2 cheap selector",
        f"cases={result['summary']['cases']} outputs={result['summary']['outputs']}",
        f"auto tighter={result['summary']['automatic_tighter_outputs']} unchanged={result['summary']['automatic_unchanged_outputs']} wider={result['summary']['automatic_wider_outputs']}",
        f"reference tighter={result['summary']['reference_tighter_outputs']}",
        f"median auto runtime/baseline={result['summary']['median_automatic_runtime_ratio']:.6g}",
        f"median reference runtime/baseline={result['summary']['median_reference_runtime_ratio']:.6g}",
        "",
    ]
    for r in rows:
        attempts=[f"{a['representation']}/o{a['order']}:{a.get('reason','')}" for a in r['automatic_attempts']]
        lines.append(
            f"{r['source']:9s} {r['name']:32s} "
            f"auto_ratio={np.array2string(np.asarray(r['automatic_width_ratio']), precision=6)} "
            f"ref_ratio={np.array2string(np.asarray(r['reference_width_ratio']), precision=6)} "
            f"auto_time={r['automatic_runtime_ratio']:.3g} ref_time={r['reference_runtime_ratio']:.3g} "
            f"attempts={attempts}"
        )
    Path("STEP20A_AUTO_SELECTOR_RESULTS.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
