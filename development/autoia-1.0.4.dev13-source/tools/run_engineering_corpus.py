#!/usr/bin/env python3
"""Run the AutoIA Version-1 engineering validation corpus.

The model files under ``validation/engineering_corpus/models`` contain no
AutoIA imports.  This harness supplies uncertainty boxes, executes AutoIA,
checks ordinary floating-point sample evaluations against the certified hull as
a compatibility sanity check, records diagnostics, and verifies expected
fail-closed/unsupported cases.

Sample containment is *not* a proof of interval soundness.  Machine-rigorous
soundness remains established by AutoIA's trusted-kernel and structural proof
audits.  The corpus answers a different release question: can ordinary
engineering-style Python execute unchanged and fail intelligibly when it cannot?
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import (
    __version__, Interval, autoia, prepare,
    AutoIAError, CertificationError, UnsupportedOperationError,
)
from validation.engineering_corpus.cases import CASES, CASE_BY_NAME
from validation.engineering_corpus.models.cantilever_beam import cantilever_beam


def _flatten_numeric(value):
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        if np.any(np.abs(arr.imag) > 1e-12):
            raise ValueError("numeric model returned genuinely complex output")
        arr = arr.real
    return np.asarray(arr, dtype=float).reshape(-1)


def _sample_points(bounds, random_samples, rng, max_corners=256):
    bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
    points = [np.asarray([(lo + hi) * 0.5 for lo, hi in bounds], dtype=float)]
    if (1 << len(bounds)) <= max_corners:
        for bits in itertools.product((0, 1), repeat=len(bounds)):
            points.append(np.asarray([bounds[i][bit] for i, bit in enumerate(bits)], dtype=float))
    for _ in range(int(random_samples)):
        points.append(np.asarray([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float))
    return points


def _sample_sanity(case, result, random_samples, seed):
    rng = np.random.default_rng(seed)
    points = _sample_points(case.bounds, random_samples, rng)
    lo = np.asarray(result.lower, dtype=float).reshape(-1)
    hi = np.asarray(result.upper, dtype=float).reshape(-1)
    sample_lo = np.full(lo.shape, np.inf)
    sample_hi = np.full(hi.shape, -np.inf)
    worst_lower = 0.0
    worst_upper = 0.0
    for point in points:
        y = _flatten_numeric(case.model(point))
        if y.shape != lo.shape:
            raise AssertionError(f"sample output shape {y.shape} != certified shape {lo.shape}")
        sample_lo = np.minimum(sample_lo, y)
        sample_hi = np.maximum(sample_hi, y)
        # The ordinary float evaluation is not the exact-real mathematical
        # reference used by the proof.  Permit a small floating sanity margin.
        scale = np.maximum(1.0, np.abs(y))
        tol = 64.0 * np.finfo(float).eps * scale
        worst_lower = max(worst_lower, float(np.max(lo - y - tol)))
        worst_upper = max(worst_upper, float(np.max(y - hi - tol)))
    if worst_lower > 0.0 or worst_upper > 0.0:
        raise AssertionError(
            f"sample escaped certificate: lower_excess={worst_lower}, upper_excess={worst_upper}"
        )
    certified_width = hi - lo
    sample_width = sample_hi - sample_lo
    ratio = np.divide(
        sample_width,
        certified_width,
        out=np.zeros_like(sample_width),
        where=certified_width > 0.0,
    )
    return {
        "sample_count": len(points),
        "sample_min": sample_lo.tolist(),
        "sample_max": sample_hi.tolist(),
        "sample_width_to_certificate_width": ratio.tolist(),
    }


def _run_case(case, random_samples, seed):
    item = {
        "name": case.name,
        "description": case.description,
        "features": list(case.features),
        "expected": case.expected,
        "order": case.order,
        "representation": case.representation,
        "input_bounds": [list(pair) for pair in case.bounds],
    }

    midpoint = np.asarray([(lo + hi) * 0.5 for lo, hi in case.bounds], dtype=float)
    numeric_started = time.perf_counter()
    numeric_value = _flatten_numeric(case.model(midpoint))
    item["ordinary_numeric_midpoint"] = numeric_value.tolist()
    item["ordinary_numeric_seconds"] = time.perf_counter() - numeric_started

    if case.invalid_witness is not None:
        with np.errstate(all="ignore"):
            witness_value = np.asarray(case.model(np.asarray(case.invalid_witness, dtype=float)))
        item["invalid_witness"] = list(case.invalid_witness)
        item["invalid_witness_output"] = [
            None if not np.isfinite(float(x)) else float(x)
            for x in np.asarray(witness_value, dtype=float).reshape(-1)
        ]
        if np.all(np.isfinite(np.asarray(witness_value, dtype=float))):
            raise AssertionError("declared invalid witness unexpectedly produced only finite values")

    intervals = [Interval(lo, hi) for lo, hi in case.bounds]
    started = time.perf_counter()
    try:
        result = autoia(
            case.model,
            intervals,
            order=case.order,
            representation=case.representation,
        )
    except AutoIAError as exc:
        item["autoia_seconds"] = time.perf_counter() - started
        item["observed"] = "exception"
        item["exception_type"] = type(exc).__name__
        item["exception_message"] = str(exc)
        item["exception_context"] = getattr(exc, "context", None)
        if case.expected == "certified":
            raise
        if case.expected_exception and type(exc).__name__ != case.expected_exception:
            raise AssertionError(
                f"{case.name}: expected {case.expected_exception}, got {type(exc).__name__}"
            )
        if case.expected_message and case.expected_message.lower() not in str(exc).lower():
            raise AssertionError(
                f"{case.name}: expected message containing {case.expected_message!r}, got {exc!s}"
            )
        item["status"] = "PASS"
        return item

    item["autoia_seconds"] = time.perf_counter() - started
    item["observed"] = "certified"
    if case.expected != "certified":
        raise AssertionError(f"{case.name}: expected {case.expected}, but AutoIA certified it")
    if not result.machine_rigorous:
        raise AssertionError(f"{case.name}: result is not machine-rigorous")

    item.update({
        "machine_rigorous": bool(result.machine_rigorous),
        "result_type": type(result).__name__,
        "lower": np.asarray(result.lower, dtype=float).tolist(),
        "upper": np.asarray(result.upper, dtype=float).tolist(),
        "width": np.asarray(result.width, dtype=float).tolist(),
        "representation_names": list(getattr(result, "representation_names", ())),
        "certificate_count": len(getattr(result, "certificates", ())),
        "branch_count": len(getattr(result, "branch_results", ())) if hasattr(result, "branch_results") else 1,
    })
    item["sample_sanity"] = _sample_sanity(case, result, random_samples, seed)
    item["status"] = "PASS"
    return item


def _run_prepared_beam_sweep(random_samples, seed):
    case = CASE_BY_NAME["cantilever_beam"]
    parent = case.bounds
    children = [
        ((940.0, 1060.0), (1.94, 2.06), (0.049, 0.051), (0.097, 0.103), (194e9, 206e9), (7700.0, 7900.0)),
        ((960.0, 1040.0), (1.96, 2.04), (0.0492, 0.0508), (0.098, 0.102), (196e9, 204e9), (7725.0, 7875.0)),
        ((980.0, 1020.0), (1.98, 2.02), (0.0495, 0.0505), (0.099, 0.101), (198e9, 202e9), (7750.0, 7850.0)),
    ]
    boxes = [parent, *children, parent]
    session = prepare(cantilever_beam, order=case.order, representation=case.representation)
    interval_boxes = [[Interval(lo, hi) for lo, hi in box] for box in boxes]
    started = time.perf_counter()
    results = session.certify_many(interval_boxes, workers=1, parallel="serial")
    elapsed = time.perf_counter() - started
    if not all(r.machine_rigorous for r in results):
        raise AssertionError("prepared beam sweep returned a non-rigorous result")
    if not np.array_equal(results[0].lower, results[-1].lower) or not np.array_equal(results[0].upper, results[-1].upper):
        raise AssertionError("exact repeated prepared box did not reproduce its cached certificate")
    # Each interior child must still pass the same ordinary numeric sanity check.
    child_checks = []
    for i, (bounds, result) in enumerate(zip(boxes, results)):
        proxy_case = type(case)(
            name=f"prepared-beam-{i}", model=case.model, bounds=tuple(bounds),
            order=case.order, representation=case.representation, expected="certified",
            features=case.features, description=case.description,
        )
        child_checks.append(_sample_sanity(proxy_case, result, max(8, random_samples // 4), seed + 1000 + i))
    stats = session.stats
    if stats.engineering_model_executions >= len(boxes):
        raise AssertionError("prepared sweep failed to reuse any certificate work")
    return {
        "status": "PASS",
        "model": "cantilever_beam",
        "box_count": len(boxes),
        "elapsed_seconds": elapsed,
        "cache_info": [getattr(r, "cache_info", None) for r in results],
        "session_stats": {
            "model_preparations": stats.model_preparations,
            "certify_requests": stats.certify_requests,
            "engineering_model_executions": stats.engineering_model_executions,
            "exact_cache_hits": stats.exact_cache_hits,
            "restricted_cache_hits": stats.restricted_cache_hits,
            "cache_misses": stats.cache_misses,
            "cache_entries": stats.cache_entries,
        },
        "sample_checks": child_checks,
    }


def _fingerprint():
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    return {
        "autoia": __version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy_version,
    }


def _write_text_summary(report, path):
    lines = [
        f"AutoIA engineering corpus checkpoint: {report['checkpoint']}",
        f"Status: {report['status']}",
        f"Cases: {len(report['cases'])}",
        "",
    ]
    for item in report["cases"]:
        detail = item.get("result_type", item.get("exception_type", ""))
        lines.append(
            f"{item['name']}: {item['status']} expected={item['expected']} observed={item['observed']} {detail} "
            f"time={item['autoia_seconds']:.6f}s"
        )
    p = report["prepared_beam_sweep"]
    lines.extend([
        "",
        f"prepared_beam_sweep: {p['status']} boxes={p['box_count']} "
        f"model_executions={p['session_stats']['engineering_model_executions']} "
        f"restricted_hits={p['session_stats']['restricted_cache_hits']} "
        f"exact_hits={p['session_stats']['exact_cache_hits']}",
        "",
        "Sampling note: sampled ordinary floating-point values are compatibility sanity checks only; they are not the soundness proof.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("ENGINEERING_CORPUS_RESULTS.json"))
    ap.add_argument("--text-output", type=Path, default=Path("ENGINEERING_CORPUS_RESULTS.txt"))
    ap.add_argument("--random-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    report = {
        "checkpoint": __version__,
        "status": "RUNNING",
        "platform": _fingerprint(),
        "sampling_is_proof": False,
        "cases": [],
    }
    started = time.perf_counter()
    try:
        for i, case in enumerate(CASES):
            print(f"[{case.name}] expected={case.expected}", flush=True)
            item = _run_case(case, args.random_samples, args.seed + i)
            report["cases"].append(item)
            print(f"[{case.name}] PASS", flush=True)
        print("[prepared_beam_sweep]", flush=True)
        report["prepared_beam_sweep"] = _run_prepared_beam_sweep(args.random_samples, args.seed)
        print("[prepared_beam_sweep] PASS", flush=True)
        report["status"] = "PASS"
        return_code = 0
    except Exception as exc:
        report["status"] = "FAIL"
        report["runner_exception_type"] = type(exc).__name__
        report["runner_exception_message"] = str(exc)
        return_code = 1
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_text_summary(report, args.text_output)
    print(f"Engineering corpus {report['status']} in {report['elapsed_seconds']:.3f}s")
    print(f"JSON: {args.output}")
    print(f"Text: {args.text_output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
