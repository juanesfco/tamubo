#!/usr/bin/env python3
"""Run the pre-existing external engineering-code validation corpus.

This harness separates three questions:
1. Does the vendored external source execute normally with ordinary floats?
2. Can AutoIA certify the same source logic over an uncertainty box?
3. Do sampled ordinary evaluations lie inside the returned certificate as a
   compatibility sanity check?  Sampling is not the soundness proof.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import __version__, Interval, autoia, AutoIAError
from validation.external_engineering.cases import CASES


def _flatten(value):
    return np.asarray(value, dtype=float).reshape(-1)


def _sample_points(bounds, random_samples, rng):
    bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
    pts = [np.asarray([(lo + hi)*0.5 for lo, hi in bounds], dtype=float)]
    if (1 << len(bounds)) <= 128:
        for bits in itertools.product((0, 1), repeat=len(bounds)):
            pts.append(np.asarray([bounds[i][bit] for i, bit in enumerate(bits)], dtype=float))
    for _ in range(int(random_samples)):
        pts.append(np.asarray([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float))
    return pts


def _sample_sanity(case, result, random_samples, seed):
    rng = np.random.default_rng(seed)
    lo = np.asarray(result.lower, dtype=float).reshape(-1)
    hi = np.asarray(result.upper, dtype=float).reshape(-1)
    sample_lo = np.full(lo.shape, np.inf)
    sample_hi = np.full(hi.shape, -np.inf)
    for p in _sample_points(case.bounds, random_samples, rng):
        y = _flatten(case.model(p))
        if y.shape != lo.shape:
            raise AssertionError(f"{case.name}: numeric shape {y.shape} != certificate shape {lo.shape}")
        scale = np.maximum(1.0, np.abs(y))
        tol = 64.0*np.finfo(float).eps*scale
        if np.any(y < lo - tol) or np.any(y > hi + tol):
            raise AssertionError(f"{case.name}: ordinary sample escaped certificate")
        sample_lo = np.minimum(sample_lo, y)
        sample_hi = np.maximum(sample_hi, y)
    width = hi - lo
    sampled_width = sample_hi - sample_lo
    ratio = np.divide(sampled_width, width, out=np.zeros_like(width), where=width > 0.0)
    return {
        "sample_count": int(len(_sample_points(case.bounds, random_samples, np.random.default_rng(seed)))),
        "sample_min": sample_lo.tolist(),
        "sample_max": sample_hi.tolist(),
        "sample_width_to_certificate_width": ratio.tolist(),
    }


def _source_hashes():
    base = ROOT/'validation'/'external_engineering'/'external_sources'
    out = {}
    for path in sorted(base.glob('*.py')):
        data = path.read_bytes()
        out[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return out


def _run_case(case, samples, seed):
    mid = np.asarray([(lo + hi)*0.5 for lo, hi in case.bounds], dtype=float)
    t0 = time.perf_counter()
    numeric = _flatten(case.model(mid))
    numeric_seconds = time.perf_counter() - t0
    item = {
        "name": case.name,
        "project": case.project,
        "license": case.license,
        "provenance": case.provenance,
        "upstream_symbol": case.upstream_symbol,
        "description": case.description,
        "features": list(case.features),
        "expected": case.expected,
        "bounds": [list(x) for x in case.bounds],
        "order": case.order,
        "ordinary_numeric_midpoint": numeric.tolist(),
        "ordinary_numeric_seconds": numeric_seconds,
    }
    t0 = time.perf_counter()
    try:
        result = autoia(case.model, [Interval(lo, hi) for lo, hi in case.bounds], order=case.order)
    except AutoIAError as exc:
        item.update({
            "autoia_seconds": time.perf_counter() - t0,
            "observed": "exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        if case.expected == "certified":
            item["status"] = "FAIL"
            raise
        item["status"] = "PASS"
        return item

    item["autoia_seconds"] = time.perf_counter() - t0
    if case.expected != "certified":
        raise AssertionError(f"{case.name}: unexpectedly certified; expected {case.expected}")
    if not result.machine_rigorous:
        raise AssertionError(f"{case.name}: non-rigorous result")
    item.update({
        "observed": "certified",
        "machine_rigorous": True,
        "lower": np.asarray(result.lower, dtype=float).tolist(),
        "upper": np.asarray(result.upper, dtype=float).tolist(),
        "width": np.asarray(result.width, dtype=float).tolist(),
        "branch_count": len(getattr(result, "branch_results", ())) if hasattr(result, "branch_results") else 1,
        "representation_names": list(getattr(result, "representation_names", ())),
        "sample_sanity": _sample_sanity(case, result, samples, seed),
        "status": "PASS",
    })
    return item


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


def _write_text(report, path):
    lines = [
        f"AutoIA external engineering corpus: {report['checkpoint']}",
        f"Status: {report['status']}",
        f"Cases: {len(report['cases'])}",
        "",
    ]
    for item in report['cases']:
        lines.append(
            f"{item['name']}: {item['status']} project={item['project']} "
            f"observed={item['observed']} time={item['autoia_seconds']:.6f}s"
        )
    lines.extend([
        "",
        "External source note: executable function/method bodies are vendored for reproducibility;",
        "validation adapters only unpack the AutoIA input vector, insert fixed constants, or select outputs.",
        "Sampling note: ordinary floating-point samples are compatibility checks, not soundness proofs.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('EXTERNAL_ENGINEERING_RESULTS.json'))
    ap.add_argument('--text-output', type=Path, default=Path('EXTERNAL_ENGINEERING_RESULTS.txt'))
    ap.add_argument('--random-samples', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260829)
    args = ap.parse_args()
    report = {
        "checkpoint": __version__,
        "fingerprint": _fingerprint(),
        "source_hashes": _source_hashes(),
        "cases": [],
    }
    for i, case in enumerate(CASES):
        report['cases'].append(_run_case(case, args.random_samples, args.seed+i))
    report['status'] = 'PASS' if all(x['status'] == 'PASS' for x in report['cases']) else 'FAIL'
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_text(report, args.text_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
