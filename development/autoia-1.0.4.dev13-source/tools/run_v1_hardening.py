#!/usr/bin/env python3
"""Run the reproducible Version-1 proof-hardening campaign.

The runner is intentionally ordinary Python so the exact same command can be
used locally and by CI on Linux, macOS, and Windows.  It records the platform
fingerprint, command lines, return codes, durations, and complete stdout/stderr
for every audit.  A nonzero child return code makes the campaign fail closed.

Examples
--------
    python tools/run_v1_hardening.py --profile ci --output v1_hardening.json
    python tools/run_v1_hardening.py --profile full --output v1_hardening.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scipy
except ImportError:
    scipy = None
try:
    import mpmath
except ImportError:
    mpmath = None

from autoia import __version__
from autoia.trusted import audit_binary64_environment


def _cmd(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def command_plan(profile: str, include_tests: bool) -> list[tuple[str, list[str]]]:
    if profile not in {"ci", "full"}:
        raise ValueError(profile)

    plan: list[tuple[str, list[str]]] = []
    if include_tests:
        plan.append((
            "unit-tests",
            _cmd("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"),
        ))

    if profile == "ci":
        scalar = ("10000", "1000")
        dots = "1500"
        elementary = "50"
        power = "80"
        special = ("50", "6")
        array = ("6", "20")
        solve = ("2", "10")
        branch = ("2", "20")
        generators = ("2", "20")
        manybox = "4"
        hardening_reductions = "300"
    else:
        scalar = ("100000", "10000")
        dots = "15000"
        elementary = "2500"
        power = "1000"
        special = ("1000", "80")
        array = ("80", "250")
        solve = ("30", "150")
        branch = ("20", "250")
        generators = ("30", "250")
        manybox = "80"
        hardening_reductions = "2500"

    plan.extend([
        ("v1-adversarial", _cmd("tools/audit_v1_hardening.py", "--random-reductions", hardening_reductions)),
        ("scalar-binary64", _cmd(
            "tools/audit_scalar_binary64.py",
            "--random-point-ops", scalar[0],
            "--random-interval-ops", scalar[1],
        )),
        ("trusted-reductions", _cmd("tools/audit_reductions.py", "--dot-cases", dots, "--max-length", "129")),
        ("elementary", _cmd("tools/audit_elementary.py", "--cases", elementary, "--dps", "200")),
        ("power", _cmd("tools/audit_power.py", "--cases", power, "--dps", "180")),
        ("special-functions", _cmd(
            "tools/audit_special_functions.py", "--cases", special[0], "--boxes", special[1], "--dps", "180"
        )),
        ("domain-proofs", _cmd("tools/audit_domain_proofs.py")),
        ("array-semantics", _cmd("tools/audit_array_semantics.py", "--cases", array[0], "--samples", array[1])),
        ("scipy-compatibility", _cmd(
            "tools/audit_scipy_compatibility.py",
            "--cases", "3" if profile == "ci" else "20",
            "--samples", "10" if profile == "ci" else "100",
        )),
        ("general-solve", _cmd("tools/audit_general_solve.py", "--cases", solve[0], "--samples", solve[1])),
        ("path-branching", _cmd("tools/audit_path_branching.py", "--families", branch[0], "--samples", branch[1])),
        ("shared-generators", _cmd("tools/audit_shared_generators.py", "--families", generators[0], "--samples", generators[1])),
        ("generator-aware-solve", _cmd("tools/audit_generator_aware_solve.py", "--families", generators[0], "--samples", generators[1])),
        ("many-box", _cmd("tools/audit_manybox.py", "--boxes", manybox, "--workers", "2")),
        ("packed-ranging", _cmd("tools/audit_packed_polynomial_ranging.py")),
        ("packed-monotonicity", _cmd("tools/audit_packed_monotonicity.py")),
        ("packed-bernstein", _cmd("tools/audit_packed_bernstein.py")),
    ])
    return plan


def fingerprint() -> dict:
    env = audit_binary64_environment()
    return {
        "autoia": __version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": None if scipy is None else scipy.__version__,
        "mpmath": None if mpmath is None else mpmath.__version__,
        "byteorder": sys.byteorder,
        "binary64_environment": {
            "supported": env.supported,
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in env.checks
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "full"), default="ci")
    ap.add_argument("--output", type=Path, default=Path("V1_HARDENING_REPORT.json"))
    ap.add_argument("--skip-unit-tests", action="store_true")
    ap.add_argument("--timeout-per-command", type=float, default=600.0)
    args = ap.parse_args()

    report = {
        "checkpoint": __version__,
        "profile": args.profile,
        "status": "RUNNING",
        "platform": fingerprint(),
        "commands": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started_all = time.perf_counter()
    for name, command in command_plan(args.profile, not args.skip_unit_tests):
        print(f"[{name}] {' '.join(command)}", flush=True)
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=args.timeout_per_command,
            )
            item = {
                "name": name,
                "command": command,
                "returncode": proc.returncode,
                "duration_seconds": time.perf_counter() - started,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "status": "PASS" if proc.returncode == 0 else "FAIL",
            }
        except subprocess.TimeoutExpired as exc:
            item = {
                "name": name,
                "command": command,
                "returncode": None,
                "duration_seconds": time.perf_counter() - started,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "status": "TIMEOUT",
            }
        report["commands"].append(item)
        report["elapsed_seconds"] = time.perf_counter() - started_all
        report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["commands"]) else "FAIL"
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[{name}] {item['status']} ({item['duration_seconds']:.3f} s)", flush=True)
        if item["status"] != "PASS":
            if item["stdout"]:
                print(item["stdout"], file=sys.stderr)
            if item["stderr"]:
                print(item["stderr"], file=sys.stderr)
            return 1

    report["elapsed_seconds"] = time.perf_counter() - started_all
    report["status"] = "PASS"
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"V1 hardening campaign PASS in {report['elapsed_seconds']:.3f} s")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
