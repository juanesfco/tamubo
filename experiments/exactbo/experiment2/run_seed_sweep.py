#!/usr/bin/env python3
"""Run experiment 2 repeatedly while sweeping random seeds."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "experiment_config.json"
DEFAULT_RUNNER_PATH = SCRIPT_DIR / "run_experiment.py"
RANDOM_SEED_PATTERN = re.compile(
    r'("random_seed"\s*:\s*)-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config in {path} must be a JSON object.")
    return config


def _write_seed(path: Path, seed: int) -> None:
    text = path.read_text(encoding="utf-8")
    updated_text, replacements = RANDOM_SEED_PATTERN.subn(
        rf"\g<1>{seed}",
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f'Could not update "random_seed" in {path}.')
    path.write_text(updated_text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update random_seed in experiment_config.json and run the experiment "
            "sequentially for a seed range."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER_PATH,
        help=f"Path to run_experiment.py (default: {DEFAULT_RUNNER_PATH}).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First seed in the sweep (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--end-seed",
        type=int,
        default=9,
        help="Last seed in the sweep (inclusive). Default: 9.",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=10.0,
        help="Seconds to sleep between runs. Default: 10.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = args.config.resolve()
    runner_path = args.runner.resolve()
    start_seed = int(args.start_seed)
    end_seed = int(args.end_seed)
    rest_seconds = float(args.rest_seconds)

    if start_seed > end_seed:
        raise ValueError(
            f"start-seed ({start_seed}) must be <= end-seed ({end_seed})."
        )
    if rest_seconds < 0:
        raise ValueError(f"rest-seconds must be >= 0, got {rest_seconds}.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    config = _load_config(config_path)
    if "random_seed" not in config:
        raise KeyError(f'Config file {config_path} is missing "random_seed".')

    for seed in range(start_seed, end_seed + 1):
        _write_seed(config_path, seed)
        print(f"[seed {seed}] random_seed updated in {config_path}")

        command = [sys.executable, str(runner_path), "--config", str(config_path)]
        print(f"[seed {seed}] starting experiment")
        subprocess.run(command, check=True, cwd=runner_path.parent)
        print(f"[seed {seed}] experiment finished")

        if seed < end_seed and rest_seconds > 0:
            print(f"[seed {seed}] sleeping for {rest_seconds:.1f}s")
            time.sleep(rest_seconds)

    print(f"Completed seeds {start_seed} through {end_seed}.")


if __name__ == "__main__":
    main()
