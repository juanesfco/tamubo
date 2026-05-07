"""Small cuPyNumeric matrix multiplication workload for Nsight Systems."""

from __future__ import annotations

import argparse
import time

import cupynumeric as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple dense matrix multiplication with cuPyNumeric."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Matrix dimension N for multiplying two NxN matrices.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Timed matrix multiplication iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Untimed warmup iterations before profiling measurements.",
    )
    return parser.parse_args()


def create_matrices(size: int):
    a = np.ones((size, size), dtype=np.float32)
    b = np.full((size, size), 2.0, dtype=np.float32)
    return a, b


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")

    a, b = create_matrices(args.size)

    for _ in range(args.warmup):
        c = a @ b
        float(np.sum(c))

    start = time.perf_counter()
    checksum = 0.0
    for _ in range(args.iterations):
        c = a @ b
        checksum = float(np.sum(c))
    elapsed = time.perf_counter() - start

    flops = 2 * args.size**3 * args.iterations
    print(f"matrix_size={args.size}x{args.size}")
    print(f"iterations={args.iterations}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"estimated_gflops={flops / elapsed / 1e9:.3f}")
    print(f"checksum={checksum:.6f}")


if __name__ == "__main__":
    main()
