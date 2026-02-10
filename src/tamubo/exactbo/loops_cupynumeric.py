"""Compatibility wrapper for vectorized cupynumeric loops."""

from .vectorized_loop import exactbo_loop_cupynumeric, partition_loop_cupynumeric

# Backward-compatible aliases used by development scripts.
partition_loop = partition_loop_cupynumeric
exactbo_loop = exactbo_loop_cupynumeric

__all__ = [
    "partition_loop_cupynumeric",
    "exactbo_loop_cupynumeric",
    "partition_loop",
    "exactbo_loop",
]
