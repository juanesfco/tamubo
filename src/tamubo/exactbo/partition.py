from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from itertools import product
from typing import List

Array = np.ndarray

@dataclass
class Box:
    """Axis-aligned hyperbox with (d,2) bounds [[lo,hi], ...]."""
    bounds: Array  # shape (d,2) with [lo, hi]
    sampled: bool
    active: bool = True
    parent: int | None = None

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def center(self) -> Array:
        return 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])

    @property
    def width(self) -> Array:
        return self.bounds[:, 1] - self.bounds[:, 0]

def split_box(box: Box, axis: int) -> List[Box]:
    """
    Split `box` along EVERY axis at the midpoint, returning 2^d sub-boxes
    whose union equals the original box (overlaps only on boundaries).

    Returns
    -------
    children : list[Box] of length 2^d
    """
    b = np.asarray(box.bounds, dtype=float)
    d = box.dim
    lo = b[:, 0]
    hi = b[:, 1]
    mid = 0.5 * (lo + hi)

    children: List[Box] = []
    # Each bit in pattern selects left(0)=[lo,mid] or right(1)=[mid,hi] for that axis
    for pattern in product((0, 1), repeat=d):
        child = np.empty_like(b)
        # left half on axes where bit=0, right half where bit=1
        left_mask = np.array(pattern) == 0
        right_mask = ~left_mask

        child[:, 0] = np.where(left_mask, lo, mid)  # new lo
        child[:, 1] = np.where(left_mask, mid, hi)  # new hi
        children.append(Box(bounds=child, parent=None))

    return children

def corners(bounds: Array) -> Array:
    """Return all 2^d corners of the (d,2) bounds array as (2^d, d)."""
    d = bounds.shape[0]
    grid = np.array(
        np.meshgrid(*[(bounds[i, 0], bounds[i, 1]) for i in range(d)], indexing="ij")
    )
    return grid.reshape(d, -1).T
