from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple

Array = np.ndarray

@dataclass
class Box:
    """Axis-aligned hyperbox with (d,2) bounds [[lo,hi], ...]."""
    bounds: Array  # shape (d,2)
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

def split_box(box: Box, axis: int) -> Tuple[Box, Box]:
    """Split a box in half along `axis`."""
    b = box.bounds.copy()
    mid = 0.5 * (b[axis, 0] + b[axis, 1])
    left = b.copy();  left[axis, 1] = mid
    right = b.copy(); right[axis, 0] = mid
    return Box(left, parent=None), Box(right, parent=None)

def corners(bounds: Array) -> Array:
    """Return all 2^d corners of the (d,2) bounds array as (2^d, d)."""
    d = bounds.shape[0]
    grid = np.array(
        np.meshgrid(*[(bounds[i, 0], bounds[i, 1]) for i in range(d)], indexing="ij")
    )
    return grid.reshape(d, -1).T
