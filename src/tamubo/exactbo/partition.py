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

    def __post_init__(self):
        # 1) coerce to numpy
        b = np.asarray(self.bounds, dtype=float)

        # 2) shape checks
        if b.ndim != 2:
            raise ValueError(f"Box.bounds must be 2D (d,2), got shape {b.shape}")
        if b.shape[1] != 2:
            raise ValueError(f"Box.bounds second dim must be 2 (lo, hi), got {b.shape}")

        # 3) optional: lo <= hi
        lo = b[:, 0]
        hi = b[:, 1]
        if not np.all(lo <= hi):
            raise ValueError("Box.bounds must satisfy lo <= hi for all dimensions")

        # if everything is ok, assign back
        self.bounds = b

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def center(self) -> Array:
        return 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])

    @property
    def width(self) -> Array:
        return self.bounds[:, 1] - self.bounds[:, 0]

def split_box(box: Box) -> List[Box]:
    """
    Split `box` along EVERY axis at the midpoint, returning 2^d sub-boxes
    whose union equals the original box (overlaps only on boundaries).

    Returns
    -------
    children : list[Box] of length 2^d
    """
    b = box.bounds
    d = box.dim
    lo = b[:, 0]
    hi = b[:, 1]
    mid = 0.5 * (lo + hi)

    children: List[Box] = []
    # Each bit in pattern selects left(0)=[lo,mid] or right(1)=[mid,hi] for that axis
    for pattern in product((0, 1), repeat=d):
        child = np.empty_like(b)
        # Mask where bit == 0
        mask = np.array(pattern) == 0
        # New lo: Old lo values on axes where bit=0, mid value on axes where bit=1
        child[:, 0] = np.where(mask, lo, mid)
        # New hi: mid values on axes where bit=0, old hi value on axes where bit=1
        child[:, 1] = np.where(mask, mid, hi)

        children.append(Box(bounds=child, sampled=box.sampled))

    return children

def hypermask(boxes_bounds: np.ndarray, X: np.ndarray):
        """
        Return a boolean mask marking which points in X lie inside at least one
        hyperbox from `boxes_bounds`. Bounds are inclusive.

        Parameters
        ----------
        boxes_bounds : array_like, shape (n, d, 2) or (d, 2)
            n hyperboxes in d dimensions. For each box, boxes[i, 0] is the left
            bound in dimension i and boxes[i, 1] is the right bound.
        X : array_like, shape (k, d)
            k points in the same d-dimensional space.

        Returns
        -------
        mask : ndarray, shape (k,), dtype=bool
            True if the point lies in any hyperbox (inclusive bounds).
        """
        boxes_bounds = np.asarray(boxes_bounds)
        X = np.asarray(X)

        if boxes_bounds.ndim != 3 or boxes_bounds.shape[-1] != 2:
            raise ValueError("`boxes` must have shape (n, d, 2).")
        if X.ndim != 2 or X.shape[1] != boxes_bounds.shape[1]:
            raise ValueError(f"`X` must have shape (k, {boxes_bounds.shape[1]}).")

        left, right = boxes_bounds[..., 0], boxes_bounds[..., 1]  # (n, d) each

        ge_left = X[:, None, :] >= left[None, :, :]
        le_right = X[:, None, :] <= right[None, :, :]

        inside_per_box = np.logical_and(ge_left, le_right).all(axis=-1)  # (k, n)
        return inside_per_box.any(axis=1)