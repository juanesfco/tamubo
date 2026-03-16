from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from itertools import product
from typing import List, Optional, Iterable
from .interval_arithmetics import Bounds

Array = np.ndarray

@dataclass
class Box:
    """Axis-aligned hyperbox with (d,2) bounds [[lo,hi], ...]."""
    bounds: Array  # shape (d,2) with [lo, hi]
    sampled: bool
    active: bool = True
    ei: Optional[Bounds] = None


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

@dataclass
class Boxes:
    """Collection of Box objects with convenient aggregated views."""
    items: Array = field(default_factory=lambda: np.empty((0,), dtype=object))

    def __post_init__(self):
        # Accept iterables or arrays and normalize to 1D object array
        if isinstance(self.items, np.ndarray):
            if self.items.dtype != object:
                # e.g., array of Box but wrong dtype, or nested list
                self.items = np.array(list(self.items), dtype=object)
        else:
            # items is an iterable of Box
            self.items = np.array(list(self.items), dtype=object)

        # Ensure 1D
        if self.items.ndim != 1:
            self.items = self.items.reshape(-1).astype(object)

    # --- sequence-like behavior ---
    def __len__(self) -> int:
        return self.items.size

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, box: Box) -> None:
        """Append a single Box."""
        self.items = np.concatenate(
            [self.items, np.array([box], dtype=object)]
        )

    def extend(self, boxes: Iterable[Box]) -> None:
        """Append multiple Box instances."""
        arr = np.array(list(boxes), dtype=object)
        if arr.size > 0:
            self.items = np.concatenate([self.items, arr])

    # --- aggregated properties ---

    @property
    def bounds(self) -> Array:
        """Stacked bounds as (n_boxes, d, 2)."""
        if len(self) == 0:
            return np.empty((0, 0, 2), dtype=float)
        return np.stack([b.bounds for b in self.items], axis=0)

    @property
    def sampled(self) -> Array:
        """Boolean array (n_boxes,) indicating which boxes are sampled."""
        return np.array([b.sampled for b in self.items], dtype=bool)

    @property
    def active(self) -> Array:
        """Boolean array (n_boxes,) indicating which boxes are active."""
        return np.array([b.active for b in self.items], dtype=bool)

    @property
    def centers(self) -> Array:
        """Array of centers with shape (n_boxes, d)."""
        if len(self) == 0:
            return np.empty((0, 0), dtype=float)
        return np.stack([b.center for b in self.items], axis=0)

    @property
    def widths(self) -> Array:
        """Array of widths with shape (n_boxes, d)."""
        if len(self) == 0:
            return np.empty((0, 0), dtype=float)
        return np.stack([b.width for b in self.items], axis=0)

    @property
    def ei_hi(self) -> list[Optional[Bounds]]:
        """List of EI high bounds (or None) per box."""
        n = len(self)
        out = np.full(n, np.nan, dtype=float)
        for i, b in enumerate(self.items):
            if b.ei is not None:
                out[i] = b.ei.hi
        return out
    
    @property
    def ei_lo(self) -> list[Optional[Bounds]]:
        """List of EI low bounds (or None) per box."""
        n = len(self)
        out = np.full(n, np.nan, dtype=float)
        for i, b in enumerate(self.items):
            if b.ei is not None:
                out[i] = b.ei.lo
        return out

    @property
    def ei_array(self) -> Array:
        """
        Array of EI [lower, upper] for boxes where ei is not None.
        Missing entries are NaN. Shape: (n_boxes, 2).
        """
        n = len(self)
        out = np.full((n, 2), np.nan, dtype=float)
        for i, b in enumerate(self.items):
            if b.ei is not None:
                out[i, 0] = b.ei.lo
                out[i, 1] = b.ei.hi
        return out

def split_box(box: Box, split_type: str = "full", domain_width: Array | None = None) -> List[Box]:
    """
    Split `box` along EVERY axis at the midpoint, returning 2^d sub-boxes
    whose union equals the original box (overlaps only on boundaries).

    Returns
    -------
    children : list[Box] of length 2^d
    """
    b = box.bounds
    d = box.dim
    mid = box.center
    lo = b[:, 0]
    hi = b[:, 1]
    
    children: Boxes = Boxes()

    if split_type == "full":
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

    elif split_type == "centered":
        w = box.width
        if domain_width is None:
            domain_width = np.ones(d)
        w_dw = w / domain_width
        split_order = np.argsort(w_dw)[::-1]

        mid_box_bounds  = b.copy()

        for id in range(d):
            split_d = split_order[id]
            c = mid[split_d]
            l = lo[split_d]
            r = hi[split_d]
            
            ld, rd = c - l, r - c
            wc = 0.5 * min(ld, rd)

            lb = c - 0.5 * wc
            rb = c + 0.5 * wc

            left_box_bounds = mid_box_bounds.copy();  left_box_bounds[split_d, 1] = lb
            right_box_bounds= mid_box_bounds.copy();  right_box_bounds[split_d, 0] = rb
            mid_box_bounds[split_d, 0] = lb; mid_box_bounds[split_d, 1] = rb

            children.append(Box(bounds=left_box_bounds, sampled=box.sampled))
            children.append(Box(bounds=right_box_bounds, sampled=box.sampled))
                
        children.append(Box(bounds=mid_box_bounds, sampled=box.sampled))
    
    else:
        raise ValueError("split_type not supported")

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