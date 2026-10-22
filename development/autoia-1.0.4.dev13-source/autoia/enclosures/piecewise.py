"""Finite-union enclosure produced by top-level subdivision."""

from __future__ import annotations

from ..trusted import Interval


class PiecewiseEnclosure:
    """Finite union of already-certified scalar branch enclosures.

    Piecewise values are currently top-level outputs only.  Arbitrary arithmetic
    is intentionally disabled until branch-domain/path alignment is explicit.
    """

    representation_name = "piecewise"
    promotion_priority = -1
    supports_value_arithmetic = False

    def __init__(self, branches, backend, branch_metadata=None):
        self.branches = tuple(branches)
        if not self.branches:
            raise ValueError("PiecewiseEnclosure requires at least one branch.")
        self.backend = backend
        if branch_metadata is None:
            self.branch_metadata = tuple(None for _ in self.branches)
        else:
            self.branch_metadata = tuple(branch_metadata)
            if len(self.branch_metadata) != len(self.branches):
                raise ValueError("branch_metadata must align with branches.")

    def copy(self):
        return PiecewiseEnclosure(
            [branch.copy() for branch in self.branches], self.backend,
            branch_metadata=self.branch_metadata,
        )

    def range(self, method=None):
        intervals = []
        for branch in self.branches:
            r, _ = branch.range(method)
            intervals.append(r)
        lo = min(r.lo for r in intervals)
        hi = max(r.hi for r in intervals)
        return Interval(lo, hi), {
            "method": "piecewise-hull",
            "num_branches": len(self.branches),
            "branch_ranges": intervals,
        }

    def __repr__(self):
        r, _ = self.range()
        return (
            f"PiecewiseEnclosure(branches={len(self.branches)}, "
            f"hull=[{r.lo:.12g}, {r.hi:.12g}])"
        )


__all__ = ["PiecewiseEnclosure"]
