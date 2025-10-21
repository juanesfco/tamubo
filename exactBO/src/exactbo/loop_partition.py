from __future__ import annotations
import heapq
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

from .partition import Box, split_box, corners
from .ei import expected_improvement
from .ei_bounds import ei_bounds_from_mu_sigma_intervals, EIBounds

Array = np.ndarray

@dataclass(order=True)
class _PQItem:
    priority: float  # negative of EI upper (max-heap behavior)
    idx: int

class PartitionMaxEISearch:
    """
    Branch-and-bound search for max EI over partitioned boxes WITHOUT updating the GP.

    Assumes `model.predict(X, return_std=True) -> (mu, std)` with (n,1) arrays.
    """

    def __init__(self, model, f_best: float, init_boxes: List[Box]):
        self.model = model
        self.f_best = float(f_best)
        self.boxes: List[Box] = list(init_boxes)
        self.heap: List[_PQItem] = []
        self.incumbent_ei: float = 0.0
        self.best_x: Array | None = None
        self._initialized = False

    def _mu_sigma_intervals(self, box: Box) -> Tuple[float, float, float, float]:
        """Crude but safe: evaluate μ,σ on corners and take min/max."""
        Xc = corners(box.bounds)
        mu, std = self.model.predict(Xc, return_std=True)
        mu = mu.reshape(-1)
        var = (std.reshape(-1)) ** 2
        return float(mu.min()), float(mu.max()), float(var.min()), float(var.max())

    def _push(self, idx: int):
        mu_lo, mu_hi, v_lo, v_hi = self._mu_sigma_intervals(self.boxes[idx])
        eb = ei_bounds_from_mu_sigma_intervals(mu_lo, mu_hi, v_lo, v_hi, self.f_best)
        heapq.heappush(self.heap, _PQItem(priority=-eb.upper, idx=idx))

    def initialize(self):
        for i in range(len(self.boxes)):
            self._push(i)
        self._initialized = True

    def _choose_axis(self, box: Box) -> int:
        """Split along the widest dimension (simple heuristic)."""
        return int(np.argmax(box.width))

    def run(self, max_iters: int = 200):
        if not self._initialized:
            self.initialize()

        it = 0
        while self.heap and it < max_iters:
            it += 1
            item = heapq.heappop(self.heap)
            box = self.boxes[item.idx]

            # Midpoint EI as a cheap eval
            x_mid = box.center[None, :]
            mu_mid, std_mid = self.model.predict(x_mid, return_std=True)
            ei_mid = float(expected_improvement(mu_mid, std_mid, self.f_best))
            if ei_mid > self.incumbent_ei:
                self.incumbent_ei = ei_mid
                self.best_x = x_mid

            # Subdivide
            axis = self._choose_axis(box)
            left, right = split_box(box, axis)
            li = len(self.boxes); self.boxes.append(left)
            ri = len(self.boxes); self.boxes.append(right)
            self._push(li)
            self._push(ri)

        return self.best_x, self.incumbent_ei
